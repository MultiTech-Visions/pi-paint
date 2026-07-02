"""Mesh — units find each other and share one world.

Set several boxes along a fence line, and they should behave like one
long canvas: a fish entering the left edge of the property swims out
the right edge twenty projectors later, crossing every seam in sync.

The trick that makes this cheap enough for a pile of Pis: **nothing
streams frames**.  Units share three small things —

* a *world layout*: each unit occupies a slice of a common horizontal
  strip (ordered by its `position` setting, ties broken by node id);
* a *world clock*: the leader (lowest node id) broadcasts its
  monotonic time; followers keep a smoothed offset (LAN one-way
  latency ~ms, far below what the eye can catch on drifting light);
* a *show state*: seed + mood, broadcast by the leader.

Because every performer's trajectory is a pure function of
(seed, world time), each unit independently computes identical
positions and renders only its own slice.  Sync without streaming.

Transport: UDP broadcast JSON on one port.  Pure stdlib + threads,
no Qt — testable headless.  (Automatic *overlap* discovery via
camera dual-calibration is the planned next step; today adjacent
units are ordered by their `position` number and butt edge-to-edge,
with a manual `overlap_px` trim.)
"""

import json
import socket
import threading
import time
import uuid

import numpy as np


MAGIC = "PIPAINT1"
PEER_TIMEOUT = 6.0
HELLO_INTERVAL = 1.0


class MeshNode:
    def __init__(self, port=45454, broadcast="255.255.255.255",
                 position=0, width_px=640, overlap_px=0, node_id=None):
        self.node_id = node_id or uuid.uuid4().hex[:10]
        self.port = port
        self.broadcast = broadcast
        self.position = int(position)
        self.width_px = float(width_px)
        self.overlap_px = float(overlap_px)

        # peers: id -> {position, width, last_seen}
        self.peers = {}
        self._lock = threading.Lock()
        self._clock_offset = 0.0        # leader_time - local_time
        self._show = {"seed": 1234, "mood": None, "tempo": None,
                      "video": None}

        # Measured relations from dual calibration:
        # (observer_id, flasher_id) -> {ox, oy, scale, conf}
        # meaning: flasher's canvas origin sits at (ox, oy) in the
        # observer's canvas coordinates.
        self.relations = {}
        self._dcal_in = []              # incoming dcal protocol messages
        self._rel_tick = 0

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self._sock.bind(("", port))
        self._sock.settimeout(0.5)

        self._running = True
        self._rx = threading.Thread(target=self._rx_loop, daemon=True)
        self._tx = threading.Thread(target=self._tx_loop, daemon=True)
        self._rx.start()
        self._tx.start()

    # ── Public API ──────────────────────────────────────────────────────

    def stop(self):
        self._running = False
        self._rx.join(timeout=2)
        self._tx.join(timeout=2)
        self._sock.close()

    @property
    def is_leader(self):
        with self._lock:
            ids = [self.node_id] + list(self.peers.keys())
        return self.node_id == min(ids)

    def peer_count(self):
        with self._lock:
            return len(self.peers)

    def now(self):
        """Shared world time (the leader's monotonic clock)."""
        return time.monotonic() + self._clock_offset

    def set_show(self, seed=None, mood=None, tempo=None, video=None):
        """Leader: publish show parameters to the mesh."""
        if seed is not None:
            self._show["seed"] = int(seed)
        if mood is not None:
            self._show["mood"] = mood
        if tempo is not None:
            self._show["tempo"] = tempo
        if video is not None:
            self._show["video"] = video    # "" clears

    def show_state(self):
        return dict(self._show)

    # ── Dual calibration transport + relations ──────────────────────────

    def send_dcal(self, payload):
        """Broadcast a dual-calibration protocol message."""
        self._send({"t": "dcal", **payload})

    def drain_dcal(self):
        """Fetch queued incoming dcal messages (called from the GUI side)."""
        with self._lock:
            msgs, self._dcal_in = self._dcal_in, []
        return msgs

    def add_relation(self, obs_id, flasher_id, rel):
        """Store a measured relation and share it with the mesh."""
        entry = {"ox": float(rel["ox"]), "oy": float(rel["oy"]),
                 "scale": float(rel.get("scale", 1.0)),
                 "rot": float(rel.get("rot", 0.0)),
                 "conf": float(rel.get("conf", 0.0))}
        with self._lock:
            self.relations[(obs_id, flasher_id)] = entry
        self._broadcast_relations()

    def clear_relations(self):
        with self._lock:
            self.relations = {}

    def _broadcast_relations(self):
        """Share the relations this unit observed (idempotent, re-sent
        periodically so late joiners converge on the same layout)."""
        with self._lock:
            mine = [{"obs": o, "fl": f, **r}
                    for (o, f), r in self.relations.items()
                    if o == self.node_id]
        if mine:
            self._send({"t": "rel", "rels": mine})

    def layout(self):
        """This unit's slice of the world.

        Returns (offset_px, world_w_px, n_units).  When dual
        calibration has measured relations, offsets come from real
        geometry; unmeasured units fall back to position-ordered
        butt-joints minus the manual overlap trim.
        """
        offsets, world_w, _, _ = self._solve_layout()
        return offsets[self.node_id], world_w, len(offsets)

    def blend_spans(self):
        """Measured overlap with neighbors: (left_px, right_px) of this
        unit's canvas that other units also cover — the strips to
        feather so doubled projection doesn't glow twice."""
        _, _, spans, _ = self._solve_layout()
        wx0, wx1 = spans[self.node_id]
        span_w = max(wx1 - wx0, 1e-6)
        px_per_world = self.width_px / span_w
        left = right = 0.0
        for nid, (ox0, ox1) in spans.items():
            if nid == self.node_id:
                continue
            if ox0 < wx0:
                left = max(left, ox1 - wx0)
            elif ox0 > wx0:
                right = max(right, wx1 - ox0)
        return (max(0.0, min(left * px_per_world, self.width_px)),
                max(0.0, min(right * px_per_world, self.width_px)))

    @staticmethod
    def _rel_affine(r):
        """Relation -> (A, t): observer-canvas point -> flasher-canvas."""
        s, th = r.get("scale", 1.0), r.get("rot", 0.0)
        c, sn = np.cos(th) * s, np.sin(th) * s
        A = np.array([[c, -sn], [sn, c]], np.float64)
        t = -A @ np.array([r["ox"], r["oy"]], np.float64)
        return A, t

    def _solve_layout(self):
        """World placement for every known unit, identical on all units.

        Each unit gets a full affine transform world→canvas.  Measured
        relations form a graph; BFS from the lowest-id measured unit
        (the anchor, whose canvas frame *is* the world frame) chains
        the affines, so rotated or scaled neighbors are placed with
        their true orientation, not just an offset.  Unmeasured units
        are appended by position order; world x is normalized so the
        leftmost canvas corner sits at 0.

        Returns (offsets, world_w, spans, transforms):
          offsets: unit -> leftmost world x of its canvas
          spans:   unit -> (wx0, wx1) world x range of its canvas
          transforms: unit -> (A, t) mapping world -> unit canvas
        """
        with self._lock:
            units = {self.node_id: (self.position, self.width_px)}
            for nid, p in self.peers.items():
                units[nid] = (p["position"], p["width"])
            rels = dict(self.relations)

        # Best-confidence edge per pair when both directions measured
        edges = {}
        for (obs, fl), r in rels.items():
            if obs not in units or fl not in units:
                continue
            key = tuple(sorted((obs, fl)))
            if key not in edges or r["conf"] > edges[key][0]:
                edges[key] = (r["conf"], obs, fl, r)

        adj = {}
        for _, obs, fl, r in edges.values():
            A, t = self._rel_affine(r)
            adj.setdefault(obs, []).append((fl, A, t, False))
            adj.setdefault(fl, []).append((obs, A, t, True))

        I = np.eye(2)
        transforms = {}
        if adj:
            anchor = min(nid for nid in adj if nid in units)
            transforms[anchor] = (I.copy(), np.zeros(2))
            queue = [anchor]
            while queue:
                cur = queue.pop(0)
                A_c, t_c = transforms[cur]
                for nxt, A_m, t_m, inverse in adj.get(cur, []):
                    if nxt in transforms:
                        continue
                    if inverse:     # cur = M(nxt) ⇒ T_nxt = M⁻¹ ∘ T_cur
                        A_inv = np.linalg.inv(A_m)
                        transforms[nxt] = (A_inv @ A_c, A_inv @ (t_c - t_m))
                    else:           # nxt = M(cur) ⇒ T_nxt = M ∘ T_cur
                        transforms[nxt] = (A_m @ A_c, A_m @ t_c + t_m)
                    queue.append(nxt)

        def world_span(nid):
            """World x range of a unit's canvas (corners through T⁻¹)."""
            A, t = transforms[nid]
            w = units[nid][1]
            h = w * 9.0 / 16.0      # aspect only affects tilt margins
            corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
            world = (np.linalg.inv(A) @ (corners - t).T).T
            return float(world[:, 0].min()), float(world[:, 0].max())

        # Unmeasured units: butt-joint after the measured extent
        extent = max((world_span(n)[1] for n in transforms), default=0.0)
        rest = sorted((units[n][0], n) for n in units if n not in transforms)
        for _, nid in rest:
            off = extent - self.overlap_px if extent > 0 else 0.0
            transforms[nid] = (I.copy(), np.array([-off, 0.0]))
            extent = off + units[nid][1]

        # Normalize world x so the leftmost corner sits at 0
        spans = {nid: world_span(nid) for nid in units}
        m = min(s[0] for s in spans.values())
        shift = np.array([m, 0.0])
        for nid in transforms:
            A, t = transforms[nid]
            transforms[nid] = (A, A @ shift + t)
        spans = {nid: (s[0] - m, s[1] - m) for nid, s in spans.items()}

        offsets = {nid: spans[nid][0] for nid in units}
        world_w = max(s[1] for s in spans.values())
        return offsets, max(world_w, self.width_px), spans, transforms

    def world_transform(self):
        """This unit's (A, t): world point -> own canvas point."""
        _, _, _, transforms = self._solve_layout()
        return transforms[self.node_id]

    def world(self):
        """A world adapter for performers (FishTank et al.)."""
        return _MeshWorld(self)

    # ── Wire ────────────────────────────────────────────────────────────

    def _send(self, msg):
        msg["m"] = MAGIC
        msg["id"] = self.node_id
        try:
            self._sock.sendto(json.dumps(msg).encode(),
                              (self.broadcast, self.port))
        except OSError:
            pass

    def _tx_loop(self):
        while self._running:
            self._send({"t": "hello", "pos": self.position,
                        "w": self.width_px})
            if self.is_leader:
                self._send({"t": "time", "clock": time.monotonic()})
                self._send({"t": "show", **self._show})
            self._rel_tick += 1
            if self._rel_tick % 5 == 0:
                self._broadcast_relations()
            self._prune()
            time.sleep(HELLO_INTERVAL)

    def _rx_loop(self):
        while self._running:
            try:
                data, _ = self._sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                msg = json.loads(data.decode())
            except ValueError:
                continue
            if msg.get("m") != MAGIC or msg.get("id") == self.node_id:
                continue
            self._handle(msg)

    def _handle(self, msg):
        kind = msg.get("t")
        nid = msg.get("id", "")
        if kind == "hello":
            with self._lock:
                self.peers[nid] = {
                    "position": int(msg.get("pos", 0)),
                    "width": float(msg.get("w", 640)),
                    "last_seen": time.monotonic(),
                }
        elif kind == "time" and nid < self.node_id:
            # Only trust clocks from ids that outrank ours (the leader)
            offset = float(msg.get("clock", 0.0)) - time.monotonic()
            # Smooth: first sample snaps, then ease to absorb jitter
            if self._clock_offset == 0.0:
                self._clock_offset = offset
            else:
                self._clock_offset += (offset - self._clock_offset) * 0.15
        elif kind == "show" and nid < self.node_id:
            for key in ("seed", "mood", "tempo", "video"):
                if msg.get(key) is not None:
                    self._show[key] = msg[key]
        elif kind == "dcal":
            with self._lock:
                self._dcal_in.append(msg)
        elif kind == "rel":
            with self._lock:
                for r in msg.get("rels", []):
                    try:
                        self.relations[(r["obs"], r["fl"])] = {
                            "ox": float(r["ox"]), "oy": float(r["oy"]),
                            "scale": float(r.get("scale", 1.0)),
                            "rot": float(r.get("rot", 0.0)),
                            "conf": float(r.get("conf", 0.0)),
                        }
                    except (KeyError, TypeError, ValueError):
                        continue

    def _prune(self):
        cutoff = time.monotonic() - PEER_TIMEOUT
        with self._lock:
            gone = [nid for nid, p in self.peers.items()
                    if p["last_seen"] < cutoff]
            for nid in gone:
                del self.peers[nid]


class _MeshWorld:
    """World adapter: live view of the mesh for performers.

    to_canvas() applies the full measured affine, so world content
    lands correctly even when this unit's canvas is rotated or scaled
    relative to its neighbors (non-coplanar fence panels, tilted
    mounts) — a fish crossing the seam bends with the surface.
    """

    def __init__(self, node):
        self._node = node
        self._refresh()

    def _refresh(self):
        self.offset_px, self.world_w, _ = self._node.layout()
        self.seed = self._node.show_state()["seed"]
        self._A, self._t = self._node.world_transform()

    def now(self):
        return self._node.now()

    def to_canvas(self, x, y):
        """World point -> this unit's canvas point."""
        return (self._A[0, 0] * x + self._A[0, 1] * y + self._t[0],
                self._A[1, 0] * x + self._A[1, 1] * y + self._t[1])
