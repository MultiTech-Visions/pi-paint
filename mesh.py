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
        self._show = {"seed": 1234, "mood": None, "tempo": None}

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

    def set_show(self, seed=None, mood=None, tempo=None):
        """Leader: publish show parameters to the mesh."""
        if seed is not None:
            self._show["seed"] = int(seed)
        if mood is not None:
            self._show["mood"] = mood
        if tempo is not None:
            self._show["tempo"] = tempo

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
        offsets, world_w, _ = self._solve_layout()
        return offsets[self.node_id], world_w, len(offsets)

    def blend_spans(self):
        """Measured overlap with neighbors: (left_px, right_px) of this
        unit's canvas that other units also cover — the strips to
        feather so doubled projection doesn't glow twice."""
        offsets, _, widths = self._solve_layout()
        my_off = offsets[self.node_id]
        my_w = widths[self.node_id]
        left = right = 0.0
        for nid, off in offsets.items():
            if nid == self.node_id:
                continue
            w = widths[nid]
            if off < my_off:
                left = max(left, (off + w) - my_off)
            elif off > my_off:
                right = max(right, (my_off + my_w) - off)
        return (max(0.0, min(float(left), my_w)),
                max(0.0, min(float(right), my_w)))

    def _solve_layout(self):
        """offsets/widths for every known unit, identical on all units.

        Measured relations form a graph (edge: flasher_offset =
        observer_offset + ox); BFS from the lowest-id measured unit
        places the connected component; anything unmeasured is appended
        after the current extent in (position, id) order.  Offsets are
        normalized so the leftmost unit sits at 0.
        """
        with self._lock:
            units = {self.node_id: (self.position, self.width_px)}
            for nid, p in self.peers.items():
                units[nid] = (p["position"], p["width"])
            rels = dict(self.relations)

        # Undirected offset edges between known units, best-confidence
        # relation wins when both directions were measured
        edges = {}
        for (obs, fl), r in rels.items():
            if obs not in units or fl not in units:
                continue
            key = tuple(sorted((obs, fl)))
            cand = (r["conf"], obs, fl, r["ox"])
            if key not in edges or cand[0] > edges[key][0]:
                edges[key] = cand

        adj = {}
        for _, obs, fl, ox in edges.values():
            adj.setdefault(obs, []).append((fl, ox))
            adj.setdefault(fl, []).append((obs, -ox))

        offsets = {}
        if adj:
            anchor = min(nid for nid in adj if nid in units)
            offsets[anchor] = 0.0
            queue = [anchor]
            while queue:
                cur = queue.pop(0)
                for nxt, dx in adj.get(cur, []):
                    if nxt not in offsets:
                        offsets[nxt] = offsets[cur] + dx
                        queue.append(nxt)

        # Unmeasured units: butt-joint after the current extent
        extent = max((offsets[n] + units[n][1] for n in offsets), default=0.0)
        rest = sorted((units[n][0], n) for n in units if n not in offsets)
        for _, nid in rest:
            off = extent - self.overlap_px if extent > 0 else 0.0
            offsets[nid] = off
            extent = off + units[nid][1]

        # Normalize: leftmost unit at 0
        m = min(offsets.values())
        offsets = {nid: off - m for nid, off in offsets.items()}
        widths = {nid: units[nid][1] for nid in units}
        world_w = max(offsets[n] + widths[n] for n in units)
        return offsets, max(world_w, self.width_px), widths

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
            for key in ("seed", "mood", "tempo"):
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
    """World adapter: live view of the mesh for performers."""

    def __init__(self, node):
        self._node = node
        self._refresh()

    def _refresh(self):
        self.offset_px, self.world_w, _ = self._node.layout()
        self.seed = self._node.show_state()["seed"]

    def now(self):
        return self._node.now()
