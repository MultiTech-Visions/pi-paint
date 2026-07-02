"""Dual calibration — meshed units learn where they overlap.

Each unit already knows camera→own-projector from its structured light
scan.  Dual calibration adds the missing piece: units take turns
flashing gray code patterns while every other unit's camera watches.
An observer that can see a neighbor's light decodes camera→neighbor-
projector, and combining that with its own calibration yields dense
point correspondences own-canvas ↔ neighbor-canvas.  A robust affine
fit turns those into a *relation*: where the neighbor's canvas origin
sits in my canvas coordinates.

Relations are shared over the mesh; the world layout is then solved
from measured geometry instead of manual butt-joints, and each unit
edge-blends across the measured overlaps so doubled light doesn't
glow twice.  Units that can't see each other simply measure nothing —
they stay placed by their position number.

Pure math up top (headless-testable); the DualCalibrator QObject below
orchestrates the flash/capture protocol over the mesh.
"""

import threading

import numpy as np
import cv2
from PyQt5.QtCore import QObject, QTimer, pyqtSignal
from PyQt5.QtGui import QColor

from gray_code import generate_gray_patterns, decode_gray_captures


# ── Pure math ───────────────────────────────────────────────────────────

def estimate_neighbor_transform(nb_x, nb_y, nb_conf, own_x, own_y, own_conf,
                                min_points=400, min_inlier_ratio=0.5):
    """Fit the transform own-canvas → neighbor-canvas from paired maps.

    All maps are camera-resolution: (nb_x, nb_y) give neighbor canvas
    coords per camera pixel (from decoding the neighbor's flash),
    (own_x, own_y) give own canvas coords (from own calibration,
    already scaled to canvas resolution).  -1 marks invalid.

    Returns {"ox","oy","scale","conf","n"} — the neighbor's canvas
    origin in own canvas coords, its pixel scale relative to ours, fit
    confidence, and correspondence count — or None if the neighbor
    simply isn't visible enough.
    """
    valid = ((nb_x >= 0) & (own_x >= 0) &
             (nb_conf >= 0.15) & (own_conf >= 0.15))
    ys, xs = np.nonzero(valid)
    if len(ys) < min_points:
        return None
    # Deterministic subsample, cap at 4000 pairs
    if len(ys) > 4000:
        sel = np.linspace(0, len(ys) - 1, 4000).astype(np.int64)
        ys, xs = ys[sel], xs[sel]
    src = np.stack([own_x[ys, xs], own_y[ys, xs]], axis=1).astype(np.float32)
    dst = np.stack([nb_x[ys, xs], nb_y[ys, xs]], axis=1).astype(np.float32)

    M, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    if M is None or inliers is None:
        return None
    ratio = float(inliers.mean())
    if ratio < min_inlier_ratio:
        return None

    A, t = M[:, :2], M[:, 2]
    det = float(np.linalg.det(A))
    if abs(det) < 1e-9:
        return None
    origin = np.linalg.solve(A, -t)     # where neighbor's (0,0) lands in ours
    scale = float(np.sqrt(abs(det)))    # neighbor px per own px
    rot = float(np.arctan2(A[1, 0], A[0, 0]))   # neighbor's tilt vs ours
    return {
        "ox": float(origin[0]),
        "oy": float(origin[1]),
        "scale": scale,
        "rot": rot,
        "conf": ratio,
        "n": int(len(ys)),
    }


def scale_own_maps(proj_x, proj_y, cal_w, canvas_w, canvas_h, cal_h):
    """Rescale calibration maps (projector-res coords) to canvas coords."""
    sx = canvas_w / float(cal_w)
    sy = canvas_h / float(cal_h)
    own_x = np.where(proj_x < 0, -1, proj_x * sx).astype(np.float32)
    own_y = np.where(proj_y < 0, -1, proj_y * sy).astype(np.float32)
    return own_x, own_y


# ── Protocol orchestration ──────────────────────────────────────────────

class DualCalibrator(QObject):
    """Runs the take-turns flash/observe protocol over the mesh.

    Any unit can lead (press the button on one box); the leader walks
    the mesh through the sequence with dcal messages.  Every unit runs
    this same object and reacts to incoming messages, so followers
    participate automatically.
    """

    status = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, config, get_mesh, projector, latest_frame_fn,
                 get_calibration, get_paint_engine, prepare_fn):
        """
        prepare_fn : callable -> bool.  Called when a dual calibration
            begins (locally or led by a peer): stop shows, go dark,
            ensure the camera — returns whether this unit can observe.
        """
        super().__init__()
        self.config = config
        self.get_mesh = get_mesh
        self.projector = projector
        self.latest_frame = latest_frame_fn
        self.get_calibration = get_calibration
        self.get_engine = get_paint_engine
        self.prepare = prepare_fn

        dcfg = config.get("dual_calibration", {})
        # settle: pattern-shown -> capture broadcast (camera exposure
        # catches up); dwell: capture broadcast -> next pattern (the
        # window observers have to actually grab a frame)
        self.settle_ms = int(dcfg.get("settle_ms", 300))
        self.dwell_ms = int(dcfg.get("dwell_ms", 200))

        self.active = False
        self._coordinator = False
        self._can_observe = False
        self._order = []
        self._idx = -1
        self._flash_token = 0
        self._captures = {}         # flasher_id -> {k: gray frame}
        self._totals = {}           # flasher_id -> expected step count

        # Fast drain: capture messages must be acted on well inside the
        # dwell window, before the flasher advances to the next pattern
        self._poll = QTimer()
        self._poll.timeout.connect(self._drain)
        self._poll.start(15)

        self._watchdog = QTimer()
        self._watchdog.setSingleShot(True)
        self._watchdog.timeout.connect(self._on_watchdog)

        self._relation_sig = _RelaySignal()
        self._relation_sig.done.connect(self._on_relation_done)

    # ── Leading ─────────────────────────────────────────────────────────

    def lead(self):
        """This unit coordinates a full mesh dual calibration."""
        mesh = self.get_mesh()
        if mesh is None:
            self.finished.emit(False, "mesh is not running")
            return
        ids = sorted([mesh.node_id] + list(mesh.peers.keys()))
        if len(ids) < 2:
            self.finished.emit(False, "no peers on the mesh yet")
            return
        self._coordinator = True
        self._order = ids
        self._idx = -1
        self._send_and_apply({"cmd": "begin"})
        QTimer.singleShot(1200, self._advance)

    def _advance(self):
        if not self.active or not self._coordinator:
            return
        self._watchdog.stop()
        self._idx += 1
        if self._idx >= len(self._order):
            self._send_and_apply({"cmd": "end"})
            return
        unit = self._order[self._idx]
        self.status.emit(f"unit {unit} flashing "
                         f"({self._idx + 1}/{len(self._order)})")
        self._send_and_apply({"cmd": "flash", "unit": unit})
        # Watchdog: a unit that never reports done doesn't stall the mesh
        eng = self.get_engine()
        patterns = 2 * (len(bin(eng.canvas_w)) + len(bin(eng.canvas_h)))
        est_ms = patterns * (self.settle_ms + self.dwell_ms)
        self._watchdog.start(int(est_ms * 1.5) + 5000)

    def _on_watchdog(self):
        if self.active and self._coordinator:
            self.status.emit("a unit went quiet — moving on")
            self._advance()

    # ── Message handling (all units) ────────────────────────────────────

    def _send_and_apply(self, payload):
        mesh = self.get_mesh()
        if mesh is not None:
            mesh.send_dcal(payload)
            payload = dict(payload, id=mesh.node_id)
            self._handle(payload)

    def _drain(self):
        mesh = self.get_mesh()
        if mesh is None:
            return
        for msg in mesh.drain_dcal():
            self._handle(msg)

    def _handle(self, msg):
        cmd = msg.get("cmd")
        mesh = self.get_mesh()
        me = mesh.node_id if mesh else None

        if cmd == "begin":
            self.active = True
            self._captures = {}
            self._totals = {}
            self._can_observe = bool(self.prepare())
            self.projector.show_solid(QColor(0, 0, 0))
            self.status.emit("dual calibration begins — going dark to watch")

        elif cmd == "flash" and self.active:
            if msg.get("unit") == me:
                self._start_flash()
            else:
                self.projector.show_solid(QColor(0, 0, 0))

        elif cmd == "capture" and self.active:
            unit = msg.get("unit")
            if unit != me and self._can_observe:
                frame = self.latest_frame()
                if frame is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    self._captures.setdefault(unit, {})[int(msg["k"])] = gray
                    self._totals[unit] = int(msg["total"])

        elif cmd == "flash_done" and self.active:
            unit = msg.get("unit")
            if unit != me:
                self._decode_async(unit)
            if self._coordinator:
                self._watchdog.stop()
                QTimer.singleShot(1500, self._advance)

        elif cmd == "end":
            self.active = False
            self._coordinator = False
            self.projector.show_solid(QColor(0, 0, 0))
            mesh_ = self.get_mesh()
            n_rel = len(mesh_.relations) if mesh_ else 0
            self.status.emit(f"dual calibration complete — "
                             f"{n_rel} measured relation(s)")
            self.finished.emit(True, f"{n_rel} relation(s) measured")

    # ── Flashing ────────────────────────────────────────────────────────

    def _start_flash(self):
        eng = self.get_engine()
        self._patterns = generate_gray_patterns(eng.canvas_w, eng.canvas_h)
        self._steps = len(self._patterns) * 2
        self._k = 0
        self._flash_token += 1
        self._flash_step(self._flash_token)

    def _flash_step(self, token):
        if token != self._flash_token or not self.active:
            return
        mesh = self.get_mesh()
        if self._k >= self._steps:
            self.projector.show_solid(QColor(0, 0, 0))
            self._send_and_apply({"cmd": "flash_done", "unit": mesh.node_id})
            return
        pat, inv = self._patterns[self._k // 2]
        self.projector.show_pattern(pat if self._k % 2 == 0 else inv)
        QTimer.singleShot(self.settle_ms, lambda: self._flash_capture(token))

    def _flash_capture(self, token):
        if token != self._flash_token or not self.active:
            return
        mesh = self.get_mesh()
        mesh.send_dcal({"cmd": "capture", "unit": mesh.node_id,
                        "k": self._k, "total": self._steps})
        self._k += 1
        QTimer.singleShot(self.dwell_ms, lambda: self._flash_step(token))

    # ── Observing / decoding ────────────────────────────────────────────

    def _decode_async(self, flasher_id):
        store = self._captures.pop(flasher_id, None)
        total = self._totals.get(flasher_id, 0)
        if not self._can_observe or store is None or total == 0:
            return
        if len(store) < total:
            self.status.emit(f"missed {total - len(store)} captures from "
                             f"{flasher_id} — skipping that pair")
            return
        cal = self.get_calibration()
        if not cal.has_calibration:
            return
        eng = self.get_engine()
        cw, ch = eng.canvas_w, eng.canvas_h
        own_x, own_y = scale_own_maps(cal.proj_x, cal.proj_y,
                                      cal.proj_w, cw, ch, cal.proj_h)
        own_conf = cal.confidence
        mesh = self.get_mesh()
        me = mesh.node_id

        def work():
            try:
                caps = [(store[2 * p], store[2 * p + 1])
                        for p in range(total // 2)]
                nb_x, nb_y, nb_conf = decode_gray_captures(caps, cw, ch)
                rel = estimate_neighbor_transform(nb_x, nb_y, nb_conf,
                                                  own_x, own_y, own_conf)
            except Exception as e:  # noqa: BLE001 — a failed pair is fine
                rel, e_ = None, e
            self._relation_sig.done.emit(me, flasher_id, rel)

        threading.Thread(target=work, daemon=True).start()

    def _on_relation_done(self, obs_id, flasher_id, rel):
        mesh = self.get_mesh()
        if rel is None:
            self.status.emit(f"can't see {flasher_id} from here "
                             f"(no shared surface)")
            return
        if mesh is not None:
            mesh.add_relation(obs_id, flasher_id, rel)
        self.status.emit(
            f"measured {flasher_id}: origin at ({rel['ox']:.0f}, "
            f"{rel['oy']:.0f}) in my canvas, scale {rel['scale']:.2f}, "
            f"tilt {np.degrees(rel['rot']):.1f}°, "
            f"confidence {rel['conf']:.2f}")


class _RelaySignal(QObject):
    """Cross-thread relay: decode worker -> GUI thread."""
    done = pyqtSignal(str, str, object)
