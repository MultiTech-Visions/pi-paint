"""Qt glue for the light painting experience.

Runs the tick loop: pull the latest camera frame, let the tracker find
lights and motion, feed them to the living canvas, and push the result
to the projector.  All the actual magic lives in paint_canvas.py and
light_tracker.py (both Qt-free); this file just keeps the heartbeat.
"""

import numpy as np
import cv2
from PyQt5.QtCore import QObject, QTimer, pyqtSignal

from paint_canvas import PaintCanvas
from light_tracker import LightTracker


class PaintEngine(QObject):
    preview_ready = pyqtSignal(object)   # uint8 RGB frame (projector res)
    status = pyqtSignal(str)

    def __init__(self, config, latest_frame_fn, projector, calibration):
        """
        latest_frame_fn : callable -> BGR frame or None
            Returns the most recent camera frame without touching the
            device (the control panel's preview loop owns the camera).
        """
        super().__init__()
        self.config = config
        self.latest_frame = latest_frame_fn
        self.projector = projector
        self.calibration = calibration

        pcfg = config.get("painting", {})
        self.fps = pcfg.get("fps", 30)
        proj_w = config["projector"]["width"]
        proj_h = config["projector"]["height"]
        cam_w = config["camera"]["width"]
        cam_h = config["camera"]["height"]

        # Simulate below projector resolution and let the projector
        # window upscale — big speedup on the Pi, and the softening
        # only adds to the dreaminess.
        self.render_scale = float(pcfg.get("render_scale", 0.75))
        self.canvas_w = max(160, int(proj_w * self.render_scale))
        self.canvas_h = max(90, int(proj_h * self.render_scale))

        self.canvas = PaintCanvas(self.canvas_w, self.canvas_h, fps=self.fps)
        self.canvas.set_mood(pcfg.get("mood", "aurora"))
        self.canvas.set_memory(pcfg.get("memory_seconds", 12))
        self.canvas.dreams_enabled = pcfg.get("dreams", True)

        self.tracker = LightTracker(cam_w, cam_h, self.canvas_w, self.canvas_h)

        self.sensitivity = pcfg.get("sensitivity", 1.0)
        self.motion_mist = pcfg.get("motion_mist", True)

        self._last_output = None
        self._running = False
        self._frame_count = 0
        self.performers = []        # autonomous show behaviors (autopilot)
        self._blend_mask = None     # edge feathering for mesh overlaps
        self._blend_key = (0, 0)

        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)

    @property
    def is_running(self):
        return self._running

    def refresh_calibration(self):
        """Pull the current calibration into the tracker (call on start
        and whenever a new scan completes)."""
        if self.calibration.has_calibration:
            rs = self.render_scale
            # Calibration maps carry projector coords; rescale their
            # values (and the inverse maps' shape) to canvas resolution.
            px = np.where(self.calibration.proj_x < 0, -1,
                          self.calibration.proj_x * rs)
            py = np.where(self.calibration.proj_y < 0, -1,
                          self.calibration.proj_y * rs)
            inv_x, inv_y = self.calibration.build_inverse_maps()
            if inv_x is not None and rs != 1.0:
                inv_x = cv2.resize(inv_x, (self.canvas_w, self.canvas_h),
                                   interpolation=cv2.INTER_NEAREST)
                inv_y = cv2.resize(inv_y, (self.canvas_w, self.canvas_h),
                                   interpolation=cv2.INTER_NEAREST)
            self.tracker.set_calibration(px, py, inv_x, inv_y)
            self.status.emit("Painting with calibrated mapping")
        else:
            self.tracker.clear_calibration()
            self.status.emit("Painting uncalibrated (proportional mapping) — "
                             "run a calibration scan for surface-accurate magic")

    def start(self):
        if self._running:
            return
        self.refresh_calibration()
        self._running = True
        self._timer.start(int(1000 / self.fps))

    def stop(self):
        self._running = False
        self._timer.stop()

    def release(self):
        """Dissolve the current painting into fireflies."""
        self.canvas.release()

    def set_performers(self, performers):
        """Install the autonomous show cast (empty list clears it).

        Performers paint alongside human light — people can still walk
        up and paint over the show, which is the point.
        """
        self.performers = [p for p in performers if p is not None]

    def set_edge_blend(self, left_px, right_px):
        """Feather output brightness over measured mesh overlaps so
        strips covered by two projectors don't glow twice."""
        key = (int(round(left_px)), int(round(right_px)))
        if key == self._blend_key:
            return
        self._blend_key = key
        left, right = key
        if left <= 0 and right <= 0:
            self._blend_mask = None
            return
        ramp = np.ones(self.canvas_w, dtype=np.float32)
        if left > 0:
            n = min(left, self.canvas_w)
            t = np.linspace(0.0, 1.0, n, dtype=np.float32)
            ramp[:n] = t * t * (3.0 - 2.0 * t)          # smoothstep in
        if right > 0:
            n = min(right, self.canvas_w)
            t = np.linspace(1.0, 0.0, n, dtype=np.float32)
            ramp[-n:] = np.minimum(ramp[-n:], t * t * (3.0 - 2.0 * t))
        mask_row = (ramp * 255.0).astype(np.uint8)
        self._blend_mask = np.ascontiguousarray(
            np.broadcast_to(mask_row[None, :, None],
                            (self.canvas_h, self.canvas_w, 3)))

    def _tick(self):
        frame = self.latest_frame()
        if frame is not None:
            # Mist on alternate frames (double gain) — it's soft anyway,
            # and this halves its cost
            mist_frame = self.motion_mist and self._frame_count % 2 == 0
            result = self.tracker.process(
                frame,
                projected_rgb=self._last_output,
                sensitivity=self.sensitivity,
                want_mist=mist_frame,
            )
            for light in result["lights"]:
                self.canvas.stroke(
                    light["x"], light["y"],
                    prev=light["prev"],
                    intensity=light["intensity"],
                    speed=light["speed"],
                    seed=light["seed"],
                )
            if result["mist"] is not None:
                self.canvas.add_mist(result["mist"], gain=2.0)

        t_show = self._frame_count / self.fps
        for performer in self.performers:
            performer.step(self.canvas, 1.0 / self.fps, t_show)

        out = self.canvas.step()
        if self._blend_mask is not None:
            out = cv2.multiply(out, self._blend_mask, scale=1.0 / 255.0)
        self._last_output = out
        self.projector.show_image(out)

        self._frame_count += 1
        if self._frame_count % 3 == 0:      # preview at ~10 fps, projector at 30
            self.preview_ready.emit(out)
