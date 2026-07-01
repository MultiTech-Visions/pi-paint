"""Calibration engine for camera↔projector coordinate mapping.

Runs a structured light scan using Gray code patterns, captures
each pattern with the camera, decodes the correspondence, and
builds a mapping that translates camera pixels → projector pixels.
"""

import json
import os
import numpy as np
import cv2
from PyQt5.QtCore import QObject, QTimer, pyqtSignal
from gray_code import generate_gray_patterns, decode_gray_captures


class CalibrationEngine(QObject):
    """Manages the full structured light calibration scan."""

    # Signals
    progress = pyqtSignal(int, int, str)   # (current_step, total_steps, description)
    scan_finished = pyqtSignal(bool, str)  # (success, message)
    preview_ready = pyqtSignal(object)     # numpy RGB image for preview

    SAVE_PATH = "calibration_data.npz"

    def __init__(self, config, camera_fn, projector):
        """
        Parameters
        ----------
        config : dict
            Application config (needs calibration.gray_code_wait_ms, etc.)
        camera_fn : callable
            Function that returns (success, grayscale_frame) when called.
        projector : ProjectorWindow
            Reference to projector for displaying patterns.
        """
        super().__init__()
        self.config = config
        self.grab_frame = camera_fn
        self.projector = projector

        self.proj_w = config["projector"]["width"]
        self.proj_h = config["projector"]["height"]

        self.wait_ms = config["calibration"]["gray_code_wait_ms"]
        self.delay_ms = config["calibration"]["capture_delay_ms"]

        # Scan state
        self._patterns = []
        self._captures = []
        self._scan_idx = 0
        self._phase = None  # "positive" or "negative"
        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._scan_step)
        self._scanning = False

        # Results
        self.proj_x = None
        self.proj_y = None
        self.confidence = None
        self._has_calibration = False

    @property
    def is_scanning(self):
        return self._scanning

    @property
    def has_calibration(self):
        return self._has_calibration

    def start_scan(self):
        """Begin the structured light calibration scan."""
        if self._scanning:
            return

        self._patterns = generate_gray_patterns(self.proj_w, self.proj_h)
        self._captures = []
        self._scan_idx = 0
        self._phase = "positive"
        self._scanning = True

        total = len(self._patterns) * 2  # pos + neg for each
        self.progress.emit(0, total, "Starting scan...")

        # Show first pattern
        self._show_current_pattern()
        self._timer.start(self.wait_ms)

    def cancel_scan(self):
        """Cancel an in-progress scan."""
        self._timer.stop()
        self._scanning = False
        self.projector.show_solid(__import__('PyQt5').QtGui.QColor(0, 0, 0))
        self.scan_finished.emit(False, "Scan cancelled")

    def _show_current_pattern(self):
        """Display the current pattern on the projector."""
        pat, inv = self._patterns[self._scan_idx]
        if self._phase == "positive":
            self.projector.show_pattern(pat)
        else:
            self.projector.show_pattern(inv)

    def _scan_step(self):
        """Capture the current pattern and advance to the next."""
        if not self._scanning:
            return

        # Capture frame
        ok, frame = self.grab_frame()
        if not ok:
            self.cancel_scan()
            self.scan_finished.emit(False, "Failed to capture camera frame")
            return

        # Convert to grayscale if needed
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Store capture
        if self._phase == "positive":
            self._current_pos = gray.copy()
            self._phase = "negative"
        else:
            self._captures.append((self._current_pos, gray.copy()))
            self._phase = "positive"
            self._scan_idx += 1

        total = len(self._patterns) * 2
        current = self._scan_idx * 2 + (0 if self._phase == "positive" else 1)

        if self._scan_idx >= len(self._patterns):
            # Scan complete — decode
            self._scanning = False
            self.projector.show_solid(__import__('PyQt5').QtGui.QColor(0, 0, 0))
            self.progress.emit(total, total, "Decoding...")
            self._decode_and_finish()
            return

        # Show next pattern and schedule capture
        desc = f"Pattern {self._scan_idx + 1}/{len(self._patterns)} ({self._phase})"
        self.progress.emit(current, total, desc)
        self._show_current_pattern()
        self._timer.start(self.wait_ms)

    def _decode_and_finish(self):
        """Decode captures into the camera→projector map."""
        self.proj_x, self.proj_y, self.confidence = decode_gray_captures(
            self._captures, self.proj_w, self.proj_h
        )
        self._has_calibration = True

        # Generate preview visualization
        self._emit_preview()

        # Save to disk
        self.save_calibration()

        valid_pixels = np.sum(self.confidence >= 0.1)
        total_pixels = self.confidence.size
        pct = 100.0 * valid_pixels / total_pixels
        self.scan_finished.emit(True, f"Calibration complete — {pct:.1f}% coverage ({valid_pixels:,} pixels)")

    def _emit_preview(self):
        """Generate and emit a color-coded visualization of the mapping."""
        preview = self.build_mapping_preview()
        if preview is not None:
            self.preview_ready.emit(preview)

    def build_mapping_preview(self):
        """Build a color-coded image showing the camera→projector mapping.

        Returns an RGB numpy array where hue encodes projector X and
        saturation encodes projector Y, with brightness from confidence.
        """
        if not self._has_calibration:
            return None

        cam_h, cam_w = self.proj_x.shape
        hsv = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)

        valid = self.confidence >= 0.1

        # Hue from projector X (0-179 for OpenCV HSV)
        hsv[valid, 0] = (self.proj_x[valid] / self.proj_w * 179).astype(np.uint8)
        # Saturation from projector Y
        hsv[valid, 1] = (self.proj_y[valid] / self.proj_h * 255).astype(np.uint8)
        # Value from confidence
        hsv[valid, 2] = (self.confidence[valid] * 255).astype(np.uint8)

        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return rgb

    def build_confidence_preview(self):
        """Build a grayscale image showing per-pixel confidence."""
        if not self._has_calibration:
            return None
        gray = (self.confidence * 255).astype(np.uint8)
        return np.stack([gray] * 3, axis=-1)

    def build_coverage_overlay(self):
        """Build an overlay showing which projector pixels are covered.

        Returns an RGB image at projector resolution where green pixels
        are mapped by at least one camera pixel.
        """
        if not self._has_calibration:
            return None

        overlay = np.zeros((self.proj_h, self.proj_w, 3), dtype=np.uint8)
        valid = (self.proj_x >= 0) & (self.proj_y >= 0)
        px = self.proj_x[valid].astype(np.int32)
        py = self.proj_y[valid].astype(np.int32)

        # Clamp
        px = np.clip(px, 0, self.proj_w - 1)
        py = np.clip(py, 0, self.proj_h - 1)

        overlay[py, px] = [0, 255, 0]
        return overlay

    def build_inverse_maps(self):
        """Build projector→camera lookup maps from the forward calibration.

        Returns (inv_x, inv_y): float32 arrays at projector resolution
        giving, for each projector pixel, the camera pixel that sees it.
        Holes (projector pixels no camera pixel decoded to) are filled by
        normalized-convolution interpolation from valid neighbors, so the
        maps are dense enough to warp full camera images (e.g. motion
        mist) into projector space with a single cv2.remap.

        Returns (None, None) if there is no calibration.
        """
        if not self._has_calibration:
            return None, None

        valid = (self.proj_x >= 0) & (self.proj_y >= 0) & (self.confidence >= 0.1)
        cam_ys, cam_xs = np.nonzero(valid)
        if len(cam_ys) == 0:
            return None, None

        px = np.clip(self.proj_x[valid].astype(np.int32), 0, self.proj_w - 1)
        py = np.clip(self.proj_y[valid].astype(np.int32), 0, self.proj_h - 1)

        inv_x = np.zeros((self.proj_h, self.proj_w), dtype=np.float32)
        inv_y = np.zeros((self.proj_h, self.proj_w), dtype=np.float32)
        known = np.zeros((self.proj_h, self.proj_w), dtype=np.float32)
        inv_x[py, px] = cam_xs.astype(np.float32)
        inv_y[py, px] = cam_ys.astype(np.float32)
        known[py, px] = 1.0

        # Fill holes: repeatedly average valid neighbors into unknown
        # pixels, growing the kernel so isolated gaps close quickly.
        for it in range(8):
            if known.min() > 0:
                break
            k = 5 + 4 * it
            sum_x = cv2.blur(inv_x * known, (k, k))
            sum_y = cv2.blur(inv_y * known, (k, k))
            cnt = cv2.blur(known, (k, k))
            fill = (known == 0) & (cnt > 1e-6)
            inv_x[fill] = sum_x[fill] / cnt[fill]
            inv_y[fill] = sum_y[fill] / cnt[fill]
            known[fill] = 1.0

        # Anything still unknown maps out of range (remap reads border 0)
        inv_x[known == 0] = -1
        inv_y[known == 0] = -1
        return inv_x, inv_y

    def camera_to_projector(self, cam_x, cam_y):
        """Look up the projector coordinate for a given camera pixel.

        Returns (proj_x, proj_y) or None if the pixel has no mapping.
        """
        if not self._has_calibration:
            return None
        cam_h, cam_w = self.proj_x.shape
        if not (0 <= cam_y < cam_h and 0 <= cam_x < cam_w):
            return None
        px = self.proj_x[int(cam_y), int(cam_x)]
        py = self.proj_y[int(cam_y), int(cam_x)]
        if px < 0 or py < 0:
            return None
        return (float(px), float(py))

    def save_calibration(self, path=None):
        """Save calibration data to disk."""
        path = path or self.SAVE_PATH
        if not self._has_calibration:
            return
        np.savez_compressed(
            path,
            proj_x=self.proj_x,
            proj_y=self.proj_y,
            confidence=self.confidence,
            proj_w=self.proj_w,
            proj_h=self.proj_h,
        )

    def load_calibration(self, path=None):
        """Load calibration data from disk if available."""
        path = path or self.SAVE_PATH
        if not os.path.exists(path):
            return False
        data = np.load(path)
        self.proj_x = data["proj_x"]
        self.proj_y = data["proj_y"]
        self.confidence = data["confidence"]
        self._has_calibration = True
        self._emit_preview()
        return True
