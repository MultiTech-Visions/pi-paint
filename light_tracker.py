"""Light and motion tracking — the eyes of the light painting system.

Pure numpy/cv2, no Qt.  Watches camera frames for two kinds of presence:

* Bright, compact light sources (a phone flashlight, a candle, a
  glowing toy) — tracked across frames as continuous strokes.
* Body motion — frame differencing distilled into a soft "mist"
  energy image, so waving a hand stirs the wall even without a light.

Both are translated into projector coordinates through the structured
light calibration, which is what makes the projected response land
exactly where the light or hand touched the physical surface.

Self-suppression: the projector's own output would otherwise be seen
by the camera and painted back — an infinite feedback loop.  We know
what we're projecting and we know the camera↔projector mapping, so we
predict the projector's contribution to the camera image and subtract
it before detecting anything.  This is what keeps the illusion
airtight: the system reacts to *your* light, never its own.
"""

import numpy as np
import cv2


class _Track:
    """One light source followed across frames."""

    _next_id = 0

    def __init__(self, cam_x, cam_y, intensity):
        self.id = _Track._next_id
        _Track._next_id += 1
        self.cam_x = cam_x
        self.cam_y = cam_y
        self.intensity = intensity
        self.proj_prev = None       # last emitted projector point
        self.missed = 0
        self.seed = (self.id * 0.61803398875) % 1.0


class LightTracker:
    def __init__(self, cam_w, cam_h, proj_w, proj_h,
                 blob_threshold=205, min_area=4, max_area=4000):
        self.cam_w = cam_w
        self.cam_h = cam_h
        self.proj_w = proj_w
        self.proj_h = proj_h

        self.blob_threshold = blob_threshold
        self.min_area = min_area
        self.max_area = max_area

        self.tracks = []
        self._prev_gray = None
        self._prev_predicted = None

        # Calibration state
        self._cal_proj_x = None     # camera-res float32 maps -> projector coords
        self._cal_proj_y = None
        self._inv_x = None          # projector-res float32 maps -> camera coords
        self._inv_y = None
        self._inv_x_h = None        # half-scale variants for mist warping
        self._inv_y_h = None

    # ── Calibration ─────────────────────────────────────────────────────

    @property
    def is_calibrated(self):
        return self._cal_proj_x is not None

    def set_calibration(self, proj_x, proj_y, inv_x=None, inv_y=None):
        """Install camera→projector maps (and optional prebuilt inverse)."""
        self._cal_proj_x = np.where(proj_x < 0, -1, proj_x).astype(np.float32)
        self._cal_proj_y = np.where(proj_y < 0, -1, proj_y).astype(np.float32)
        self._inv_x = inv_x
        self._inv_y = inv_y
        # Half-scale inverse maps: mist is computed at half camera
        # resolution (it's inherently soft), then warped straight to
        # projector space in one remap.
        if inv_x is not None:
            self._inv_x_h = inv_x * 0.5
            self._inv_y_h = inv_y * 0.5
        else:
            self._inv_x_h = self._inv_y_h = None

    def clear_calibration(self):
        self._cal_proj_x = self._cal_proj_y = None
        self._inv_x = self._inv_y = None
        self._inv_x_h = self._inv_y_h = None

    # ── Main entry ──────────────────────────────────────────────────────

    def process(self, frame_bgr, projected_rgb=None, sensitivity=1.0,
                want_mist=True):
        """Analyze one camera frame.

        frame_bgr: camera frame (BGR uint8, as from cv2.VideoCapture).
        projected_rgb: the frame currently on the projector (RGB uint8),
            used for self-suppression.  None disables suppression.
        sensitivity: 0.3..2.0 — scales how easily lights are detected.

        Returns dict with:
            lights: list of {id, x, y, prev, intensity, speed, seed}
                    in projector coordinates
            mist:   float32 (proj_h, proj_w) motion energy, or None
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        predicted = self._predict_projection(projected_rgb)
        if predicted is not None:
            detect = cv2.addWeighted(gray, 1.0, predicted, -0.9, 0)
        else:
            detect = gray

        lights = self._detect_and_track(detect, gray, sensitivity)

        mist = None
        if want_mist:
            mist = self._motion_mist(gray, predicted)

        self._prev_gray = gray
        self._prev_predicted = predicted
        return {"lights": lights, "mist": mist}

    # ── Projection prediction (feedback suppression) ────────────────────

    def _predict_projection(self, projected_rgb):
        """Estimate the projector's contribution to the camera image."""
        if projected_rgb is None or not self.is_calibrated:
            return None
        proj_gray = cv2.cvtColor(projected_rgb, cv2.COLOR_RGB2GRAY)
        # For each camera pixel, sample the projector pixel it sees.
        # Unmapped pixels (-1) fall outside and read as 0 — correct:
        # the projector doesn't reach them.
        pred = cv2.remap(proj_gray, self._cal_proj_x, self._cal_proj_y,
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        # Soften: projection is diffuse by the time the camera sees it
        return cv2.GaussianBlur(pred, (0, 0), 3.0)

    # ── Bright light detection + tracking ───────────────────────────────

    def _detect_and_track(self, detect_img, gray, sensitivity):
        thr = float(np.clip(self.blob_threshold / max(sensitivity, 0.05),
                            140, 250))
        _, mask = cv2.threshold(detect_img, thr, 1, cv2.THRESH_BINARY)

        n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        detections = []
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.min_area or area > self.max_area:
                continue
            cx, cy = centroids[i]
            # Compactness: a hand-held light is a blob, not a streak of
            # ambient wash.  Reject long thin regions.
            w_, h_ = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            if max(w_, h_) > 8 * max(1, min(w_, h_)):
                continue
            inten = float(gray[int(cy), int(cx)]) / 255.0
            detections.append((cx, cy, inten))

        # Greedy nearest-neighbor matching to existing tracks
        unmatched = list(range(len(detections)))
        for tr in self.tracks:
            best, best_d = None, 90.0
            for j in unmatched:
                d = np.hypot(detections[j][0] - tr.cam_x,
                             detections[j][1] - tr.cam_y)
                if d < best_d:
                    best, best_d = j, d
            if best is not None:
                cx, cy, inten = detections[best]
                # EMA smoothing keeps strokes silky instead of jittery
                tr.cam_x += (cx - tr.cam_x) * 0.55
                tr.cam_y += (cy - tr.cam_y) * 0.55
                tr.intensity = inten
                tr.missed = 0
                unmatched.remove(best)
            else:
                tr.missed += 1

        for j in unmatched:
            cx, cy, inten = detections[j]
            self.tracks.append(_Track(cx, cy, inten))

        # A few missed frames of grace so strokes bridge flicker
        self.tracks = [t for t in self.tracks if t.missed <= 6]

        lights = []
        for tr in self.tracks:
            if tr.missed > 0:
                continue
            pt = self._to_projector(tr.cam_x, tr.cam_y)
            if pt is None:
                continue
            px, py = pt
            speed = 0.0
            prev = tr.proj_prev
            if prev is not None:
                speed = float(np.hypot(px - prev[0], py - prev[1]))
                if speed > 220:     # calibration seam jump — break the ribbon
                    prev = None
            lights.append({
                "id": tr.id, "x": px, "y": py, "prev": prev,
                "intensity": tr.intensity, "speed": speed, "seed": tr.seed,
            })
            tr.proj_prev = (px, py)
        return lights

    def _to_projector(self, cam_x, cam_y):
        """Map a camera point into projector space."""
        if not self.is_calibrated:
            # Uncalibrated fallback: proportional mapping
            return (cam_x / self.cam_w * self.proj_w,
                    cam_y / self.cam_h * self.proj_h)
        xi = int(np.clip(cam_x, 0, self.cam_w - 1))
        yi = int(np.clip(cam_y, 0, self.cam_h - 1))
        px = self._cal_proj_x[yi, xi]
        py = self._cal_proj_y[yi, xi]
        if px >= 0 and py >= 0:
            return float(px), float(py)
        # The exact pixel wasn't decoded (light saturated it during the
        # scan, or it sits in a mapping hole) — average valid neighbors.
        r = 9
        y0, y1 = max(0, yi - r), min(self.cam_h, yi + r + 1)
        x0, x1 = max(0, xi - r), min(self.cam_w, xi + r + 1)
        wx = self._cal_proj_x[y0:y1, x0:x1]
        wy = self._cal_proj_y[y0:y1, x0:x1]
        ok = (wx >= 0) & (wy >= 0)
        if not np.any(ok):
            return None
        return float(wx[ok].mean()), float(wy[ok].mean())

    # ── Motion mist ─────────────────────────────────────────────────────

    def _motion_mist(self, gray, predicted):
        if self._prev_gray is None:
            return None
        diff = cv2.absdiff(gray, self._prev_gray)
        # The projector's own animation moves every frame; subtract its
        # predicted change so only real-world motion remains.
        if predicted is not None and self._prev_predicted is not None:
            pchange = cv2.absdiff(predicted, self._prev_predicted)
            diff = cv2.subtract(diff, cv2.convertScaleAbs(pchange, alpha=1.4))
        diff = cv2.subtract(diff, 10)   # noise floor (saturating)
        # Mist is inherently soft — process at half resolution
        half = cv2.resize(diff, (self.cam_w // 2, self.cam_h // 2),
                          interpolation=cv2.INTER_AREA)
        half = cv2.GaussianBlur(half, (0, 0), 2.0)
        if half.max() <= 0:
            return None
        mist_cam = np.minimum(half.astype(np.float32) / 90.0, 1.0)

        if self._inv_x_h is not None:
            # Warp into projector space so mist lands where the body is
            mist = cv2.remap(mist_cam, self._inv_x_h, self._inv_y_h,
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        else:
            mist = cv2.resize(mist_cam, (self.proj_w, self.proj_h),
                              interpolation=cv2.INTER_LINEAR)
        return mist
