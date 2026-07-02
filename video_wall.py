"""Video wall — one video source spread across many projectors.

Every unit holds a local copy of the same file (drop it in videos/ on
each box).  Nothing is streamed: the mesh world clock chooses the
frame — position = world_time mod duration — so all units decode the
same frame within a few milliseconds of each other, joining or
rebooting mid-show lands in the right place automatically, and twenty
units cost the same bandwidth as one.

Each unit warps the video through its measured world→canvas affine
(from dual calibration), so the video is stretched across the whole
mesh strip and bends correctly over tilted or scaled units.  Stand-
alone (no mesh) the video simply fills this unit's canvas.

The layer composites as a crisp base under the living paint: light
painting, fish, and every performer glow over the movie, and mesh
edge blending applies to the whole output.

Pure numpy/cv2, no Qt — testable headless.
"""

import numpy as np
import cv2


class VideoLayer:
    def __init__(self, path, canvas_w, canvas_h, brightness=1.0):
        self.path = path
        self.canvas_w = int(canvas_w)
        self.canvas_h = int(canvas_h)
        self.brightness = float(brightness)

        self.cap = cv2.VideoCapture(path)
        self.ok = self.cap.isOpened()
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not self.fps or self.fps <= 1:
            self.fps = 30.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.vid_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vid_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.n_frames <= 0 or self.vid_w <= 0:
            self.ok = False

        self._cur = -1
        self._raw = None            # last decoded frame (RGB)
        self._warped = None         # cached canvas-space frame
        self._M = None
        # Default geometry: the video fills this unit's canvas
        self.set_geometry(np.eye(2), np.zeros(2),
                          self.canvas_w, self.canvas_h)

    @property
    def duration(self):
        return self.n_frames / self.fps if self.ok else 0.0

    def set_geometry(self, A, t, world_w, world_h):
        """Install the world→canvas affine and the world strip size.

        The video is stretched to fill the whole world strip; this
        unit renders the slice its transform says it owns.
        """
        if not self.ok:
            return
        # video px -> world px (stretch fill), then world -> canvas
        S = np.array([[world_w / self.vid_w, 0.0],
                      [0.0, world_h / self.vid_h]], np.float64)
        M_A = np.asarray(A, np.float64) @ S
        M = np.hstack([M_A, np.asarray(t, np.float64).reshape(2, 1)])
        M = M.astype(np.float32)
        if self._M is None or not np.allclose(M, self._M, atol=1e-4):
            self._M = M
            self._warped = None

    def set_brightness(self, brightness):
        self.brightness = float(brightness)
        self._warped = None

    def frame_at(self, t_world):
        """The canvas-space video frame for a moment of world time.

        Sequential reads while playback is smooth; a real seek only on
        drift or loop wrap, so decode stays cheap.
        """
        if not self.ok:
            return None
        target = int(t_world * self.fps) % self.n_frames
        if target != self._cur:
            delta = (target - self._cur) % self.n_frames
            ret, frame = False, None
            if 1 <= delta <= 5 and self._cur >= 0:
                for _ in range(delta):
                    ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                ret, frame = self.cap.read()
            if ret:
                self._raw = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self._cur = target
                self._warped = None
        if self._warped is None and self._raw is not None:
            warped = cv2.warpAffine(
                self._raw, self._M, (self.canvas_w, self.canvas_h),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            if abs(self.brightness - 1.0) > 0.02:
                warped = cv2.convertScaleAbs(warped, alpha=self.brightness)
            self._warped = warped
        return self._warped

    def release(self):
        if self.cap is not None:
            self.cap.release()
        self.ok = False
