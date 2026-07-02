"""Performers — autonomous show behaviors the director casts onto surfaces.

Each performer is a small living thing that paints on the canvas every
tick.  The scene director analyzes the physical scene, picks performers,
and assigns each to a surface.  Pure numpy/cv2, no Qt.

World-aware performers (the fish tank) take a `world` object providing
a shared coordinate strip across meshed units:

    world.offset_px   -> where this unit's canvas starts in world space
    world.world_w     -> total world width in px
    world.seed        -> shared seed, same on every unit
    world.now()       -> shared world time in seconds

Because trajectories are pure functions of (seed, world time), every
unit computes identical positions with no frame streaming — a fish
swims seamlessly from one projector to the next.
"""

import numpy as np
import cv2


class LocalWorld:
    """Standalone world: this unit is the whole tank."""

    def __init__(self, width_px, seed=1234):
        import time
        self.offset_px = 0.0
        self.world_w = float(width_px)
        self.seed = seed
        self._t0 = time.monotonic()
        self._time = __import__("time")

    def now(self):
        return self._time.monotonic() - self._t0


class Performer:
    """Base: step(canvas, dt, t) paints one tick of the behavior."""

    def step(self, canvas, dt, t):
        raise NotImplementedError


class AuroraDrift(Performer):
    """A wisp sweeping slowly along a surface's long axis."""

    def __init__(self, surface, w, h, tempo=0.5, seed=0):
        x0, y0, x1, y1 = surface["bbox"] if surface else (0, 0, w, h)
        self.cx, self.cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
        self.ax = max(30, (x1 - x0) * 0.42)
        self.ay = max(20, (y1 - y0) * 0.30)
        self.rate = 0.05 + 0.10 * tempo
        rng = np.random.default_rng(seed)
        self.p1, self.p2 = rng.uniform(0, 6.28, 2)
        self.hue_off = rng.uniform(0, 1)

    def step(self, canvas, dt, t):
        x = self.cx + self.ax * np.sin(self.rate * 2 * np.pi * t + self.p1)
        y = self.cy + self.ay * np.sin(self.rate * 1.37 * 2 * np.pi * t + self.p2)
        pulse = 0.55 + 0.45 * np.sin(0.6 * t + self.p1)
        canvas.dab(x, y, canvas.palette(self.hue_off), gain=0.32 * pulse, width=1.25)


class BreathingGlow(Performer):
    """A pool of light that breathes at a surface's heart."""

    def __init__(self, surface, w, h, tempo=0.5, seed=0):
        self.x, self.y = surface["centroid"] if surface else (w * 0.5, h * 0.5)
        self.rate = 0.35 + 0.5 * tempo      # breaths per ~10s
        rng = np.random.default_rng(seed)
        self.phase = rng.uniform(0, 6.28)
        self.hue_off = rng.uniform(0, 1)

    def step(self, canvas, dt, t):
        breath = 0.5 + 0.5 * np.sin(self.rate * t + self.phase)
        canvas.dab(self.x, self.y, canvas.palette(self.hue_off),
                   gain=0.10 + 0.28 * breath ** 2, width=1.55)


class Fireflies(Performer):
    """Motes rising now and then from within a surface."""

    def __init__(self, surface, w, h, tempo=0.5, seed=0):
        self.rng = np.random.default_rng(seed)
        self.interval = max(0.4, 2.2 - 1.6 * tempo)
        self._next = 0.0
        mask = surface.get("mask") if surface else None
        if mask is not None and mask.any():
            ys, xs = np.nonzero(mask)
            self.scale = surface.get("mask_scale", 4)
            self.points = np.stack([xs, ys], axis=1)
        else:
            self.points = None
            self.w, self.h = w, h

    def step(self, canvas, dt, t):
        if t < self._next:
            return
        self._next = t + self.interval * self.rng.uniform(0.5, 1.5)
        n = int(self.rng.integers(1, 4))
        if self.points is not None:
            pick = self.rng.integers(0, len(self.points), n)
            pos = self.points[pick].astype(np.float32) * self.scale
            pos += self.rng.normal(0, 2, (n, 2)).astype(np.float32)
        else:
            pos = self.rng.uniform([0, 0], [self.w, self.h], (n, 2)).astype(np.float32)
        vel = self.rng.normal(0, 9, (n, 2)).astype(np.float32)
        vel[:, 1] -= 7.0
        col = np.tile(canvas.palette(0.45) * 0.85, (n, 1))
        canvas.motes(pos, vel, col, life=(2.5, 6.0))


class ContourTrace(Performer):
    """A point of light traveling the outline of a physical object.

    This is the 'it can see my things' moment: the director found a
    shape in the room and now light traces its silhouette forever.
    """

    def __init__(self, surface, w, h, tempo=0.5, seed=0):
        self.path = None
        mask = surface.get("mask") if surface else None
        if mask is not None:
            scale = surface.get("mask_scale", 4)
            m = (mask.astype(np.uint8)) * 255
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
                c *= scale
                if len(c) >= 3:
                    seg = np.linalg.norm(np.diff(np.vstack([c, c[:1]]), axis=0), axis=1)
                    self.cum = np.concatenate([[0], np.cumsum(seg)])
                    self.total = float(self.cum[-1])
                    self.path = np.vstack([c, c[:1]])
        self.speed = 40 + 90 * tempo        # px/s along the outline
        rng = np.random.default_rng(seed)
        self.hue_off = rng.uniform(0, 1)
        self.s0 = rng.uniform(0, 1)

    def _point_at(self, s):
        s = s % self.total
        i = int(np.searchsorted(self.cum, s, side="right")) - 1
        i = min(i, len(self.path) - 2)
        seg_len = self.cum[i + 1] - self.cum[i]
        f = (s - self.cum[i]) / max(seg_len, 1e-6)
        return self.path[i] * (1 - f) + self.path[i + 1] * f

    def step(self, canvas, dt, t):
        if self.path is None or self.total < 40:
            return
        # Two tracers half a loop apart, meeting and parting forever
        for k, off in enumerate((0.0, 0.5)):
            s = self.s0 * self.total + self.speed * t * (1 if k == 0 else 0.93) \
                + off * self.total
            p = self._point_at(s)
            canvas.dab(p[0], p[1], canvas.palette(self.hue_off + k * 0.13),
                       gain=0.5, width=0.55)


class FishTank(Performer):
    """Fish swimming through a shared world — across every projector.

    Trajectories are closed-form functions of world time, so meshed
    units render the same fish in perfect sync without talking.
    """

    def __init__(self, surface, w, h, tempo=0.5, seed=0, world=None, n_fish=None):
        self.world = world if world is not None else LocalWorld(w, seed=seed)
        self.canvas_w, self.canvas_h = w, h
        if surface and surface.get("bbox"):
            x0, y0, x1, y1 = surface["bbox"]
            self.y_lo, self.y_hi = y0 + (y1 - y0) * 0.15, y1 - (y1 - y0) * 0.15
        else:
            self.y_lo, self.y_hi = h * 0.2, h * 0.85
        density = max(1, int(self.world.world_w / 320))
        self.n = n_fish if n_fish is not None else min(9, 2 + density)
        self._bubble_next = 0.0
        self._local_rng = np.random.default_rng(seed + 999)

        rng = np.random.default_rng(int(self.world.seed))
        span = self.y_hi - self.y_lo
        self.f_y0 = self.y_lo + rng.uniform(0.1, 0.9, self.n) * span
        self.f_speed = rng.uniform(26, 62, self.n) * (0.6 + 0.8 * tempo)
        self.f_dir = np.where(rng.random(self.n) < 0.5, -1.0, 1.0)
        self.f_x0 = rng.uniform(0, self.world.world_w, self.n)
        self.f_amp = rng.uniform(6, 22, self.n)
        self.f_freq = rng.uniform(0.5, 1.3, self.n)
        self.f_phase = rng.uniform(0, 6.28, self.n)
        self.f_size = rng.uniform(0.32, 0.62, self.n)
        self.f_hue = rng.uniform(0, 1, self.n)

    def step(self, canvas, dt, t):
        wt = self.world.now()
        margin = 80.0
        wrap = self.world.world_w + 2 * margin
        for i in range(self.n):
            head_wx = (self.f_x0[i] + self.f_dir[i] * self.f_speed[i] * wt) % wrap - margin
            lx = head_wx - self.world.offset_px
            if lx < -margin or lx > self.canvas_w + margin:
                continue
            color = canvas.palette(self.f_hue[i])
            # Spine: head + trailing segments with a travelling wiggle
            seg = 9.0 * self.f_size[i] * 2.2
            for k in range(6):
                sx = lx - self.f_dir[i] * k * seg
                ph = self.f_freq[i] * 2 * np.pi * wt + self.f_phase[i] - k * 0.75
                sy = self.f_y0[i] + self.f_amp[i] * np.sin(ph)
                fade = (1.0 - k / 6.0)
                canvas.dab(sx, sy, color, gain=0.55 * fade ** 1.5,
                           width=self.f_size[i] * (1.0 - 0.09 * k))
        # A bubble now and then, from wherever a fish just was
        if t >= self._bubble_next and self.n > 0:
            self._bubble_next = t + self._local_rng.uniform(1.2, 3.0)
            i = int(self._local_rng.integers(0, self.n))
            head_wx = (self.f_x0[i] + self.f_dir[i] * self.f_speed[i] * wt) % wrap - margin
            lx = head_wx - self.world.offset_px
            if 0 <= lx < self.canvas_w:
                sy = float(self.f_y0[i])
                pos = np.array([[lx, sy]], np.float32)
                vel = np.array([[0.0, -26.0]], np.float32)
                col = (canvas.palette(0.5) * 0.7)[None, :]
                canvas.motes(pos, vel, col, life=(1.5, 3.0))


# ── Casting ─────────────────────────────────────────────────────────────

REGISTRY = {
    "aurora_drift": AuroraDrift,
    "breathing_glow": BreathingGlow,
    "fireflies": Fireflies,
    "contour_trace": ContourTrace,
    "fish_tank": FishTank,
}

BEHAVIOR_NAMES = list(REGISTRY.keys())


def create(spec, surfaces, w, h, tempo=0.5, world=None, seed=0):
    """Build one performer from a director's behavior spec.

    spec: {"type": name, "region": surface id or None}
    Returns None for unknown types (a director hallucination, ignored).
    """
    cls = REGISTRY.get(spec.get("type"))
    if cls is None:
        return None
    surface = None
    rid = spec.get("region")
    if rid is not None:
        surface = next((s for s in surfaces if s["id"] == rid), None)
    if surface is None and surfaces:
        surface = surfaces[0]
    kwargs = {"tempo": tempo, "seed": seed}
    if cls is FishTank:
        kwargs["world"] = world
    return cls(surface, w, h, **kwargs)
