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

    def to_canvas(self, x, y):
        return x, y


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


class PerspectiveBox(Performer):
    """A wireframe box breathing in and out of the surface — fake 3D.

    The front face sits on the surface; the back face is shifted toward
    a vanishing direction and shrunk, and the depth pulses so the box
    seems to push out of and sink back into the wall.

    `perspective` sets which way the depth recedes, so the illusion
    matches the angle the surface is seen from:
      "left"/"right"/"up"/"down" — the back of the box leans that way
      "center" — recedes straight back (one-point, head-on)
      "auto"   — recedes toward the canvas center (correct one-point
                 perspective for a viewer facing the middle)
    """

    DIRECTIONS = {
        "left": (-1.0, -0.22),
        "right": (1.0, -0.22),
        "up": (0.0, -1.0),
        "down": (0.0, 1.0),
        "center": (0.0, 0.0),
    }

    def __init__(self, surface, w, h, tempo=0.5, seed=0, perspective="auto"):
        if surface and surface.get("bbox"):
            x0, y0, x1, y1 = surface["bbox"]
            self.cx, self.cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
            self.hw = float(np.clip((x1 - x0) * 0.26, 24, w * 0.17))
            self.hh = float(np.clip((y1 - y0) * 0.26, 20, h * 0.17))
        else:
            self.cx, self.cy = w * 0.5, h * 0.5
            self.hw, self.hh = w * 0.13, h * 0.16

        if perspective in self.DIRECTIONS:
            dx, dy = self.DIRECTIONS[perspective]
        else:                       # auto: recede toward canvas center
            dx, dy = w * 0.5 - self.cx, h * 0.5 - self.cy
            if np.hypot(dx, dy) < 40:       # box at center — go head-on
                dx, dy = 0.0, 0.0
        n = np.hypot(dx, dy)
        self.dir = (dx / n, dy / n) if n > 1e-6 else (0.0, 0.0)

        self.max_shift = 0.85 * min(self.hw, self.hh)
        self.rate = (0.25 + 0.45 * tempo) * 2 * np.pi / 6.0  # ~one breath / 6-13s
        rng = np.random.default_rng(seed)
        self.phase = rng.uniform(0, 6.28)
        self.wobble_phase = rng.uniform(0, 6.28)
        self.hue_off = rng.uniform(0, 1)
        self._glint_next = 0.0
        self._glint_rng = rng

    def _corners(self, cx, cy, hw, hh):
        return [(cx - hw, cy - hh), (cx + hw, cy - hh),
                (cx + hw, cy + hh), (cx - hw, cy + hh)]

    def step(self, canvas, dt, t):
        # Depth breathes: never flat, never fully detached
        d = 0.2 + 0.8 * (0.5 + 0.5 * np.sin(self.rate * t + self.phase))
        # A slow, tiny sway of the vanishing direction keeps it alive
        wob = 0.10 * np.sin(0.31 * t + self.wobble_phase)
        c, s = np.cos(wob), np.sin(wob)
        dx = self.dir[0] * c - self.dir[1] * s
        dy = self.dir[0] * s + self.dir[1] * c

        shift = self.max_shift * d
        shrink = 1.0 - 0.38 * d
        bx, by = self.cx + dx * shift, self.cy + dy * shift
        front = self._corners(self.cx, self.cy, self.hw, self.hh)
        back = self._corners(bx, by, self.hw * shrink, self.hh * shrink)

        col_f = canvas.palette(self.hue_off)
        col_b = canvas.palette(self.hue_off + 0.12)
        # Back face dim (atmospheric depth), connectors fading toward
        # the back, front face brightest — depth cues in light
        for i in range(4):
            canvas.glow_line(back[i], back[(i + 1) % 4], col_b, gain=0.16)
        for i in range(4):
            mid = ((front[i][0] + back[i][0]) * 0.5,
                   (front[i][1] + back[i][1]) * 0.5)
            canvas.glow_line(front[i], mid, col_b, gain=0.30)
            canvas.glow_line(mid, back[i], col_b, gain=0.18)
        for i in range(4):
            canvas.glow_line(front[i], front[(i + 1) % 4], col_f, gain=0.38)
        # Vertex glints
        for p in front:
            canvas.dab(p[0], p[1], col_f, gain=0.22, width=0.2)
        # Now and then a corner sheds a spark
        if t >= self._glint_next:
            self._glint_next = t + self._glint_rng.uniform(2.0, 5.0)
            p = front[int(self._glint_rng.integers(0, 4))]
            pos = np.array([p], np.float32)
            vel = self._glint_rng.normal(0, 14, (1, 2)).astype(np.float32)
            canvas.motes(pos, vel, (col_f * 1.1)[None, :], life=(1.0, 2.2))


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
            # Quick cull in world x before doing per-segment transforms
            lx = head_wx - self.world.offset_px
            if lx < -margin or lx > self.canvas_w + margin:
                continue
            color = canvas.palette(self.f_hue[i])
            # Spine: head + trailing segments with a travelling wiggle,
            # each point placed through the measured world→canvas
            # transform so fish bend correctly across tilted units
            seg = 9.0 * self.f_size[i] * 2.2
            for k in range(6):
                swx = head_wx - self.f_dir[i] * k * seg
                ph = self.f_freq[i] * 2 * np.pi * wt + self.f_phase[i] - k * 0.75
                swy = self.f_y0[i] + self.f_amp[i] * np.sin(ph)
                cx, cy = self.world.to_canvas(swx, swy)
                fade = (1.0 - k / 6.0)
                canvas.dab(cx, cy, color, gain=0.55 * fade ** 1.5,
                           width=self.f_size[i] * (1.0 - 0.09 * k))
        # A bubble now and then, from wherever a fish just was
        if t >= self._bubble_next and self.n > 0:
            self._bubble_next = t + self._local_rng.uniform(1.2, 3.0)
            i = int(self._local_rng.integers(0, self.n))
            head_wx = (self.f_x0[i] + self.f_dir[i] * self.f_speed[i] * wt) % wrap - margin
            cx, cy = self.world.to_canvas(head_wx, float(self.f_y0[i]))
            if 0 <= cx < self.canvas_w:
                pos = np.array([[cx, cy]], np.float32)
                vel = np.array([[0.0, -26.0]], np.float32)
                col = (canvas.palette(0.5) * 0.7)[None, :]
                canvas.motes(pos, vel, col, life=(1.5, 3.0))


class Constellation(Performer):
    """Drifting stars that form and break living constellations.

    Points wander slowly inside the surface; glow lines connect the
    ones that drift near each other — a star map that keeps redrawing
    itself."""

    def __init__(self, surface, w, h, tempo=0.5, seed=0):
        self.rng = np.random.default_rng(seed)
        if surface and surface.get("bbox"):
            x0, y0, x1, y1 = surface["bbox"]
        else:
            x0, y0, x1, y1 = w * 0.1, h * 0.1, w * 0.9, h * 0.9
        self.bounds = (x0, y0, x1, y1)
        n = 11
        self.pos = self.rng.uniform([x0, y0], [x1, y1], (n, 2)).astype(np.float32)
        self.vel = self.rng.normal(0, 4.5, (n, 2)).astype(np.float32)
        self.tw_phase = self.rng.uniform(0, 6.28, n).astype(np.float32)
        self.link_r = max(70.0, 0.35 * min(x1 - x0, y1 - y0))
        self.hue_off = self.rng.uniform(0, 1)
        self.speed = 0.6 + 0.9 * tempo

    def step(self, canvas, dt, t):
        x0, y0, x1, y1 = self.bounds
        self.vel += self.rng.normal(0, 1.6, self.vel.shape).astype(np.float32) * dt * 10
        self.vel *= 0.995
        self.pos += self.vel * dt * self.speed
        # Soft bounce at the surface's edges
        for axis, lo, hi in ((0, x0, x1), (1, y0, y1)):
            low = self.pos[:, axis] < lo
            high = self.pos[:, axis] > hi
            self.vel[low | high, axis] *= -1
            self.pos[:, axis] = np.clip(self.pos[:, axis], lo, hi)

        col_star = canvas.palette(self.hue_off)
        col_line = canvas.palette(self.hue_off + 0.08)
        tw = 0.5 + 0.5 * np.sin(t * 2.1 + self.tw_phase)
        for i, p in enumerate(self.pos):
            # Crisp twinkling star on the fast field; a whisper of it
            # lingers on the slow field as it drifts
            canvas.glow_line(p, p, col_star, gain=0.45 + 0.4 * tw[i])
            canvas.dab(p[0], p[1], col_star, gain=0.035, width=0.2)
        # Lines between close pairs, brightening as they near
        d = np.linalg.norm(self.pos[:, None] - self.pos[None, :], axis=2)
        ii, jj = np.nonzero((d < self.link_r) & (d > 1.0))
        for i, j in zip(ii, jj):
            if i < j:
                g = 0.3 * (1.0 - d[i, j] / self.link_r)
                canvas.glow_line(self.pos[i], self.pos[j], col_line, gain=g)


class RainOfLight(Performer):
    """Streaks of light falling through a surface, splashing at its
    lower edge into motes."""

    def __init__(self, surface, w, h, tempo=0.5, seed=0):
        self.rng = np.random.default_rng(seed)
        if surface and surface.get("bbox"):
            x0, y0, x1, y1 = surface["bbox"]
        else:
            x0, y0, x1, y1 = 0, 0, w, h
        self.bounds = (x0, y0, x1, y1)
        self.max_drops = 6 + int(8 * tempo)
        self.drops = []                 # [x, y, vy, hue]
        self.spawn_acc = 0.0
        self.rate = 2.0 + 6.0 * tempo   # drops per second
        self.hue_off = self.rng.uniform(0, 1)

    def step(self, canvas, dt, t):
        x0, y0, x1, y1 = self.bounds
        self.spawn_acc += dt * self.rate
        while self.spawn_acc >= 1.0 and len(self.drops) < self.max_drops:
            self.spawn_acc -= 1.0
            self.drops.append([self.rng.uniform(x0, x1), y0,
                               self.rng.uniform(150, 260),
                               self.rng.uniform(-0.06, 0.06)])
        splashed = []
        for drop in self.drops:
            drop[1] += drop[2] * dt
            color = canvas.palette(self.hue_off + drop[3])
            tail = drop[2] * 0.075
            canvas.glow_line((drop[0], drop[1] - tail), (drop[0], drop[1]),
                             color, gain=0.34)
            if drop[1] >= y1:
                splashed.append(drop)
        for drop in splashed:
            self.drops.remove(drop)
            k = int(self.rng.integers(2, 5))
            pos = np.tile([drop[0], y1], (k, 1)).astype(np.float32)
            vel = self.rng.normal(0, 30, (k, 2)).astype(np.float32)
            vel[:, 1] = -np.abs(vel[:, 1]) * 0.8
            col = np.tile(canvas.palette(self.hue_off + drop[3]) * 0.8, (k, 1))
            canvas.motes(pos, vel, col, life=(0.6, 1.6))


class Orbits(Performer):
    """A tiny solar system: glowing bodies circling a breathing sun,
    inner ones faster, leaving comet trails on the slow field."""

    def __init__(self, surface, w, h, tempo=0.5, seed=0):
        self.cx, self.cy = surface["centroid"] if surface else (w * 0.5, h * 0.5)
        rng = np.random.default_rng(seed)
        if surface and surface.get("bbox"):
            x0, y0, x1, y1 = surface["bbox"]
            r_max = 0.42 * min(x1 - x0, y1 - y0)
        else:
            r_max = 0.3 * min(w, h)
        r_max = max(r_max, 40.0)
        self.n = 3
        self.radius = np.linspace(0.35, 1.0, self.n) * r_max
        # Kepler-flavored: inner orbits faster
        self.omega = (0.5 + tempo) * 1.6 / np.sqrt(self.radius / self.radius[0])
        self.phase = rng.uniform(0, 6.28, self.n)
        self.tilt = rng.uniform(0.5, 0.8, self.n)    # ellipse squash
        self.hue = rng.uniform(0, 1, self.n)
        self.sun_phase = rng.uniform(0, 6.28)
        ang = np.linspace(0, 2 * np.pi, 25)
        self._pcos, self._psin = np.cos(ang), np.sin(ang)

    def step(self, canvas, dt, t):
        breath = 0.5 + 0.5 * np.sin(0.5 * t + self.sun_phase)
        canvas.dab(self.cx, self.cy, canvas.palette(0.0),
                   gain=0.04 + 0.08 * breath, width=0.7)
        canvas.glow_line((self.cx, self.cy), (self.cx, self.cy),
                         canvas.palette(0.0), gain=0.5 + 0.3 * breath)
        for i in range(self.n):
            color = canvas.palette(self.hue[i])
            # Faint orbit path so the celestial mechanics read
            xs = self.cx + self.radius[i] * self._pcos
            ys = self.cy + self.radius[i] * self.tilt[i] * self._psin
            for k in range(24):
                canvas.glow_line((xs[k], ys[k]), (xs[k + 1], ys[k + 1]),
                                 color, gain=0.045)
            a = self.omega[i] * t + self.phase[i]
            x = self.cx + self.radius[i] * np.cos(a)
            y = self.cy + self.radius[i] * self.tilt[i] * np.sin(a)
            # Crisp body + a soft comet trail left on the slow field
            canvas.glow_line((x, y), (x, y), color, gain=0.8)
            canvas.dab(x, y, color, gain=0.12, width=0.28)


class PulseRings(Performer):
    """Rings of light rippling out from a point, like rain on water."""

    SEGMENTS = 26

    def __init__(self, surface, w, h, tempo=0.5, seed=0):
        self.cx, self.cy = surface["centroid"] if surface else (w * 0.5, h * 0.5)
        if surface and surface.get("bbox"):
            x0, y0, x1, y1 = surface["bbox"]
            self.r_max = 0.55 * max(x1 - x0, y1 - y0)
        else:
            self.r_max = 0.4 * max(w, h)
        self.speed = 40 + 60 * tempo            # px/s ring growth
        self.rng = np.random.default_rng(seed)
        self.interval = max(1.2, 4.5 - 3.0 * tempo)
        self._next = 0.5
        self.rings = []                         # birth times
        self.hue_off = self.rng.uniform(0, 1)
        ang = np.linspace(0, 2 * np.pi, self.SEGMENTS + 1)
        self._cos, self._sin = np.cos(ang), np.sin(ang)

    def step(self, canvas, dt, t):
        if t >= self._next and len(self.rings) < 3:
            self._next = t + self.interval * self.rng.uniform(0.7, 1.3)
            self.rings.append(t)
        alive = []
        for t0 in self.rings:
            r = (t - t0) * self.speed
            if r >= self.r_max:
                continue
            alive.append(t0)
            fade = (1.0 - r / self.r_max) ** 1.1
            color = canvas.palette(self.hue_off + 0.1 * (t - t0))
            xs = self.cx + r * self._cos
            ys = self.cy + r * self._sin * 0.92      # slight squash: on a wall
            for k in range(self.SEGMENTS):
                canvas.glow_line((xs[k], ys[k]), (xs[k + 1], ys[k + 1]),
                                 color, gain=0.5 * fade)
        self.rings = alive


class Tendrils(Performer):
    """Vines of light growing across the surface, forking, dissolving,
    and regrowing — painted onto the slow field, so the canvas itself
    remembers and melts them like ivy in a dream."""

    MAX_TIPS = 5

    def __init__(self, surface, w, h, tempo=0.5, seed=0):
        self.rng = np.random.default_rng(seed)
        if surface and surface.get("bbox"):
            self.bounds = surface["bbox"]
        else:
            self.bounds = (w * 0.05, h * 0.05, w * 0.95, h * 0.95)
        self.speed = 26 + 40 * tempo            # growth px/s
        self.hue_off = self.rng.uniform(0, 1)
        self.tips = []
        self._sprout(t=0.0)

    def _sprout(self, t):
        x0, y0, x1, y1 = self.bounds
        # New vines climb from a surface edge
        edge = int(self.rng.integers(0, 4))
        if edge == 0:
            pos = [self.rng.uniform(x0, x1), y1]; heading = -np.pi / 2
        elif edge == 1:
            pos = [self.rng.uniform(x0, x1), y0]; heading = np.pi / 2
        elif edge == 2:
            pos = [x0, self.rng.uniform(y0, y1)]; heading = 0.0
        else:
            pos = [x1, self.rng.uniform(y0, y1)]; heading = np.pi
        self.tips.append({"pos": np.array(pos, np.float32),
                          "heading": heading, "born": t,
                          "life": self.rng.uniform(6.0, 14.0),
                          "hue": self.hue_off + self.rng.uniform(-0.08, 0.08)})

    def step(self, canvas, dt, t):
        x0, y0, x1, y1 = self.bounds
        survivors = []
        for tip in self.tips:
            age = t - tip["born"]
            if age > tip["life"]:
                continue
            tip["heading"] += self.rng.normal(0, 1.4) * dt * 3.0
            step_len = self.speed * dt
            tip["pos"][0] += step_len * np.cos(tip["heading"])
            tip["pos"][1] += step_len * np.sin(tip["heading"])
            px, py = tip["pos"]
            if not (x0 - 20 <= px <= x1 + 20 and y0 - 20 <= py <= y1 + 20):
                continue
            # Paint onto the slow field: the canvas remembers the vine —
            # with a bright growing tip on the crisp field
            color = canvas.palette(tip["hue"])
            canvas.dab(px, py, color, gain=0.4, width=0.3)
            canvas.glow_line((px, py), (px, py), color, gain=0.55)
            # Occasional fork; occasional leaf-mote
            if len(self.tips) < self.MAX_TIPS and self.rng.random() < 0.4 * dt:
                self.tips.append({"pos": tip["pos"].copy(),
                                  "heading": tip["heading"]
                                  + self.rng.choice([-1, 1]) * 0.8,
                                  "born": t,
                                  "life": self.rng.uniform(3.0, 8.0),
                                  "hue": tip["hue"] + 0.05})
            if self.rng.random() < 0.25 * dt:
                canvas.motes(np.array([[px, py]], np.float32),
                             self.rng.normal(0, 8, (1, 2)).astype(np.float32),
                             (color * 0.8)[None, :], life=(2.0, 4.0))
            survivors.append(tip)
        self.tips = survivors
        if not self.tips:
            self._sprout(t)


# ── Casting ─────────────────────────────────────────────────────────────

REGISTRY = {
    "aurora_drift": AuroraDrift,
    "breathing_glow": BreathingGlow,
    "fireflies": Fireflies,
    "contour_trace": ContourTrace,
    "fish_tank": FishTank,
    "perspective_box": PerspectiveBox,
    "constellation": Constellation,
    "rain": RainOfLight,
    "orbits": Orbits,
    "pulse_rings": PulseRings,
    "tendrils": Tendrils,
}

BEHAVIOR_NAMES = list(REGISTRY.keys())

PERSPECTIVES = ["auto", "left", "right", "up", "down", "center"]


def create(spec, surfaces, w, h, tempo=0.5, world=None, seed=0):
    """Build one performer from a director's behavior spec.

    spec: {"type": name, "region": surface id or None,
           "perspective": optional, for perspective_box}
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
    if cls is PerspectiveBox:
        persp = spec.get("perspective", "auto")
        kwargs["perspective"] = persp if persp in PERSPECTIVES else "auto"
    return cls(surface, w, h, **kwargs)
