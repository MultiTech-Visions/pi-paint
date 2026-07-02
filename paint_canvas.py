"""The living canvas — the heart of the light painting experience.

A pure numpy/cv2 simulation (no Qt) of a luminous field that people
paint into with light.  Design goal: mesmerization.  Nothing here is a
plain "draw a line" — every mark becomes part of a slowly breathing,
drifting, dissolving organism:

* Strokes land as glowing ribbons that pool when you linger and
  tighten when you sweep fast.
* The whole field is advected by a slow, evolving wind so trails
  drift like smoke and never sit still.
* Old light softens into aurora washes, sheds fireflies, and
  sparkles as it fades.
* Colors are never chosen — each mood is a drifting palette, so the
  same gesture is never the same color twice.
* Left alone, the canvas dreams: a faint wisp draws to itself,
  inviting people to step in.

All coordinates are in projector space.  Feed it strokes and mist,
call step() at ~30 fps, project the returned frame.
"""

import colorsys
import numpy as np
import cv2


# ── Moods ───────────────────────────────────────────────────────────────
# Each mood is a loop of hue anchors (degrees) plus saturation/feel.
# The palette phase drifts continuously, so color is a river, not a picker.

MOODS = {
    "aurora":     {"hues": [140, 165, 200, 265, 290, 170], "sat": 0.85, "mist_sat": 0.7},
    "embers":     {"hues": [4, 18, 36, 52, 20, 8],         "sat": 0.95, "mist_sat": 0.8},
    "moonlight":  {"hues": [205, 220, 240, 210, 195, 225], "sat": 0.45, "mist_sat": 0.35},
    "prism":      {"hues": [0, 60, 120, 180, 240, 300],    "sat": 0.9,  "mist_sat": 0.75},
    "biolume":    {"hues": [165, 185, 210, 300, 320, 175], "sat": 1.0,  "mist_sat": 0.85},
}

MOOD_NAMES = list(MOODS.keys())

GOLDEN = 0.61803398875


def _palette_color(mood, phase):
    """Sample a mood's hue loop at a continuous phase in [0, 1)."""
    m = MOODS[mood]
    hues = m["hues"]
    n = len(hues)
    f = (phase % 1.0) * n
    i = int(f) % n
    j = (i + 1) % n
    t = f - int(f)
    # Interpolate hues the short way around the wheel
    h0, h1 = hues[i], hues[j]
    dh = ((h1 - h0 + 180) % 360) - 180
    hue = (h0 + dh * t) % 360
    r, g, b = colorsys.hsv_to_rgb(hue / 360.0, m["sat"], 1.0)
    return np.array([r, g, b], dtype=np.float32)


class _Particles:
    """Fireflies / embers.  Fixed-capacity vectorized particle pool."""

    def __init__(self, capacity, rng):
        self.cap = capacity
        self.rng = rng
        self.pos = np.zeros((capacity, 2), dtype=np.float32)
        self.vel = np.zeros((capacity, 2), dtype=np.float32)
        self.col = np.zeros((capacity, 3), dtype=np.float32)
        self.age = np.zeros(capacity, dtype=np.float32)
        self.life = np.zeros(capacity, dtype=np.float32)   # 0 => dead
        self.phase = np.zeros(capacity, dtype=np.float32)

    def spawn(self, positions, velocities, colors, life_range=(1.5, 4.0)):
        """Spawn len(positions) particles into free slots (drops overflow)."""
        n = len(positions)
        if n == 0:
            return
        free = np.nonzero(self.life <= self.age)[0]
        if len(free) == 0:
            return
        k = min(n, len(free))
        idx = free[:k]
        self.pos[idx] = positions[:k]
        self.vel[idx] = velocities[:k]
        self.col[idx] = colors[:k]
        self.age[idx] = 0.0
        self.life[idx] = self.rng.uniform(*life_range, size=k).astype(np.float32)
        self.phase[idx] = self.rng.uniform(0, 6.28, size=k).astype(np.float32)

    def step(self, dt, w, h):
        alive = self.life > self.age
        if not np.any(alive):
            return
        n = int(np.count_nonzero(alive))
        # Wander: gentle brownian jitter, a whisper of lift, and drag —
        # tuned so motes meander like fireflies rather than fall like rain.
        jitter = self.rng.normal(0, 14.0, size=(n, 2)).astype(np.float32)
        self.vel[alive] += jitter * dt
        self.vel[alive, 1] -= 5.5 * dt          # faint updraft
        self.vel[alive] *= (1.0 - 0.6 * dt)     # drag
        self.pos[alive] += self.vel[alive] * dt
        self.age[alive] += dt
        # Kill particles that drift off canvas
        p = self.pos[alive]
        off = (p[:, 0] < -4) | (p[:, 0] >= w + 4) | (p[:, 1] < -4) | (p[:, 1] >= h + 4)
        dead_idx = np.nonzero(alive)[0][off]
        self.life[dead_idx] = 0.0

    def render_into(self, E, t):
        """Additively splat living particles into the energy field E."""
        alive = self.life > self.age
        if not np.any(alive):
            return
        h, w = E.shape[:2]
        p = self.pos[alive]
        xs = p[:, 0].astype(np.int32)
        ys = p[:, 1].astype(np.int32)
        ok = (xs >= 1) & (xs < w - 1) & (ys >= 1) & (ys < h - 1)
        if not np.any(ok):
            return
        xs, ys = xs[ok], ys[ok]
        frac = 1.0 - self.age[alive][ok] / self.life[alive][ok]
        # Twinkle: each mote breathes at its own tempo
        tw = 0.55 + 0.45 * np.sin(self.age[alive][ok] * 9.0 + self.phase[alive][ok])
        inten = (frac ** 1.4) * tw
        col = self.col[alive][ok] * inten[:, None] * 2.4
        # Plus-shaped splat so motes read as glowing points, not pixels
        for dx, dy, g in ((0, 0, 1.0), (1, 0, 0.35), (-1, 0, 0.35), (0, 1, 0.35), (0, -1, 0.35)):
            np.add.at(E, (ys + dy, xs + dx), col * g)


class PaintCanvas:
    """A breathing field of light.  step() returns the next projector frame."""

    def __init__(self, width, height, fps=30, rng=None):
        self.w = int(width)
        self.h = int(height)
        self.fps = fps
        self.dt = 1.0 / fps
        self.rng = rng if rng is not None else np.random.default_rng()

        # Energy field: linear-light RGB, tonemapped on output
        self.E = np.zeros((self.h, self.w, 3), dtype=np.float32)

        self.mood = "aurora"
        self.phase = self.rng.uniform(0, 1)     # palette river position
        self.phase_speed = 0.0021               # palette drift per frame
        self.brush_gain = 1.0
        self.set_memory(12.0)

        self.particles = _Particles(700, self.rng)

        # Flow field (the slow wind).  Coarse field upsampled to remap maps.
        self._flow_seed = self.rng.uniform(0, 100, size=6).astype(np.float32)
        self._map_x = None
        self._map_y = None
        self._base_x, self._base_y = np.meshgrid(
            np.arange(self.w, dtype=np.float32),
            np.arange(self.h, dtype=np.float32),
        )

        # Brush sprite: bright core + wide soft skirt (precomputed, rescaled per use)
        self._sprite = self._make_sprite(43)
        self._sprite_cache = {}     # width-keyed rescales (performers redraw a lot)
        self._ones3 = np.ones((1, 3), dtype=np.float32)   # fast channel-sum kernel

        # Shared glow-line layer (quarter res): performers draw wireframes
        # here with cv2.line; step() composites the whole layer in one
        # blur+upscale+add, so ten boxes cost the same as one.
        self._line_buf = np.zeros((self.h // 4, self.w // 4, 3), np.float32)
        self._line_dirty = False
        # Lines live in their own fast-fading field: no drift, no
        # diffusion, ~0.35s half-life — wireframes stay crisp while the
        # painting behind them keeps its long dreamy memory.
        self.E_line = np.zeros((self.h, self.w, 3), dtype=np.float32)
        self._line_decay = 0.5 ** (1.0 / (fps * 0.35))

        self.t = 0                  # frame counter
        self.idle_frames = 10 ** 9  # start "long idle" so dreams can begin softly
        self.idle_delay = int(18 * fps)
        self.dreams_enabled = True
        self._wisp_gain = 0.0
        self._wisp_seed = self.rng.uniform(0, 100, size=8).astype(np.float32)

        self._dissolve_frames = 0

    # ── Configuration ───────────────────────────────────────────────────

    def set_mood(self, name):
        if name in MOODS:
            self.mood = name

    def set_memory(self, seconds):
        """How long light lingers — half-life of the energy field."""
        seconds = max(1.0, float(seconds))
        self.memory_seconds = seconds
        self.decay = 0.5 ** (1.0 / (self.fps * seconds * 0.5))

    def set_brush_gain(self, gain):
        self.brush_gain = float(np.clip(gain, 0.1, 4.0))

    # ── Input ───────────────────────────────────────────────────────────

    def stroke(self, x, y, prev=None, intensity=1.0, speed=0.0, seed=0.0):
        """Paint a light stroke ending at (x, y) in projector coords.

        prev: previous point (x, y) for continuous ribbons, or None.
        intensity: 0..1 brightness of the source light.
        speed: projector px/frame — modulates ribbon width and ember shedding.
        seed: stable per-stroke value so each stroke owns a palette offset.
        """
        self.idle_frames = 0
        color = _palette_color(self.mood, self.phase + seed * GOLDEN)

        # Slow light pools wide; fast light draws a tight ribbon
        width_scale = float(np.clip(1.55 - speed / 55.0, 0.5, 1.55))
        gain = 0.85 * self.brush_gain * float(np.clip(intensity, 0.05, 1.5))

        if prev is None:
            self._stamp(x, y, color, gain, width_scale)
        else:
            px, py = prev
            dist = float(np.hypot(x - px, y - py))
            step_len = max(4.0, 10.0 * width_scale)
            n = max(1, int(dist / step_len))
            for i in range(1, n + 1):
                f = i / n
                self._stamp(px + (x - px) * f, py + (y - py) * f,
                            color, gain, width_scale)

        # Fast gestures shed embers in their wake
        shed = min(3, int(speed / 14.0))
        if shed > 0:
            pos = np.tile([x, y], (shed, 1)).astype(np.float32)
            pos += self.rng.normal(0, 5, size=(shed, 2)).astype(np.float32)
            vel = self.rng.normal(0, 26, size=(shed, 2)).astype(np.float32)
            cols = np.tile(color * 0.9, (shed, 1))
            self.particles.spawn(pos, vel, cols, life_range=(1.2, 3.2))

    def add_mist(self, mist, gain=1.0):
        """Add body-motion mist: float32 mono image at projector resolution."""
        if mist is None:
            return
        amount = float(mist.sum())
        if amount > 60.0:           # real presence, not sensor noise
            self.idle_frames = 0
        m = MOODS[self.mood]
        hue = m["hues"][int(self.phase * len(m["hues"])) % len(m["hues"])]
        r, g, b = colorsys.hsv_to_rgb(hue / 360.0, m["mist_sat"], 1.0)
        color = np.array([r, g, b], dtype=np.float32)
        # Mist is a whisper: brake hard so it hovers well below stroke light
        lum = cv2.transform(self.E, self._ones3)
        brake = np.clip(1.0 - lum / 1.6, 0.0, 1.0)
        self.E += (mist * brake)[:, :, None] * color[None, None, :] * (0.05 * gain)

    # Public hooks for performers (autonomous show behaviors) — these
    # deliberately do NOT reset the idle clock: performers are ambient,
    # and only real human light should wake the canvas from dreaming.

    def palette(self, offset=0.0):
        """Current mood color at a phase offset (float32 RGB in 0..1)."""
        return _palette_color(self.mood, self.phase + offset)

    def dab(self, x, y, color, gain=0.5, width=1.0):
        """Place one soft mark of light without registering as input."""
        self._stamp(x, y, color, gain, width)

    def motes(self, positions, velocities, colors, life=(1.5, 4.0)):
        """Spawn glowing motes (fireflies/bubbles/embers)."""
        self.particles.spawn(np.asarray(positions, np.float32),
                             np.asarray(velocities, np.float32),
                             np.asarray(colors, np.float32),
                             life_range=life)

    def glow_line(self, p, q, color, gain=0.4, thickness=1.0):
        """Draw one glowing line segment (canvas coords, sub-pixel).

        Batched: all lines drawn this frame are composited together in
        step().  This is the primitive for wireframe performers.
        """
        shift = 4                   # fixed-point sub-pixel positioning
        sc = 16.0 / 4.0             # 2**shift / quarter-res factor
        p_ = (int(round(p[0] * sc)), int(round(p[1] * sc)))
        q_ = (int(round(q[0] * sc)), int(round(q[1] * sc)))
        col = (float(color[0]) * gain, float(color[1]) * gain,
               float(color[2]) * gain)
        cv2.line(self._line_buf, p_, q_, col,
                 max(1, int(round(thickness))), cv2.LINE_AA, shift=shift)
        self._line_dirty = True

    def release(self):
        """Let it go: the painting dissolves upward into fireflies."""
        lum = cv2.transform(self.E, self._ones3)
        ys, xs = np.nonzero(lum > 0.35)
        if len(ys) > 0:
            k = min(240, len(ys))
            pick = self.rng.choice(len(ys), size=k, replace=False)
            ys, xs = ys[pick], xs[pick]
            pos = np.stack([xs, ys], axis=1).astype(np.float32)
            vel = self.rng.normal(0, 18, size=(k, 2)).astype(np.float32)
            vel[:, 1] -= 22.0       # they rise
            cols = self.E[ys, xs]
            cols = cols / np.maximum(cols.max(axis=1, keepdims=True), 1e-3)
            self.particles.spawn(pos, vel, cols.astype(np.float32),
                                 life_range=(2.0, 5.0))
        self._dissolve_frames = int(2.4 * self.fps)

    # ── Simulation ──────────────────────────────────────────────────────

    def step(self):
        """Advance one frame; returns the uint8 RGB projector frame."""
        self.t += 1
        self.idle_frames += 1
        self.phase = (self.phase + self.phase_speed) % 1.0

        self._advect()

        # Fade — memory melting faster while a release is in progress
        d = self.decay * (0.955 if self._dissolve_frames > 0 else 1.0)
        if self._dissolve_frames > 0:
            self._dissolve_frames -= 1
        self.E *= d

        # Diffuse: light softens into washes as it ages
        if self.t % 3 == 0:
            blur = cv2.GaussianBlur(self.E, (0, 0), 1.7)
            cv2.addWeighted(self.E, 0.72, blur, 0.28, 0, dst=self.E)

        self._dream()

        # Composite this frame's glow lines (wireframe performers)
        self.E_line *= self._line_decay
        if self._line_dirty:
            lb = cv2.GaussianBlur(self._line_buf, (0, 0), 1.1)
            up = cv2.resize(lb, (self.w, self.h),
                            interpolation=cv2.INTER_LINEAR)
            lum = cv2.transform(self.E_line, self._ones3)
            brake = np.clip(1.0 - lum / (self.E_SATURATE * 1.2), 0.0, 1.0)
            self.E_line += up * brake[:, :, None]
            self._line_buf[:] = 0
            self._line_dirty = False

        # Old bright regions occasionally exhale a firefly
        if self.t % 9 == 0:
            self._shed_from_field()

        self.particles.step(self.dt, self.w, self.h)
        self.particles.render_into(self.E, self.t)

        np.clip(self.E, 0, 7.0, out=self.E)
        return self._compose()

    # ── Internals ───────────────────────────────────────────────────────

    def _make_sprite(self, radius):
        yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1].astype(np.float32)
        d2 = xx * xx + yy * yy
        s = radius / 3.2
        core = np.exp(-d2 / (2 * s * s))
        skirt = 0.33 * np.exp(-d2 / (2 * (s * 2.6) ** 2))
        # Feather to exact zero at the radius — otherwise the skirt is
        # truncated by the square patch and accumulated stamps leave
        # hard-edged rectangular pedestals in the field.
        d = np.sqrt(d2)
        window = np.clip((radius - d) / (radius * 0.35), 0.0, 1.0)
        window = window * window * (3.0 - 2.0 * window)     # smoothstep
        return ((core + skirt) * window).astype(np.float32)

    # Light pools toward this level, never past it — like film, fresh
    # marks are vivid but lingering light saturates into a rich bloom
    # instead of blowing out to a white slab.
    E_SATURATE = 3.2

    def _stamp(self, x, y, color, gain, width_scale):
        sp = self._sprite
        if abs(width_scale - 1.0) > 0.08:
            key = int(round(width_scale * 20))
            sp = self._sprite_cache.get(key)
            if sp is None:
                new = max(9, int(self._sprite.shape[0] * key / 20.0)) | 1
                sp = cv2.resize(self._sprite, (new, new),
                                interpolation=cv2.INTER_LINEAR)
                if len(self._sprite_cache) > 48:
                    self._sprite_cache.clear()
                self._sprite_cache[key] = sp
        r = sp.shape[0] // 2
        xi, yi = int(round(x)), int(round(y))
        x0, x1 = max(0, xi - r), min(self.w, xi + r + 1)
        y0, y1 = max(0, yi - r), min(self.h, yi + r + 1)
        if x0 >= x1 or y0 >= y1:
            return
        sx0, sy0 = x0 - (xi - r), y0 - (yi - r)
        patch = sp[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)]
        region = self.E[y0:y1, x0:x1]
        brake = np.clip(1.0 - region.max(axis=2) / self.E_SATURATE, 0.0, 1.0)
        region += (patch * brake)[:, :, None] * (color * gain)[None, None, :]

    def _advect(self):
        """Drift the whole field on a slow, evolving wind."""
        if self._map_x is None or self.t % 6 == 0:
            gw, gh = 28, 16
            gx, gy = np.meshgrid(
                np.linspace(0, 4.5, gw, dtype=np.float32),
                np.linspace(0, 3.0, gh, dtype=np.float32),
            )
            s = self._flow_seed
            tt = self.t * self.dt
            fx = (np.sin(gy * 1.9 + tt * 0.21 + s[0]) +
                  0.6 * np.sin(gx * 1.3 - tt * 0.13 + s[1])) * 0.42
            fy = (np.sin(gx * 1.7 - tt * 0.17 + s[2]) +
                  0.6 * np.sin(gy * 1.1 + tt * 0.11 + s[3])) * 0.34 - 0.14
            fx = cv2.resize(fx, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            fy = cv2.resize(fy, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            # Sample "upwind" so content moves along the flow
            self._map_x = self._base_x - fx
            self._map_y = self._base_y - fy
        self.E = cv2.remap(self.E, self._map_x, self._map_y,
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def _dream(self):
        """When nobody paints, a wisp quietly draws to itself."""
        want = self.dreams_enabled and self.idle_frames > self.idle_delay
        target = 0.35 if want else 0.0
        self._wisp_gain += (target - self._wisp_gain) * 0.012
        if self._wisp_gain < 0.01:
            return
        s = self._wisp_seed
        tt = self.t * self.dt
        x = self.w * (0.5 + 0.34 * np.sin(0.23 * tt + s[0]) +
                      0.11 * np.sin(0.61 * tt + s[1]))
        y = self.h * (0.5 + 0.30 * np.sin(0.19 * tt + s[2]) +
                      0.12 * np.sin(0.47 * tt + s[3]))
        pulse = 0.6 + 0.4 * np.sin(0.7 * tt + s[4])
        color = _palette_color(self.mood, self.phase + 0.31)
        self._stamp(x, y, color, 0.5 * self._wisp_gain * pulse, 0.8)
        # And a stray mote now and then, wandering in from nowhere
        if self.t % 45 == 0:
            pos = self.rng.uniform([0, 0], [self.w, self.h], size=(2, 2)).astype(np.float32)
            vel = self.rng.normal(0, 10, size=(2, 2)).astype(np.float32)
            cols = np.tile(_palette_color(self.mood, self.phase + 0.5) * 0.7, (2, 1))
            self.particles.spawn(pos, vel, cols, life_range=(3.0, 6.0))

    def _shed_from_field(self):
        lum = cv2.transform(self.E, self._ones3)
        ys, xs = np.nonzero(lum > 1.4)
        if len(ys) == 0:
            return
        k = min(2, len(ys))
        pick = self.rng.choice(len(ys), size=k, replace=False)
        ys, xs = ys[pick], xs[pick]
        pos = np.stack([xs, ys], axis=1).astype(np.float32)
        vel = self.rng.normal(0, 12, size=(k, 2)).astype(np.float32)
        vel[:, 1] -= 8.0
        cols = self.E[ys, xs]
        cols = cols / np.maximum(cols.max(axis=1, keepdims=True), 1e-3)
        self.particles.spawn(pos, vel, cols.astype(np.float32), life_range=(1.5, 3.5))

    def _compose(self):
        """Bloom + filmic tonemap + sparkle -> uint8 RGB frame."""
        total = cv2.add(self.E, self.E_line)
        # Wide bloom computed at quarter resolution (cheap, dreamy)
        small = cv2.resize(total, (self.w // 4, self.h // 4),
                           interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 3.0)
        glow = cv2.resize(small, (self.w, self.h), interpolation=cv2.INTER_LINEAR)

        # -1.1 * (total + 0.7*glow) folded into one pass
        neg_L = cv2.addWeighted(total, -1.1, glow, -0.77, 0)
        e = cv2.exp(neg_L)
        # (1 - e) * 255, saturating to uint8
        frame = cv2.convertScaleAbs(e, alpha=-255.0, beta=255.0)

        # Sparkle: transient glints on bright light (found at quarter res)
        lum_s = cv2.transform(small, self._ones3)
        ys, xs = np.nonzero(lum_s > 1.6)
        if len(ys) > 0:
            k = min(36, len(ys))
            pick = self.rng.choice(len(ys), size=k, replace=False)
            jy = self.rng.integers(0, 4, size=k)
            jx = self.rng.integers(0, 4, size=k)
            sy = np.minimum(ys[pick] * 4 + jy, self.h - 1)
            sx = np.minimum(xs[pick] * 4 + jx, self.w - 1)
            frame[sy, sx] = 255
        return frame
