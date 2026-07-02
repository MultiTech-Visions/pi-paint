"""Headless demo of autonomous mode and the multi-unit fish tank.

Part 1 — the director: builds a synthetic room photo (dark floor,
lit wall, a hanging shape, a warm lamp), lets the scene analysis and
the instinct director cast a show, then performs it on a canvas and
writes frames — including the director's-eye view with surfaces
outlined and numbered.

Part 2 — the mesh dream: three simulated units side by side share one
world (same seed, same clock, adjacent offsets — exactly what mesh.py
negotiates over UDP).  Each renders its own canvas independently; the
composite strip shows fish swimming seamlessly across all three
projectors.

Usage:
    python demo_autopilot.py [output_dir]
"""

import os
import sys
import time

import numpy as np
import cv2

from paint_canvas import PaintCanvas
from scene_director import (analyze_surfaces, draw_surface_overlay,
                            InstinctDirector)
import performers as perf

FPS = 30
CW, CH = 640, 360


def synth_room():
    """A believable little room for the director to look at."""
    img = np.zeros((480, 854, 3), np.uint8)
    img[:] = (34, 38, 44)                                   # dim wall
    img[300:, :] = (20, 16, 14)                             # dark floor
    cv2.rectangle(img, (90, 70), (300, 250), (60, 70, 78), -1)     # poster
    cv2.circle(img, (620, 150), 70, (40, 90, 150), -1)             # warm lamp glow
    pts = np.array([[430, 120], [510, 90], [560, 180], [470, 230]])
    cv2.fillPoly(img, [pts], (52, 48, 90))                  # a hanging shape
    noise = np.random.default_rng(0).normal(0, 4, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def part1_director(out_dir):
    print("── Part 1: the director looks at a room ──")
    frame = synth_room()
    surfaces, view = analyze_surfaces(frame, CW, CH)
    program = InstinctDirector().direct(surfaces, view)

    print(f"surfaces: {len(surfaces)}")
    for s in surfaces:
        print(f"  #{s['id']}: {s['area_frac']*100:4.1f}% area, "
              f"brightness {s['brightness']:.2f}, texture {s['edge_density']:.2f}")
    print(f"theme:  “{program['theme']}”")
    print(f"mood:   {program['mood']}  tempo: {program['tempo']:.2f}")
    print(f"cast:   {[b['type'] + '@' + str(b['region']) for b in program['behaviors']]}")

    overlay = draw_surface_overlay(view, surfaces)
    cv2.imwrite(os.path.join(out_dir, "directors_eye.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    canvas = PaintCanvas(CW, CH, fps=FPS, rng=np.random.default_rng(3))
    canvas.set_mood(program["mood"])
    canvas.dreams_enabled = False
    cast = [perf.create(b, surfaces, CW, CH, tempo=program["tempo"], seed=i * 101)
            for i, b in enumerate(program["behaviors"])]
    cast = [c for c in cast if c is not None]

    times = []
    for i in range(12 * FPS):
        t = i / FPS
        t0 = time.perf_counter()
        for c in cast:
            c.step(canvas, 1.0 / FPS, t)
        out = canvas.step()
        times.append(time.perf_counter() - t0)
        if i % 60 == 0:
            cv2.imwrite(os.path.join(out_dir, f"show_{i:04d}.png"),
                        cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print(f"show pipeline: {np.mean(times)*1000:.1f} ms/frame\n")


class _SimWorld:
    """What mesh.py negotiates over UDP, simulated for the demo:
    shared seed, shared clock, adjacent offsets."""

    def __init__(self, offset, world_w, seed, clock):
        self.offset_px = offset
        self.world_w = world_w
        self.seed = seed
        self._clock = clock

    def now(self):
        return self._clock[0]

    def to_canvas(self, x, y):
        return x - self.offset_px, y


def _smoothstep_ramp(width, left, right):
    ramp = np.ones(width, np.float32)
    if left > 0:
        t = np.linspace(0.0, 1.0, left, dtype=np.float32)
        ramp[:left] = t * t * (3.0 - 2.0 * t)
    if right > 0:
        t = np.linspace(1.0, 0.0, right, dtype=np.float32)
        ramp[-right:] = np.minimum(ramp[-right:], t * t * (3.0 - 2.0 * t))
    return ramp


def part2_mesh_fish(out_dir):
    """Three units whose projections OVERLAP by 120px, as dual
    calibration would measure — each edge-blends across the measured
    overlap, and the composite (light adds physically where projectors
    overlap) is seamless."""
    print("── Part 2: one fish tank across three overlapping projectors ──")
    n_units = 3
    overlap = 120
    offsets = [u * (CW - overlap) for u in range(n_units)]
    world_w = offsets[-1] + CW
    seed = 4242
    clock = [0.0]

    units = []
    for u in range(n_units):
        canvas = PaintCanvas(CW, CH, fps=FPS, rng=np.random.default_rng(50 + u))
        canvas.set_mood("biolume")
        canvas.dreams_enabled = False
        world = _SimWorld(offsets[u], world_w, seed, clock)
        tank = perf.FishTank(None, CW, CH, tempo=0.6, seed=7, world=world,
                             n_fish=7)
        left = overlap if u > 0 else 0
        right = overlap if u < n_units - 1 else 0
        ramp = _smoothstep_ramp(CW, left, right)
        units.append((canvas, tank, ramp))

    for i in range(14 * FPS):
        clock[0] = i / FPS
        world_img = np.zeros((CH, world_w, 3), np.float32)
        for u, (canvas, tank, ramp) in enumerate(units):
            tank.step(canvas, 1.0 / FPS, clock[0])
            out = canvas.step().astype(np.float32) * ramp[None, :, None]
            world_img[:, offsets[u]:offsets[u] + CW] += out   # light adds
        if i % 45 == 0:
            strip = np.clip(world_img, 0, 255).astype(np.uint8)
            for off in offsets[1:]:                # overlap zone markers
                strip[-4:, off:off + overlap] = (40, 40, 60)
            cv2.imwrite(os.path.join(out_dir, f"mesh_{i:04d}.png"),
                        cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    print(f"wrote blended composites ({world_w}x{CH}, {overlap}px measured "
          f"overlaps) to {out_dir}/")


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "demo_autopilot_frames"
    os.makedirs(out_dir, exist_ok=True)
    part1_director(out_dir)
    part2_mesh_fish(out_dir)


if __name__ == "__main__":
    main()
