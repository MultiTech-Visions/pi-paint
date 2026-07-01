"""Headless demo / smoke test for the light painting experience.

No projector, camera, or Qt required.  Simulates a person painting
with a flashlight (a bright moving dot in synthetic camera frames),
runs the full tracker -> canvas pipeline, and writes preview frames
plus timing stats.  Useful for tuning the feel on any machine and for
sanity-checking performance headroom before running on the Pi.

Usage:
    python demo_painting.py [output_dir]
"""

import os
import sys
import time

import numpy as np
import cv2

from paint_canvas import PaintCanvas
from light_tracker import LightTracker

W, H = 854, 480             # camera / projector resolution
FPS = 30
SECONDS = 14
RENDER_SCALE = 0.75         # canvas simulated below projector res (as in engine)
CW, CH = int(W * RENDER_SCALE), int(H * RENDER_SCALE)


def synth_camera_frame(t, rng):
    """A dark room with a hand-held light tracing a figure."""
    frame = rng.normal(14, 4, size=(H, W, 3)).clip(0, 255).astype(np.uint8)
    # Flashlight path: a slow lissajous with a wandering tempo
    if 1.0 < t < 9.5:
        x = W * (0.5 + 0.33 * np.sin(0.9 * t) + 0.08 * np.sin(2.3 * t))
        y = H * (0.5 + 0.30 * np.sin(0.6 * t + 1.2))
        cv2.circle(frame, (int(x), int(y)), 7, (255, 255, 255), -1)
        cv2.circle(frame, (int(x), int(y)), 13, (140, 140, 140), 3)
    # A second light joins for a while
    if 4.0 < t < 8.0:
        x = W * (0.5 + 0.28 * np.cos(1.1 * t))
        y = H * (0.5 + 0.26 * np.cos(0.8 * t + 0.5))
        cv2.circle(frame, (int(x), int(y)), 6, (250, 250, 250), -1)
    return frame


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "demo_frames"
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(7)
    canvas = PaintCanvas(CW, CH, fps=FPS, rng=np.random.default_rng(11))
    canvas.set_mood("aurora")
    canvas.set_memory(10)
    canvas.idle_delay = int(2.5 * FPS)      # dreams arrive quickly in the demo
    tracker = LightTracker(W, H, CW, CH)    # uncalibrated: proportional mapping

    last_out = None
    times = []
    total = SECONDS * FPS
    for i in range(total):
        t = i / FPS
        cam = synth_camera_frame(t, rng)

        t0 = time.perf_counter()
        result = tracker.process(cam, projected_rgb=last_out,
                                 want_mist=(i % 2 == 0))
        for light in result["lights"]:
            canvas.stroke(light["x"], light["y"], prev=light["prev"],
                          intensity=light["intensity"], speed=light["speed"],
                          seed=light["seed"])
        if result["mist"] is not None:
            canvas.add_mist(result["mist"], gain=2.0)
        if i == int(10.0 * FPS):
            canvas.release()                # the "Let It Go" moment
        out = canvas.step()
        times.append(time.perf_counter() - t0)

        last_out = out
        if i % 15 == 0:
            cv2.imwrite(os.path.join(out_dir, f"frame_{i:04d}.png"),
                        cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

    times = np.array(times) * 1000
    print(f"Rendered {total} frames, canvas {CW}x{CH} (projector {W}x{H})")
    print(f"Per-frame pipeline: mean {times.mean():.1f} ms, "
          f"p95 {np.percentile(times, 95):.1f} ms "
          f"(budget {1000 / FPS:.1f} ms for {FPS} fps)")
    print(f"Preview frames written to {out_dir}/")


if __name__ == "__main__":
    main()
