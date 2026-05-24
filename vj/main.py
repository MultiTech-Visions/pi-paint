import argparse
import os
import sys

import pygame

from config import Config
from engine import Engine


def parse_args():
    p = argparse.ArgumentParser(description="pi-paint VJ — manual mode")
    p.add_argument("--width", type=int, default=854)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--fullscreen", action="store_true")
    p.add_argument("--display", type=int, default=0,
                   help="Display index to render on (0 = primary)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        width=args.width, height=args.height, fps=args.fps,
        fullscreen=args.fullscreen, display=args.display,
    )

    pygame.init()
    flags = pygame.FULLSCREEN | pygame.SCALED if cfg.fullscreen else 0
    try:
        screen = pygame.display.set_mode(
            (cfg.width, cfg.height), flags, display=cfg.display
        )
    except TypeError:
        # Older pygame without display kwarg
        os.environ.setdefault("SDL_VIDEO_FULLSCREEN_DISPLAY", str(cfg.display))
        screen = pygame.display.set_mode((cfg.width, cfg.height), flags)

    pygame.display.set_caption("pi-paint VJ")
    pygame.mouse.set_visible(False)

    print(f"[vj] window {cfg.width}x{cfg.height} fullscreen={cfg.fullscreen} display={cfg.display}")
    print(f"[vj] clips dir:    {cfg.clips_dir}")
    print(f"[vj] overlays dir: {cfg.overlays_dir}")

    engine = Engine(cfg, screen)
    print(f"[vj] {len(engine.clips)} clip(s), {len(engine.overlays)} overlay(s) loaded")
    print("[vj] keys: 1-0 clips · QWERTY overlays · ASD generative · ZXCVB hits · F1-F7 FX · Space blackout · Esc kill · Shift+Esc quit")

    try:
        engine.run()
    finally:
        pygame.quit()
        sys.exit(0)


if __name__ == "__main__":
    main()
