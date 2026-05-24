# pi-paint VJ

Manual VJ rig for Raspberry Pi 5 + mini wireless keyboard + projector. A
standalone pygame app — it does not import from the rest of pi-paint, so
it can be lifted into its own repo at any time.

This is **Phase 1**: manual mode only, no audio. Phase 2 (aubio beat
detection + auto mode) and Phase 3 (Hailo person-matte) are planned but
not built yet.

## Install

```bash
cd vj
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py                       # windowed, primary display
python main.py --fullscreen          # fullscreen on primary
python main.py --fullscreen --display 1   # fullscreen on second display (the projector)
python main.py --width 1280 --height 720  # bump output res
```

Drop `.mp4` files into `assets/clips/` and `assets/overlays/` before
launching — see the READMEs in those folders.

## Keyboard map

The layout matches a standard QWERTY keyboard. Designed for the Tosuny /
Rii mini wireless keyboards (~70 keys + trackpad).

| Keys                | Action                                              |
|---------------------|-----------------------------------------------------|
| `1 2 3 4 5 6 7 8 9 0` | Pick base clip from `assets/clips/` (slot 1-10)   |
| `Q W E R T Y U I O P` | Toggle overlay from `assets/overlays/` (slot 1-10) |
| `A S D`             | Generative base: plasma / tunnel / starfield (toggle) |
| `Z`                 | HIT: strobe flash                                  |
| `X`                 | HIT: black flash                                   |
| `C`                 | HIT: invert flash                                  |
| `V`                 | HIT: zoom punch                                    |
| `B`                 | HIT: RGB smash                                     |
| `F1`                | FX: kaleidoscope (segments controlled by mouse X)  |
| `F2`                | FX: horizontal mirror                              |
| `F3`                | FX: feedback / trails (mouse XY = zoom / rotate)   |
| `F4`                | FX: invert (persistent)                            |
| `F5`                | FX: posterize (mouse Y = levels)                   |
| `F6`                | FX: edge detect                                    |
| `F7`                | FX: RGB split / chromatic aberration               |
| `Space`             | Blackout toggle (panic button)                     |
| `Backspace`         | Freeze frame toggle                                |
| `Esc`               | Kill all FX, overlays, hits — back to clean base   |
| `Shift+Esc`         | Quit                                               |
| **Trackpad XY**     | Controls active-FX parameters (kaleido segments, feedback zoom/rotate, etc.) |

Press a clip key (1-0) and a generative key (A/S/D) — last one wins.
Press an overlay key once to turn it on, again to turn it off.

## Asset sources

Free libraries (download once, no internet needed at the party):

- **Base clips:** [Beeple VJ Loops](https://www.beeple-crap.com/vjloops),
  [Mantissa CC0 4K](https://mantissa.xyz/vj.html),
  [Videezy VJ Loops](https://www.videezy.com/free-video/vj-loop)
- **Overlays (fire, sparks, lasers, lens flares):**
  [Videezy Spark Overlays](https://www.videezy.com/free-video/spark-overlay),
  [Vecteezy Sparks](https://www.vecteezy.com/free-videos/sparks-overlay)

For overlays: pick footage that's already pre-keyed against a **black**
background. The compositor uses screen-blend, so anything bright pops
through and the black drops out. Don't bother with true alpha-channel
video on Pi — the hardware decoder doesn't like it.

Recommended pre-processing (one-time, on a desktop):

```bash
# Downsample to projector resolution + re-encode to H.264 for hardware decode
ffmpeg -i input.mp4 -vf scale=854:480 -c:v libx264 -preset slow -crf 22 -an output.mp4
```

## Architecture

```
main.py        argparse + pygame init + main loop wiring
engine.py      Engine class: state, render pipeline, public actions
effects.py     Generative + transformative numpy/OpenCV effects
clips.py       ClipPool: lazy MP4 loader keyed by slot index
keymap.py      Pygame key → engine action dispatch table
config.py      Config dataclass
```

Render pipeline each frame:

```
base layer   →  active clip OR active generative OR black
   ↓
FX chain     →  kaleido, mirror, rgb_split, posterize, edges, invert, feedback
   ↓
overlay      →  screen-blend the active overlay clip
   ↓
hits         →  transient strobe/flash/punch (5 frames)
   ↓
blit to pygame screen
```

`prev_frame` is captured before the next render so feedback works.

## Performance notes

- Targets 30 fps at 854×480 on Pi 5. Generatives are vectorized numpy.
- `kaleidoscope` is the heaviest effect (per-pixel remap). Stack 2-3
  effects max for headroom.
- MP4 decode uses OpenCV's `VideoCapture` — relies on libavcodec; on
  Pi 5 it does software H.264 decode but stays well under a frame budget
  for 854×480.
- Clip frames are read **once per render**, so don't try to play more
  than one clip slot simultaneously — only the most recently selected
  base and overlay are advanced.

## Not yet built (Phase 2/3)

- **Auto mode:** aubio beat detector listening on a USB mic, scenes
  swap on downbeats, hits fire on kick onsets, FX intensity scales
  with smoothed RMS energy.
- **Hailo person matte:** silhouette the dancer/DJ from the webcam,
  composite over the base — fire and lasers "shoot from hands."
- **Pre-baked Shadertoy MP4s:** bake favorite shaders offline to
  854×480 H.264 loops so the rig can play them back without live GLSL.
