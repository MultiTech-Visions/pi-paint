# Pi Paint

Autonomous projection mapping system built for Raspberry Pi 5 with Hailo-8L AI accelerator. Point it at a scene, turn it on, and let it go -- it analyzes the environment, identifies surfaces and objects, generates creative mapping ideas, and projects animated visualizations in real time.

Multiple units can mesh together with positional awareness to create synchronized, large-scale projection installations.

## Light Painting

The heart of the project: an interactive light painting experience designed for one thing — **mesmerization**.

Bring any light to the projection surface — a phone flashlight, a candle, a glowing toy — and the wall remembers where light touched it, blooming into living ribbons of color exactly there. Because the structured light calibration maps every camera pixel to its projector pixel, the response lands *on the physical spot you touched*, which is what makes it read as magic instead of a screen.

What makes it feel alive:

- **The canvas breathes.** Trails don't just fade — they drift on a slow, evolving wind, soften into aurora washes, shed fireflies as they age, and sparkle where the light pools.
- **Color is a river, not a picker.** Each mood (aurora, embers, moonlight, prism, biolume) is a drifting palette. The same gesture is never the same color twice.
- **Light behaves like film.** Fresh strokes are vivid; lingering light pools into a rich bloom instead of blowing out. Fast sweeps draw tight ribbons and shed embers; slow ones pool wide.
- **Your body paints too.** Motion leaves a faint mist even without a light, so waving a hand stirs the wall.
- **It dreams when alone.** Left idle, a faint wisp quietly draws to itself, inviting people in. Real light makes it yield instantly.
- **"Let It Go."** One button dissolves the whole painting upward into a swarm of fireflies.
- **It never sees itself.** The engine predicts the projector's own contribution to the camera image and subtracts it before detecting anything — no feedback loops, no ghosts. It reacts to *your* light only.

Use the **Light Painting** tab in the control panel: start the camera, run a calibration scan once, then Begin Painting. It also runs uncalibrated (proportional mapping) if you just want to play.

Try the feel on any machine — no camera, projector, or Pi needed:

```bash
python demo_painting.py          # simulates a flashlight, writes preview frames + timing
```

## How It Works

```
Camera (OpenCV)
    |
    v
Calibration ──── Structured light scanning, camera-to-projector coordinate mapping
    |
    v
Scene Analysis ── Hailo-8L + YOLOv8-seg object/surface detection
    |
    v
AI Agent ──────── Claude generates animation concepts based on scene geometry
    |
    v
Animation Engine ─ Renders and drives projector output
    |
    v
Projector (PyQt5 framebuffer)
```

## Modes

- **Auto** -- Fully autonomous. The system continuously scans, analyzes, and projects without intervention.
- **Manual** -- Step through each stage yourself: capture, calibrate, analyze, design, and project. Influence the output at every step.

## Hardware

| Component | Role |
|-----------|------|
| Raspberry Pi 5 | Main compute |
| Hailo-8L AI HAT | On-device object detection / segmentation |
| USB camera | Scene capture |
| Pico projector | Projection output |

The system is designed to run fully offline after setup.

## Project Status

Early stage. The foundation is in place -- the rest is being built out.

- [x] Multi-display management (control panel + projector)
- [x] Camera capture and live preview
- [x] Projector output (solid colors, images, patterns, test patterns)
- [x] Configuration system
- [x] Setup and install scripts
- [x] Structured light calibration (camera-projector mapping)
- [x] Interactive light painting (living canvas, light tracking, motion mist, dreams)
- [ ] Hailo-8L scene analysis (YOLOv8-seg)
- [ ] Claude AI agent integration (animation brain)
- [ ] Animation / visualization engine
- [ ] Multi-unit mesh sync
- [ ] Auto / manual mode toggle

## Setup

Requires Raspberry Pi OS (Debian-based). The setup script handles everything:

```bash
git clone https://github.com/multitech-visions/pi-paint.git
cd pi-paint
chmod +x setup.sh
./setup.sh
```

This installs system packages, creates a Python virtual environment, installs Python dependencies, and adds a desktop shortcut.

### Dependencies

**System:** Python 3, PyQt5, OpenCV/FFmpeg libs, OpenGL, v4l-utils, image libs

**Python:** `opencv-python-headless`, `numpy`, `anthropic`, `Pillow`

**Hardware:** `hailo-all` + `hailo_platform` (optional, falls back gracefully)

## Running

```bash
# Using the launcher script
./run.sh

# Or directly
source venv/bin/activate
python main.py
```

A desktop shortcut ("Light Painter") is also created during setup.

## Configuration

Edit `config.json` to match your hardware:

```json
{
    "display": {
        "control_display": 0,
        "projector_display": 1
    },
    "projector": {
        "width": 854,
        "height": 480
    },
    "camera": {
        "device_index": 0,
        "width": 854,
        "height": 480
    },
    "calibration": {
        "gray_code_wait_ms": 500,
        "capture_delay_ms": 300
    },
    "agent": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-20250514",
        "refresh_interval_sec": 15
    }
}
```

| Key | Description |
|-----|-------------|
| `display.control_display` | Screen index for the control panel UI |
| `display.projector_display` | Screen index for projector output |
| `projector.width/height` | Projector resolution |
| `camera.device_index` | `/dev/video` index for USB camera |
| `camera.width/height` | Camera capture resolution |
| `calibration.*` | Timing for structured light calibration |
| `painting.mood` | Starting palette: aurora, embers, moonlight, prism, biolume |
| `painting.memory_seconds` | How long light lingers on the canvas |
| `painting.render_scale` | Canvas simulation scale (0.75 default; lower = faster on Pi, dreamier) |
| `painting.motion_mist` | Bodies stir the wall even without a light |
| `painting.dreams` | Idle wisp that paints to itself when nobody's there |
| `agent.model` | Claude model for the AI brain |
| `agent.refresh_interval_sec` | How often the agent re-evaluates the scene |

## Project Structure

```
pi-paint/
├── main.py              # Entry point -- display detection, window routing
├── control_panel.py     # Control UI with tabbed interface (Camera, Calibration, Light Painting, ...)
├── projector_window.py  # Fullscreen projector output window
├── calibration.py       # Structured light scan engine + camera<->projector maps
├── gray_code.py         # Gray code pattern generation/decoding
├── paint_canvas.py      # The living canvas: energy field, moods, fireflies, dreams (pure numpy/cv2)
├── light_tracker.py     # Light + motion detection with projector self-suppression (pure numpy/cv2)
├── paint_engine.py      # Qt heartbeat: camera -> tracker -> canvas -> projector
├── demo_painting.py     # Headless demo/benchmark -- try the feel on any machine
├── vj/                  # Standalone VJ mode (pygame) for live visuals
├── config.json          # Hardware and agent configuration
├── setup.sh             # One-shot install script
└── run.sh               # Launch wrapper
```

## Multi-Unit Mesh (Planned)

Multiple Pi Paint units will be able to discover each other on a local network, share positional awareness of their projection areas, and synchronize to create unified visualizations across overlapping or adjacent surfaces.

## License

TBD
