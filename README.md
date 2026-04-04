# Pi Paint

Autonomous projection mapping system built for Raspberry Pi 5 with Hailo-8L AI accelerator. Point it at a scene, turn it on, and let it go -- it analyzes the environment, identifies surfaces and objects, generates creative mapping ideas, and projects animated visualizations in real time.

Multiple units can mesh together with positional awareness to create synchronized, large-scale projection installations.

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
- [ ] Structured light calibration (camera-projector mapping)
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
| `agent.model` | Claude model for the AI brain |
| `agent.refresh_interval_sec` | How often the agent re-evaluates the scene |

## Project Structure

```
pi-paint/
├── main.py              # Entry point -- display detection, window routing
├── control-panel.py     # Control UI with tabbed interface (Camera, Calibration, Scene Analysis, Projection, Settings)
├── projector-window.py  # Fullscreen projector output window
├── config.json          # Hardware and agent configuration
├── setup.sh             # One-shot install script
└── run.sh               # Launch wrapper
```

## Multi-Unit Mesh (Planned)

Multiple Pi Paint units will be able to discover each other on a local network, share positional awareness of their projection areas, and synchronize to create unified visualizations across overlapping or adjacent surfaces.

## License

TBD
