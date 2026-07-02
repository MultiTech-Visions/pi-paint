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

## Scenes — the making of a setup

Setting up at a venue is real work: aim the projector, pick displays, run the calibration scan, tune the feel. **Scenes** make that work durable.

The **Settings** tab covers the whole making of a scene:

- **Displays** — choose which screen gets the control panel and which the projector, with an *Identify Displays* button that flashes each screen's number. Applies live.
- **Camera** — detect connected cameras, pick device and resolution.
- **Projector** — set output resolution (invalidates the calibration scan, which is checked against resolution on load).
- **Calibration timing** — pattern wait and capture delay for the structured light scan.
- **Scenes** — save the entire setup (all of the above + the painting feel + the calibration scan itself) under a name like `wedding_wall` or `living_room`. Load it back anytime; the last-used scene is restored automatically at startup, so the box powers on ready to paint.

Scenes live in `scenes/<name>/` as a `scene.json` plus that scene's `calibration_data.npz`.

## The Show — set it and forget it

Autonomous mode, in **The Show** tab. Press *Begin the Show* and walk away:

1. **Calibrates itself** — runs the structured light scan if there isn't one.
2. **Observes** — goes dark for a breath and photographs the raw scene.
3. **Directs** — segments the view (warped into projector space, so it reasons about exactly what the projector can touch) into surfaces, then a *director* casts a show: a mood, a tempo, a one-line theme, and performers assigned to surfaces.
4. **Performs** — and every few minutes it blinks, looks again, and re-directs, so the show follows the room as it changes. Humans can still paint over it with their own lights the whole time.

Two directors:

- **Instinct** (default) — built-in CV rules, fully offline, zero setup. Dark expanses become aquariums, distinct objects get their silhouettes traced in light, big walls host auroras, warm corners breathe and spark.
- **Local VLM** — point it at Ollama or any OpenAI-compatible server running a small vision model on the Pi (`moondream`, `llava-phi3`, `qwen2.5vl`). The model sees the scene photo plus the surface data and writes the program — including the theme line. Model output is validated against reality (hallucinated behaviors dropped, bad region ids remapped), and *any* failure falls back to instinct. The show always goes on.

Performers: `aurora_drift`, `contour_trace` (light forever tracing the outline of a physical object it found), `fireflies`, `breathing_glow`, `fish_tank`, and `perspective_box` — a wireframe box whose depth breathes in and out of the surface, a fake-3D illusion you can throw on any wall. Its `perspective` option (`left`/`right`/`up`/`down`/`center`, default `auto`) sets which way the depth recedes, so the illusion matches the angle the surface is seen from — a box on the right side of the room should lean differently than one on the left. `auto` recedes toward the canvas center (correct one-point perspective for a viewer facing the middle); the VLM director can choose it per surface.

## Mesh — one world across many projectors

Enable **Mesh** on units sharing a network, number them left-to-right, and they become one long canvas. The far-off dream: a fence line of these, a fish entering the leftmost projector and swimming out the rightmost, crossing every seam in sync.

How it works (`mesh.py`): nothing streams frames. Units discover each other by UDP broadcast and share three small things — a **world layout** (each unit owns a slice of a common strip), a **world clock** (the leader broadcasts time; followers track it within a few ms), and a **show state** (seed + mood). Since every performer's trajectory is a pure function of *(seed, world time)*, each unit independently computes identical fish and renders only its slice. Twenty units cost the same bandwidth as two. Units joining or leaving heal the layout automatically.

### Dual calibration — units learn where they overlap

Once the boxes are placed, press **Calibrate the Mesh** on any one unit (the rest follow automatically). Units take turns flashing gray code patterns while every other unit's camera watches. An observer that can see a neighbor's light decodes camera→neighbor-projector, combines it with its own structured-light calibration, and fits a robust transform: *exactly where the neighbor's canvas sits in its own canvas coordinates* (sub-pixel in simulation).

These measured relations are shared over the mesh and two things happen:

- **The world layout snaps to real geometry.** Offsets come from measurement instead of butt-joints, so a fish crossing a seam lines up exactly. Units that can't see each other simply measure nothing and stay placed by position number.
- **Seams are blended.** Each unit feathers its output across the measured overlap (smoothstep ramps), so strips covered by two projectors don't glow double — projected light adds physically, and the blend keeps the composite even.

Timing rides on the mesh world clock: the flasher broadcasts a capture cue per pattern with settle/dwell windows (`dual_calibration.settle_ms/dwell_ms`) sized to absorb camera latency.

Try both dreams headless:

```bash
python demo_autopilot.py    # director casts a show for a synthetic room,
                            # then 3 simulated units render one fish tank —
                            # composite strips show fish crossing the seams
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
- [x] Settings & scene profiles (displays, camera, projector, saved setups, startup restore)
- [x] Autonomous mode (self-calibration, scene analysis, director, performers)
- [x] Local VLM director (Ollama / OpenAI-compatible, instinct fallback)
- [x] Multi-unit mesh sync (discovery, world clock, shared-world rendering)
- [x] Mesh dual-calibration (measured overlaps, geometric layout, edge blending)
- [ ] Hailo-8L scene analysis (YOLOv8-seg surfaces for the director)

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
| `scenes.dir` | Where scene profiles are stored |
| `scenes.current` | Scene restored automatically at startup |
| `director.backend` | `instinct` (built-in), `ollama`, or `openai` (any compatible server) |
| `director.url` / `director.model` | Where the local VLM lives and which model directs |
| `director.redirect_interval_sec` | How often the show re-observes the scene (0 = never) |
| `mesh.enabled` / `mesh.position` | Join the shared world; this unit's place in the line |
| `mesh.port` / `mesh.broadcast` | Mesh transport (UDP broadcast on the local network) |
| `mesh.overlap_px` | Manual trim where neighboring projections overlap |
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
├── scene_manager.py     # Scene profiles: saved setups (config + calibration bundles)
├── scene_director.py    # Surface analysis + show directors (instinct CV / local VLM)
├── performers.py        # Show behaviors: auroras, contour tracing, fireflies, fish
├── autopilot.py         # Set-it-and-forget-it state machine (calibrate/observe/direct/perform)
├── mesh.py              # Multi-unit shared world: UDP discovery, world clock, layout
├── dual_calibration.py  # Units flash/watch each other to measure real overlaps
├── demo_painting.py     # Headless demo/benchmark -- try the feel on any machine
├── demo_autopilot.py    # Headless demo: autonomous direction + multi-projector fish tank
├── vj/                  # Standalone VJ mode (pygame) for live visuals
├── config.json          # Hardware and agent configuration
├── setup.sh             # One-shot install script
└── run.sh               # Launch wrapper
```

## Multi-Unit Mesh (Planned)

Multiple Pi Paint units will be able to discover each other on a local network, share positional awareness of their projection areas, and synchronize to create unified visualizations across overlapping or adjacent surfaces.

## License

TBD
