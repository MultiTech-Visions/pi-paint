"""Autopilot — set it down, turn it on, walk away.

The state machine behind autonomous mode:

    CALIBRATING  no scan yet?  run the structured light scan itself
    OBSERVING    go dark for a breath, photograph the raw scene
    DIRECTING    analyze surfaces + ask the director (VLM or instinct)
                 in a background thread
    PERFORMING   cast the performers, run the show — humans can still
                 paint over it with their own lights
    (every redirect_interval, it blinks dark, looks again, and
     re-directs — the show follows the room as it changes)

If mesh is enabled, the unit joins the shared world: the leader
publishes seed/mood, every unit renders its slice, and world-aware
performers (the fish) swim across all of them in sync.
"""

import threading

import numpy as np
from PyQt5.QtCore import QObject, QTimer, pyqtSignal
from PyQt5.QtGui import QColor

import performers as perf
from scene_director import analyze_surfaces, draw_surface_overlay, make_director
from mesh import MeshNode


class Autopilot(QObject):
    state_changed = pyqtSignal(str)         # state name for the UI
    program_ready = pyqtSignal(dict)        # the director's decision
    scene_ready = pyqtSignal(object)        # director's-eye view (RGB)
    mesh_changed = pyqtSignal(str)          # mesh status line

    def __init__(self, config, get_paint_engine, get_calibration,
                 latest_frame_fn, ensure_camera_fn, start_scan_fn, projector):
        """
        get_paint_engine / get_calibration : callables returning the
            *current* engines (the panel recreates them on scene changes).
        ensure_camera_fn : callable -> bool, starts the camera if needed.
        start_scan_fn : callable, kicks off a calibration scan through
            the panel (so its camera housekeeping applies).
        """
        super().__init__()
        self.config = config
        self.engine = get_paint_engine
        self.calibration = get_calibration
        self.latest_frame = latest_frame_fn
        self.ensure_camera = ensure_camera_fn
        self.start_scan = start_scan_fn
        self.projector = projector

        self.state = "off"
        self.running = False
        self.program = None
        self.mesh = None
        self._direct_thread = None

        self._redirect_timer = QTimer()
        self._redirect_timer.setSingleShot(True)
        self._redirect_timer.timeout.connect(self._begin_observation)

        self._mesh_timer = QTimer()
        self._mesh_timer.timeout.connect(self._mesh_tick)

        # Cross-thread: director finishes in a worker thread; Qt queues
        # the signal back onto the GUI thread.
        self.program_ready.connect(self._on_program)

    # ── Lifecycle ───────────────────────────────────────────────────────

    def start(self):
        if self.running:
            return
        self.running = True
        self._start_mesh()

        have_camera = self.ensure_camera()
        if not have_camera:
            # No eyes: still put on a show (dreams + default program)
            self._set_state("performing")
            self._apply_program(self._blind_program(), [], None)
            return

        if not self.calibration().has_calibration:
            self._set_state("calibrating")
            self.calibration().scan_finished.connect(self._on_scan_finished)
            self.start_scan()
            if not self.calibration().is_scanning:
                # Scan refused (camera hiccup?) — carry on uncalibrated
                try:
                    self.calibration().scan_finished.disconnect(
                        self._on_scan_finished)
                except TypeError:
                    pass
                self._begin_observation()
        else:
            self._begin_observation()

    def stop(self):
        self.running = False
        self._redirect_timer.stop()
        self._mesh_timer.stop()
        eng = self.engine()
        eng.set_performers([])
        eng.stop()
        self.projector.show_solid(QColor(0, 0, 0))
        if self.mesh is not None:
            self.mesh.stop()
            self.mesh = None
        self._set_state("off")

    def _set_state(self, state):
        self.state = state
        self.state_changed.emit(state)

    # ── Calibration step ────────────────────────────────────────────────

    def _on_scan_finished(self, success, message):
        try:
            self.calibration().scan_finished.disconnect(self._on_scan_finished)
        except TypeError:
            pass
        if not self.running:
            return
        # Calibrated or not, carry on — uncalibrated is proportional
        self._begin_observation()

    # ── Observation: a dark breath, then a photograph ───────────────────

    def _begin_observation(self):
        if not self.running:
            return
        self._set_state("observing")
        eng = self.engine()
        was_painting = eng.is_running
        eng.stop()
        self.projector.show_solid(QColor(0, 0, 0))
        self._was_painting = was_painting
        QTimer.singleShot(700, self._capture_and_direct)

    def _capture_and_direct(self):
        if not self.running:
            return
        frame = self.latest_frame()
        if frame is None:
            self._set_state("performing")
            self._apply_program(self._blind_program(), [], None)
            return
        frame = frame.copy()
        self._set_state("directing")

        eng = self.engine()
        cal = self.calibration()
        proj_w, proj_h = eng.canvas_w, eng.canvas_h

        def work():
            try:
                if cal.has_calibration:
                    inv_x, inv_y = cal.build_inverse_maps()
                else:
                    inv_x = inv_y = None
                surfaces, view = analyze_surfaces(frame, proj_w, proj_h,
                                                  inv_x, inv_y)
                director = make_director(self.config)
                program = director.direct(surfaces, view)
            except Exception as e:  # noqa: BLE001 — the show must go on
                surfaces, view = [], None
                program = self._blind_program()
                program["notes"] = f"scene analysis failed ({e})"
            program["_surfaces"] = surfaces
            program["_view"] = view
            self.program_ready.emit(program)

        self._direct_thread = threading.Thread(target=work, daemon=True)
        self._direct_thread.start()

    def _on_program(self, program):
        if not self.running or self.state != "directing":
            return
        surfaces = program.pop("_surfaces", [])
        view = program.pop("_view", None)
        if view is not None:
            self.scene_ready.emit(draw_surface_overlay(view, surfaces))
        self._apply_program(program, surfaces, view)
        self._set_state("performing")

    # ── Performing ──────────────────────────────────────────────────────

    def _apply_program(self, program, surfaces, view):
        self.program = program
        eng = self.engine()
        canvas = eng.canvas

        # Mesh: the leader decides for everyone; followers adopt
        world = None
        if self.mesh is not None:
            if self.mesh.is_leader:
                self.mesh.set_show(mood=program["mood"],
                                   tempo=program["tempo"])
            else:
                show = self.mesh.show_state()
                if show.get("mood"):
                    program["mood"] = show["mood"]
                if show.get("tempo") is not None:
                    program["tempo"] = show["tempo"]
            world = self.mesh.world()

        canvas.set_mood(program["mood"])
        tempo = program.get("tempo", 0.5)
        cast = []
        for i, spec in enumerate(program.get("behaviors", [])):
            cast.append(perf.create(spec, surfaces, eng.canvas_w,
                                    eng.canvas_h, tempo=tempo,
                                    world=world, seed=i * 101 + 7))
        eng.set_performers(cast)
        if not eng.is_running:
            eng.start()

        interval = int(self.config.get("director", {})
                       .get("redirect_interval_sec", 180))
        if interval > 0:
            self._redirect_timer.start(interval * 1000)
        self.program_ready_snapshot = dict(program)

    def _blind_program(self):
        """No camera / no scene: an ambient default so the show still runs."""
        return {
            "theme": "painting from memory — no eyes tonight",
            "mood": self.config.get("painting", {}).get("mood", "aurora"),
            "tempo": 0.45,
            "behaviors": [{"type": "aurora_drift", "region": None},
                          {"type": "fireflies", "region": None},
                          {"type": "fish_tank", "region": None}],
            "source": "blind",
            "notes": "",
        }

    # ── Mesh ────────────────────────────────────────────────────────────

    def _start_mesh(self):
        mcfg = self.config.get("mesh", {})
        if not mcfg.get("enabled", False):
            return
        eng = self.engine()
        self.mesh = MeshNode(
            port=int(mcfg.get("port", 45454)),
            broadcast=mcfg.get("broadcast", "255.255.255.255"),
            position=int(mcfg.get("position", 0)),
            width_px=eng.canvas_w,
            overlap_px=float(mcfg.get("overlap_px", 0)),
        )
        self._mesh_timer.start(2000)

    def _mesh_tick(self):
        if self.mesh is None:
            return
        offset, world_w, n = self.mesh.layout()
        role = "leader" if self.mesh.is_leader else "follower"
        self.mesh_changed.emit(
            f"{n} unit(s) — this one at {offset:.0f}px of a "
            f"{world_w:.0f}px world ({role})")
        # Followers keep their canvas mood in step with the leader
        if not self.mesh.is_leader and self.state == "performing":
            show = self.mesh.show_state()
            if show.get("mood"):
                self.engine().canvas.set_mood(show["mood"])
