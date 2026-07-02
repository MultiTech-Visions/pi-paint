import json

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QPushButton, QLabel, QStatusBar,
    QProgressBar, QGroupBox, QComboBox, QSlider, QCheckBox,
    QLineEdit, QSpinBox, QScrollArea, QApplication, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QColor
import cv2
import numpy as np
from calibration import CalibrationEngine
from paint_engine import PaintEngine
from paint_canvas import MOOD_NAMES
from scene_manager import SceneManager
from autopilot import Autopilot
from mesh import MeshNode
from dual_calibration import DualCalibrator

CONFIG_PATH = "config.json"


class ControlPanel(QMainWindow):
    def __init__(self, config, screen, projector_win):
        super().__init__()
        self.config = config
        self.target_screen = screen
        self.projector = projector_win

        self.setWindowTitle("Light Painter - Control Panel")
        self.setMinimumSize(800, 480)

        # Camera setup
        self.camera = None
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self._update_camera_frame)

        # Scene profiles
        scenes_cfg = config.get("scenes", {})
        self.scenes = SceneManager(scenes_cfg.get("dir", "scenes"))

        # Calibration + painting engines (recreated when the scene's
        # resolutions change, hence the helpers)
        self.last_frame = None
        self.calibration = self._make_calibration_engine()
        self.paint_engine = self._make_paint_engine()

        # Mesh node: owned here so it outlives shows and serves dual
        # calibration; created lazily by _ensure_mesh()
        self.mesh_node = None

        # Autopilot: the set-it-and-forget-it show director
        self.autopilot = Autopilot(
            config,
            get_paint_engine=lambda: self.paint_engine,
            get_calibration=lambda: self.calibration,
            latest_frame_fn=self._latest_camera_frame,
            ensure_camera_fn=self._ensure_camera,
            start_scan_fn=self._start_calibration_scan,
            projector=projector_win,
            get_mesh=self._ensure_mesh,
        )
        self.autopilot.state_changed.connect(self._on_autopilot_state)
        self.autopilot.program_ready.connect(self._on_autopilot_program)
        self.autopilot.scene_ready.connect(self._on_autopilot_scene)
        self.autopilot.mesh_changed.connect(self._on_mesh_status)

        # Dual calibrator: always listening, so this unit participates
        # when any box on the mesh leads a calibration
        self.dual_cal = DualCalibrator(
            config,
            get_mesh=lambda: self.mesh_node,
            projector=projector_win,
            latest_frame_fn=self._latest_camera_frame,
            get_calibration=lambda: self.calibration,
            get_paint_engine=lambda: self.paint_engine,
            prepare_fn=self._prepare_for_dual_cal,
        )
        self.dual_cal.status.connect(self._on_mesh_status)
        self.dual_cal.finished.connect(self._on_dual_cal_finished)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready")

        # Central widget with tabs
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Build tabs
        self.tabs.addTab(self._build_camera_tab(), "Camera")
        self.tabs.addTab(self._build_calibration_tab(), "Calibration")
        self.tabs.addTab(self._build_painting_tab(), "Light Painting")
        self.tabs.addTab(self._build_projection_tab(), "The Show")
        self.tabs.addTab(self._build_scene_tab(), "Scene Analysis")
        self.tabs.addTab(self._build_settings_tab(), "Settings")

        self._position_on_screen()

    def _position_on_screen(self):
        if self.target_screen is not None:
            geo = self.target_screen.geometry()
            self.move(geo.x(), geo.y())

    # ── Engine lifecycle ────────────────────────────────────────

    def _make_calibration_engine(self):
        cal = CalibrationEngine(self.config, self._grab_camera_frame, self.projector)
        cal.progress.connect(self._on_calibration_progress)
        cal.scan_finished.connect(self._on_calibration_finished)
        cal.preview_ready.connect(self._on_calibration_preview)
        return cal

    def _make_paint_engine(self):
        eng = PaintEngine(self.config, self._latest_camera_frame,
                          self.projector, self.calibration)
        eng.preview_ready.connect(self._on_paint_preview)
        eng.status.connect(self._on_paint_status)
        eng.video_geometry_fn = self._video_geometry
        eng.video_clock_fn = self._video_clock
        return eng

    def _recreate_engines(self):
        """Rebuild both engines after a resolution/scene change."""
        was_painting = self.paint_engine.is_running
        video_path = (self.paint_engine.video.path
                      if self.paint_engine.video is not None else None)
        if self.calibration.is_scanning:
            self.calibration.cancel_scan()
        self.paint_engine.clear_video()
        self.paint_engine.stop()

        self.calibration = self._make_calibration_engine()
        self.paint_engine = self._make_paint_engine()
        if video_path:
            self.paint_engine.set_video(
                video_path,
                self.config.get("video", {}).get("brightness", 1.0))

        # Reload the working calibration if it matches the new resolution
        if self.calibration.load_calibration():
            self.calib_status_label.setText("Loaded saved calibration")
            self._update_calib_viz()
        else:
            self.calib_preview.setPixmap(QPixmap())
            self.calib_preview.setText("No calibration data")
            self.calib_status_label.setText(
                "No calibration for this resolution — run a scan")

        if was_painting:
            self._start_painting()
        else:
            self.btn_paint_start.setEnabled(True)
            self.btn_paint_stop.setEnabled(False)

    # ── Camera Tab ──────────────────────────────────────────────

    def _build_camera_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.camera_preview = QLabel("Camera not started")
        self.camera_preview.setAlignment(Qt.AlignCenter)
        self.camera_preview.setMinimumSize(640, 360)
        self.camera_preview.setStyleSheet("background-color: #1a1a1a; color: white;")
        layout.addWidget(self.camera_preview)

        btn_row = QHBoxLayout()

        self.btn_cam_start = QPushButton("Start Camera")
        self.btn_cam_start.clicked.connect(self._start_camera)
        btn_row.addWidget(self.btn_cam_start)

        self.btn_cam_stop = QPushButton("Stop Camera")
        self.btn_cam_stop.clicked.connect(self._stop_camera)
        self.btn_cam_stop.setEnabled(False)
        btn_row.addWidget(self.btn_cam_stop)

        btn_test = QPushButton("Test Pattern → Projector")
        btn_test.clicked.connect(self.projector.show_test_pattern)
        btn_row.addWidget(btn_test)

        btn_black = QPushButton("Black → Projector")
        btn_black.clicked.connect(lambda: self.projector.show_solid(QColor(0, 0, 0)))
        btn_row.addWidget(btn_black)

        layout.addLayout(btn_row)
        return tab

    def _start_camera(self):
        cam_idx = self.config["camera"]["device_index"]
        cam_w = self.config["camera"]["width"]
        cam_h = self.config["camera"]["height"]

        self.camera = cv2.VideoCapture(cam_idx)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)

        if not self.camera.isOpened():
            self.status.showMessage(f"ERROR: Cannot open camera at index {cam_idx}")
            self.camera = None
            return

        self.camera_timer.start(33)  # ~30fps
        self.btn_cam_start.setEnabled(False)
        self.btn_cam_stop.setEnabled(True)
        self.status.showMessage(f"Camera started (index {cam_idx})")

    def _stop_camera(self):
        self.camera_timer.stop()
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.last_frame = None
        self.camera_preview.setText("Camera stopped")
        self.btn_cam_start.setEnabled(True)
        self.btn_cam_stop.setEnabled(False)
        self.status.showMessage("Camera stopped")

    def _update_camera_frame(self):
        if self.camera is None:
            return
        ret, frame = self.camera.read()
        if not ret:
            self.status.showMessage("WARNING: Failed to read camera frame")
            return
        self.last_frame = frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        scaled = QPixmap.fromImage(img).scaled(
            self.camera_preview.width(),
            self.camera_preview.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.camera_preview.setPixmap(scaled)

    def _grab_camera_frame(self):
        """Grab a single frame from the camera for calibration capture."""
        if self.camera is None:
            return False, None
        ret, frame = self.camera.read()
        return ret, frame

    def _latest_camera_frame(self):
        """Most recent frame from the preview loop (no device contention)."""
        return self.last_frame

    def _ensure_camera(self):
        """Start the camera if it isn't running; True when available."""
        if self.camera is None:
            self._start_camera()
        return self.camera is not None

    # ── Mesh node lifecycle ─────────────────────────────────────

    def _ensure_mesh(self):
        """Bring the mesh node up if enabled; returns it (or None)."""
        mcfg = self.config.get("mesh", {})
        if not mcfg.get("enabled", False):
            return None
        if self.mesh_node is None:
            self.mesh_node = MeshNode(
                port=int(mcfg.get("port", 45454)),
                broadcast=mcfg.get("broadcast", "255.255.255.255"),
                position=int(mcfg.get("position", 0)),
                width_px=self.paint_engine.canvas_w,
                overlap_px=float(mcfg.get("overlap_px", 0)),
            )
        return self.mesh_node

    def _stop_mesh(self):
        if self.mesh_node is not None:
            self.mesh_node.stop()
            self.mesh_node = None

    # ── Video wall wiring ───────────────────────────────────────

    def _video_geometry(self):
        """World geometry for the video layer: mesh transform when the
        mesh is up, None (canvas-filling default) otherwise."""
        if self.mesh_node is None:
            return None
        A, t = self.mesh_node.world_transform()
        _, world_w, _ = self.mesh_node.layout()
        return A, t, world_w, self.paint_engine.canvas_h

    def _video_clock(self):
        """Video time source: the mesh world clock keeps every unit on
        the same frame; None falls back to the local show clock."""
        if self.mesh_node is None:
            return None
        return self.mesh_node.now()

    # ── Dual calibration ────────────────────────────────────────

    def _prepare_for_dual_cal(self):
        """Go quiet for a mesh calibration: no show, no painting, and
        the camera up if we have one."""
        if self.autopilot.running:
            self._stop_show()
        if self.paint_engine.is_running:
            self._stop_painting()
        return self._ensure_camera() and self.calibration.has_calibration

    def _calibrate_mesh(self):
        node = self._ensure_mesh()
        if node is None:
            self.status.showMessage("Enable mesh first")
            return
        self.btn_mesh_calibrate.setEnabled(False)
        self.status.showMessage("Dual calibration starting — this unit leads")
        self.dual_cal.lead()

    def _on_dual_cal_finished(self, success, message):
        self.btn_mesh_calibrate.setEnabled(True)
        self.status.showMessage(f"Dual calibration: {message}")

    # ── Calibration Tab ────────────────────────────────────────

    def _build_calibration_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # ── Preview area ──
        self.calib_preview = QLabel("No calibration data")
        self.calib_preview.setAlignment(Qt.AlignCenter)
        self.calib_preview.setMinimumSize(640, 360)
        self.calib_preview.setStyleSheet("background-color: #1a1a1a; color: #888;")
        layout.addWidget(self.calib_preview)

        # ── Progress bar ──
        self.calib_progress = QProgressBar()
        self.calib_progress.setVisible(False)
        layout.addWidget(self.calib_progress)

        self.calib_status_label = QLabel("")
        self.calib_status_label.setStyleSheet("color: #aaa; font-size: 12px;")
        layout.addWidget(self.calib_status_label)

        # ── Scan controls ──
        scan_group = QGroupBox("Structured Light Scan")
        scan_layout = QVBoxLayout(scan_group)

        scan_desc = QLabel(
            "Projects Gray code patterns through the projector and captures them\n"
            "with the camera to build a pixel-level camera↔projector mapping."
        )
        scan_desc.setStyleSheet("color: #aaa; font-size: 11px;")
        scan_desc.setWordWrap(True)
        scan_layout.addWidget(scan_desc)

        scan_btn_row = QHBoxLayout()

        self.btn_start_scan = QPushButton("Start Scan")
        self.btn_start_scan.clicked.connect(self._start_calibration_scan)
        scan_btn_row.addWidget(self.btn_start_scan)

        self.btn_cancel_scan = QPushButton("Cancel Scan")
        self.btn_cancel_scan.clicked.connect(self._cancel_calibration_scan)
        self.btn_cancel_scan.setEnabled(False)
        scan_btn_row.addWidget(self.btn_cancel_scan)

        self.btn_load_calib = QPushButton("Load Previous")
        self.btn_load_calib.clicked.connect(self._load_calibration)
        scan_btn_row.addWidget(self.btn_load_calib)

        scan_layout.addLayout(scan_btn_row)
        layout.addWidget(scan_group)

        # ── Visualization controls ──
        viz_group = QGroupBox("Mapping Visualization")
        viz_layout = QVBoxLayout(viz_group)

        viz_row = QHBoxLayout()

        self.calib_viz_mode = QComboBox()
        self.calib_viz_mode.addItems([
            "Coordinate Map (HSV)",
            "Confidence Map",
            "Projector Coverage"
        ])
        self.calib_viz_mode.currentIndexChanged.connect(self._update_calib_viz)
        viz_row.addWidget(QLabel("View:"))
        viz_row.addWidget(self.calib_viz_mode)

        viz_layout.addLayout(viz_row)

        # Alignment test buttons
        align_row = QHBoxLayout()

        btn_checker = QPushButton("Checkerboard → Projector")
        btn_checker.clicked.connect(self.projector.show_checkerboard)
        align_row.addWidget(btn_checker)

        btn_crosshair = QPushButton("Crosshair → Projector")
        btn_crosshair.clicked.connect(self.projector.show_crosshair)
        align_row.addWidget(btn_crosshair)

        btn_black = QPushButton("Black → Projector")
        btn_black.clicked.connect(lambda: self.projector.show_solid(QColor(0, 0, 0)))
        align_row.addWidget(btn_black)

        viz_layout.addLayout(align_row)
        layout.addWidget(viz_group)

        # Try loading existing calibration on startup
        QTimer.singleShot(500, self._load_calibration_quiet)

        return tab

    def _start_calibration_scan(self):
        """Begin the structured light calibration scan."""
        if self.camera is None:
            self.status.showMessage("ERROR: Start camera first (Camera tab)")
            return
        if self.calibration.is_scanning:
            return

        # Painting would fight the scan for the projector — pause it
        self._painting_was_active = self.paint_engine.is_running
        if self._painting_was_active:
            self._stop_painting()

        self.btn_start_scan.setEnabled(False)
        self.btn_cancel_scan.setEnabled(True)
        self.calib_progress.setVisible(True)
        self.calib_progress.setValue(0)
        self.status.showMessage("Calibration scan started...")

        # Pause the live camera preview during scan to avoid frame contention
        was_running = self.camera_timer.isActive()
        self._camera_was_previewing = was_running
        if was_running:
            self.camera_timer.stop()

        self.calibration.start_scan()

    def _cancel_calibration_scan(self):
        """Cancel the in-progress scan."""
        self.calibration.cancel_scan()

    def _on_calibration_progress(self, current, total, desc):
        """Handle scan progress updates."""
        self.calib_progress.setMaximum(total)
        self.calib_progress.setValue(current)
        self.calib_status_label.setText(desc)

    def _on_calibration_finished(self, success, message):
        """Handle scan completion."""
        self.btn_start_scan.setEnabled(True)
        self.btn_cancel_scan.setEnabled(False)
        self.calib_progress.setVisible(False)
        self.calib_status_label.setText(message)
        self.status.showMessage(message)

        # Resume camera preview if it was running
        if getattr(self, '_camera_was_previewing', False) and self.camera is not None:
            self.camera_timer.start(33)

        if success:
            self._update_calib_viz()
            if self.paint_engine.is_running:
                self.paint_engine.refresh_calibration()

        # Resume painting if the scan interrupted it
        if getattr(self, '_painting_was_active', False):
            self._painting_was_active = False
            self._start_painting()

    def _on_calibration_preview(self, rgb_image):
        """Handle a new calibration preview image."""
        self._show_numpy_on_label(rgb_image, self.calib_preview)

    def _update_calib_viz(self):
        """Update the calibration preview based on selected visualization mode."""
        if not self.calibration.has_calibration:
            return

        mode = self.calib_viz_mode.currentIndex()
        if mode == 0:
            img = self.calibration.build_mapping_preview()
        elif mode == 1:
            img = self.calibration.build_confidence_preview()
        elif mode == 2:
            img = self.calibration.build_coverage_overlay()
            # Also show on projector
            if img is not None:
                self.projector.show_image(img)
        else:
            return

        if img is not None:
            self._show_numpy_on_label(img, self.calib_preview)

    def _load_calibration(self):
        """Load saved calibration data."""
        if self.calibration.load_calibration():
            self.status.showMessage("Loaded saved calibration data")
            self.calib_status_label.setText("Loaded from calibration_data.npz")
            self._update_calib_viz()
        else:
            self.status.showMessage("No saved calibration found")

    def _load_calibration_quiet(self):
        """Try to load calibration data silently on startup."""
        if self.calibration.load_calibration():
            self.calib_status_label.setText("Loaded saved calibration")
            self._update_calib_viz()

    def _show_numpy_on_label(self, rgb_array, label):
        """Display a numpy RGB array on a QLabel."""
        h, w, ch = rgb_array.shape
        img = QImage(rgb_array.data, w, h, ch * w, QImage.Format_RGB888)
        scaled = QPixmap.fromImage(img).scaled(
            label.width(), label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        label.setPixmap(scaled)

    # ── Light Painting Tab ──────────────────────────────────────

    def _build_painting_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.paint_preview = QLabel("The canvas is dark.\n\nStart painting to wake it up.")
        self.paint_preview.setAlignment(Qt.AlignCenter)
        self.paint_preview.setMinimumSize(640, 360)
        self.paint_preview.setStyleSheet("background-color: #0a0a12; color: #667;")
        layout.addWidget(self.paint_preview)

        # ── Main controls ──
        btn_row = QHBoxLayout()

        self.btn_paint_start = QPushButton("Begin Painting")
        self.btn_paint_start.clicked.connect(self._start_painting)
        btn_row.addWidget(self.btn_paint_start)

        self.btn_paint_stop = QPushButton("Stop")
        self.btn_paint_stop.clicked.connect(self._stop_painting)
        self.btn_paint_stop.setEnabled(False)
        btn_row.addWidget(self.btn_paint_stop)

        self.btn_release = QPushButton("Let It Go ✨")
        self.btn_release.setToolTip("The painting dissolves upward into fireflies")
        self.btn_release.clicked.connect(lambda: self.paint_engine.release())
        btn_row.addWidget(self.btn_release)

        layout.addLayout(btn_row)

        # ── The feel ──
        feel_group = QGroupBox("The Feel")
        feel_layout = QVBoxLayout(feel_group)

        mood_row = QHBoxLayout()
        mood_row.addWidget(QLabel("Mood:"))
        self.paint_mood = QComboBox()
        self.paint_mood.addItems([m.capitalize() for m in MOOD_NAMES])
        self.paint_mood.currentIndexChanged.connect(self._on_mood_changed)
        mood_row.addWidget(self.paint_mood)
        mood_row.addStretch()
        feel_layout.addLayout(mood_row)

        mem_row = QHBoxLayout()
        mem_row.addWidget(QLabel("Memory:"))
        self.paint_memory = QSlider(Qt.Horizontal)
        self.paint_memory.setRange(2, 30)
        self.paint_memory.setValue(int(self.config.get("painting", {}).get("memory_seconds", 12)))
        self.paint_memory.valueChanged.connect(self._on_memory_changed)
        mem_row.addWidget(self.paint_memory)
        self.paint_memory_label = QLabel(f"{self.paint_memory.value()}s")
        self.paint_memory_label.setMinimumWidth(36)
        mem_row.addWidget(self.paint_memory_label)
        feel_layout.addLayout(mem_row)

        sens_row = QHBoxLayout()
        sens_row.addWidget(QLabel("Sensitivity:"))
        self.paint_sensitivity = QSlider(Qt.Horizontal)
        self.paint_sensitivity.setRange(3, 20)      # 0.3 .. 2.0
        self.paint_sensitivity.setValue(10)
        self.paint_sensitivity.valueChanged.connect(self._on_sensitivity_changed)
        sens_row.addWidget(self.paint_sensitivity)
        feel_layout.addLayout(sens_row)

        toggle_row = QHBoxLayout()
        self.paint_mist_toggle = QCheckBox("Motion mist (bodies stir the wall)")
        self.paint_mist_toggle.setChecked(True)
        self.paint_mist_toggle.toggled.connect(self._on_mist_toggled)
        toggle_row.addWidget(self.paint_mist_toggle)

        self.paint_dreams_toggle = QCheckBox("Dreams when idle")
        self.paint_dreams_toggle.setChecked(True)
        self.paint_dreams_toggle.toggled.connect(self._on_dreams_toggled)
        toggle_row.addWidget(self.paint_dreams_toggle)
        feel_layout.addLayout(toggle_row)

        layout.addWidget(feel_group)

        hint = QLabel(
            "Paint with any light — a phone flashlight, a candle, a glowing toy. "
            "Run a calibration scan first so trails land exactly where light touches the surface."
        )
        hint.setStyleSheet("color: #667; font-size: 11px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        return tab

    def _start_painting(self):
        if self.camera is None:
            self._start_camera()
        self.paint_engine.start()
        self.btn_paint_start.setEnabled(False)
        self.btn_paint_stop.setEnabled(True)
        self.status.showMessage("Light painting active — bring a light to the surface")

    def _stop_painting(self):
        self.paint_engine.stop()
        self.btn_paint_start.setEnabled(True)
        self.btn_paint_stop.setEnabled(False)
        self.projector.show_solid(QColor(0, 0, 0))
        self.status.showMessage("Light painting stopped")

    def _on_paint_preview(self, frame):
        self._show_numpy_on_label(frame, self.paint_preview)

    def _on_paint_status(self, message):
        self.status.showMessage(message)

    def _on_mood_changed(self, idx):
        self.paint_engine.canvas.set_mood(MOOD_NAMES[idx])
        self.config.setdefault("painting", {})["mood"] = MOOD_NAMES[idx]

    def _on_memory_changed(self, seconds):
        self.paint_engine.canvas.set_memory(seconds)
        self.paint_memory_label.setText(f"{seconds}s")
        self.config.setdefault("painting", {})["memory_seconds"] = seconds

    def _on_sensitivity_changed(self, value):
        self.paint_engine.sensitivity = value / 10.0
        self.config.setdefault("painting", {})["sensitivity"] = value / 10.0

    def _on_mist_toggled(self, checked):
        self.paint_engine.motion_mist = checked
        self.config.setdefault("painting", {})["motion_mist"] = checked

    def _on_dreams_toggled(self, checked):
        self.paint_engine.canvas.dreams_enabled = checked
        self.config.setdefault("painting", {})["dreams"] = checked

    # ── Scene Analysis Tab (placeholder) ────────────────────────

    def _build_scene_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        label = QLabel("Scene analysis will go here.\n\nHailo-8L will run YOLOv8-seg\nto identify objects and surfaces.")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("color: #888; font-size: 14px;")
        layout.addWidget(label)
        return tab

    # ── The Show Tab: autonomous mode ───────────────────────────

    def _build_projection_tab(self):
        outer = QScrollArea()
        outer.setWidgetResizable(True)
        tab = QWidget()
        outer.setWidget(tab)
        layout = QVBoxLayout(tab)

        # Director's-eye view of the scene (surfaces outlined + numbered)
        self.show_scene_preview = QLabel(
            "Autonomous mode.\n\nSet it down, press Begin, walk away:\n"
            "it calibrates, looks at the scene, casts a show, and runs it."
        )
        self.show_scene_preview.setAlignment(Qt.AlignCenter)
        self.show_scene_preview.setMinimumSize(640, 300)
        self.show_scene_preview.setStyleSheet(
            "background-color: #0a0a12; color: #667;")
        layout.addWidget(self.show_scene_preview)

        self.show_state_label = QLabel("state: off")
        self.show_state_label.setStyleSheet("color: #7af; font-size: 12px;")
        layout.addWidget(self.show_state_label)

        self.show_theme_label = QLabel("")
        self.show_theme_label.setStyleSheet(
            "color: #cbd; font-size: 13px; font-style: italic;")
        self.show_theme_label.setWordWrap(True)
        layout.addWidget(self.show_theme_label)

        btn_row = QHBoxLayout()
        self.btn_show_start = QPushButton("Begin the Show")
        self.btn_show_start.clicked.connect(self._start_show)
        btn_row.addWidget(self.btn_show_start)
        self.btn_show_stop = QPushButton("End the Show")
        self.btn_show_stop.clicked.connect(self._stop_show)
        self.btn_show_stop.setEnabled(False)
        btn_row.addWidget(self.btn_show_stop)
        self.btn_show_redirect = QPushButton("Look Again Now")
        self.btn_show_redirect.setToolTip(
            "Blink dark, re-photograph the scene, and re-direct the show")
        self.btn_show_redirect.clicked.connect(self._redirect_show)
        self.btn_show_redirect.setEnabled(False)
        btn_row.addWidget(self.btn_show_redirect)
        layout.addLayout(btn_row)

        # ── Director settings ──
        dir_group = QGroupBox("Director")
        dir_layout = QVBoxLayout(dir_group)

        backend_row = QHBoxLayout()
        backend_row.addWidget(QLabel("Mind:"))
        self.director_backend = QComboBox()
        self.director_backend.addItem("Instinct (built-in, no model)", "instinct")
        self.director_backend.addItem("Ollama (local VLM)", "ollama")
        self.director_backend.addItem("OpenAI-compatible server", "openai")
        self.director_backend.currentIndexChanged.connect(self._on_director_changed)
        backend_row.addWidget(self.director_backend, stretch=1)
        dir_layout.addLayout(backend_row)

        vlm_row = QHBoxLayout()
        vlm_row.addWidget(QLabel("URL:"))
        self.director_url = QLineEdit()
        self.director_url.editingFinished.connect(self._on_director_changed)
        vlm_row.addWidget(self.director_url, stretch=2)
        vlm_row.addWidget(QLabel("Model:"))
        self.director_model = QLineEdit()
        self.director_model.editingFinished.connect(self._on_director_changed)
        vlm_row.addWidget(self.director_model, stretch=1)
        dir_layout.addLayout(vlm_row)

        interval_row = QHBoxLayout()
        interval_row.addWidget(QLabel("Look again every (s):"))
        self.director_interval = QSpinBox()
        self.director_interval.setRange(0, 3600)
        self.director_interval.setSpecialValueText("never")
        self.director_interval.valueChanged.connect(self._on_director_changed)
        interval_row.addWidget(self.director_interval)
        interval_row.addStretch()
        dir_layout.addLayout(interval_row)

        dir_note = QLabel(
            "Instinct always works offline. For a local VLM on the Pi, run "
            "Ollama with a small vision model (moondream, llava-phi3, qwen2.5vl). "
            "Any failure falls back to instinct — the show always goes on."
        )
        dir_note.setStyleSheet("color: #667; font-size: 11px;")
        dir_note.setWordWrap(True)
        dir_layout.addWidget(dir_note)
        layout.addWidget(dir_group)

        # ── Video wall ──
        video_group = QGroupBox("Video Wall")
        video_layout = QVBoxLayout(video_group)

        vsrc_row = QHBoxLayout()
        vsrc_row.addWidget(QLabel("Source:"))
        self.video_source = QLineEdit()
        self.video_source.setPlaceholderText(
            "videos/your_clip.mp4 (same file on every unit)")
        vsrc_row.addWidget(self.video_source, stretch=1)
        self.btn_video_play = QPushButton("Play")
        self.btn_video_play.clicked.connect(self._play_video)
        vsrc_row.addWidget(self.btn_video_play)
        self.btn_video_stop = QPushButton("Stop")
        self.btn_video_stop.clicked.connect(self._stop_video)
        self.btn_video_stop.setEnabled(False)
        vsrc_row.addWidget(self.btn_video_stop)
        video_layout.addLayout(vsrc_row)

        vbr_row = QHBoxLayout()
        vbr_row.addWidget(QLabel("Brightness:"))
        self.video_brightness = QSlider(Qt.Horizontal)
        self.video_brightness.setRange(2, 20)       # 0.2 .. 2.0
        self.video_brightness.setValue(10)
        self.video_brightness.valueChanged.connect(self._on_video_brightness)
        vbr_row.addWidget(self.video_brightness)
        vbr_row.addStretch()
        video_layout.addLayout(vbr_row)

        self.video_status = QLabel("")
        self.video_status.setStyleSheet("color: #667; font-size: 11px;")
        video_layout.addWidget(self.video_status)

        video_note = QLabel(
            "One video across the whole mesh: each unit plays its own copy of "
            "the file, the shared world clock picks the frame, and the "
            "measured calibration warps each unit's slice — nothing is "
            "streamed. Light painting and the show glow over the movie."
        )
        video_note.setStyleSheet("color: #667; font-size: 11px;")
        video_note.setWordWrap(True)
        video_layout.addWidget(video_note)

        layout.addWidget(video_group)

        # ── Mesh ──
        mesh_group = QGroupBox("Mesh (multi-unit world)")
        mesh_layout = QVBoxLayout(mesh_group)

        mesh_row = QHBoxLayout()
        self.mesh_enabled = QCheckBox("Join the shared world")
        self.mesh_enabled.toggled.connect(self._on_mesh_changed)
        mesh_row.addWidget(self.mesh_enabled)
        mesh_row.addWidget(QLabel("Position in line:"))
        self.mesh_position = QSpinBox()
        self.mesh_position.setRange(0, 63)
        self.mesh_position.valueChanged.connect(self._on_mesh_changed)
        mesh_row.addWidget(self.mesh_position)
        mesh_row.addStretch()
        mesh_layout.addLayout(mesh_row)

        cal_mesh_row = QHBoxLayout()
        self.btn_mesh_calibrate = QPushButton("Calibrate the Mesh")
        self.btn_mesh_calibrate.setToolTip(
            "Units take turns flashing patterns while the others watch — "
            "measuring exactly where their projections overlap")
        self.btn_mesh_calibrate.clicked.connect(self._calibrate_mesh)
        cal_mesh_row.addWidget(self.btn_mesh_calibrate)
        cal_mesh_row.addStretch()
        mesh_layout.addLayout(cal_mesh_row)

        self.mesh_status_label = QLabel("mesh off")
        self.mesh_status_label.setStyleSheet("color: #667; font-size: 11px;")
        mesh_layout.addWidget(self.mesh_status_label)

        mesh_note = QLabel(
            "Units on the same network discover each other and share one long "
            "canvas: number them left-to-right and the fish swim across all of "
            "them in sync. Run Calibrate the Mesh once they're placed: each "
            "unit flashes while the others watch, measuring real overlaps so "
            "the world lines up exactly and seams are blended. Press it on "
            "one unit only — the rest follow."
        )
        mesh_note.setStyleSheet("color: #667; font-size: 11px;")
        mesh_note.setWordWrap(True)
        mesh_layout.addWidget(mesh_note)
        layout.addWidget(mesh_group)

        layout.addStretch()
        return outer

    def _start_show(self):
        self.btn_show_start.setEnabled(False)
        self.btn_show_stop.setEnabled(True)
        self.btn_show_redirect.setEnabled(True)
        # Manual painting controls hand over to the autopilot
        if self.paint_engine.is_running:
            self._stop_painting()
        self.autopilot.start()
        self.status.showMessage("Autonomous show starting...")

    def _stop_show(self):
        self.autopilot.stop()
        self.btn_show_start.setEnabled(True)
        self.btn_show_stop.setEnabled(False)
        self.btn_show_redirect.setEnabled(False)
        self.btn_paint_start.setEnabled(True)
        self.btn_paint_stop.setEnabled(False)
        self.show_theme_label.setText("")
        self.status.showMessage("Show ended")

    def _redirect_show(self):
        if self.autopilot.running:
            self.autopilot._begin_observation()

    def _on_autopilot_state(self, state):
        pretty = {
            "off": "off",
            "calibrating": "calibrating — scanning the scene with light",
            "observing": "observing — a dark breath, then a photograph",
            "directing": "directing — deciding what this scene wants",
            "performing": "performing",
        }.get(state, state)
        self.show_state_label.setText(f"state: {pretty}")

    def _on_autopilot_program(self, program):
        theme = program.get("theme", "")
        src = program.get("source", "")
        notes = program.get("notes", "")
        cast = ", ".join(b["type"] for b in program.get("behaviors", []))
        text = f"“{theme}”\n{program.get('mood', '')} · {cast} · by {src}"
        if notes:
            text += f"\n({notes})"
        self.show_theme_label.setText(text)

    def _on_autopilot_scene(self, view_rgb):
        self._show_numpy_on_label(view_rgb, self.show_scene_preview)

    def _on_mesh_status(self, text):
        self.mesh_status_label.setText(text)

    def _on_director_changed(self, *_):
        dcfg = self.config.setdefault("director", {})
        dcfg["backend"] = self.director_backend.currentData()
        dcfg["url"] = self.director_url.text().strip()
        dcfg["model"] = self.director_model.text().strip()
        dcfg["redirect_interval_sec"] = self.director_interval.value()

    def _play_video(self):
        path = self.video_source.text().strip()
        if not path:
            self.status.showMessage("Give the video wall a source file first")
            return
        vcfg = self.config.setdefault("video", {})
        brightness = self.video_brightness.value() / 10.0
        if not self.paint_engine.set_video(path, brightness):
            self.video_status.setText(f"could not open {path}")
            self.status.showMessage(f"ERROR: could not open video '{path}'")
            return
        vcfg["source"] = path
        vcfg["brightness"] = brightness
        self._ensure_mesh()     # if mesh is enabled, sync clock+geometry
        if not self.paint_engine.is_running:
            self.paint_engine.start()
        v = self.paint_engine.video
        self.video_status.setText(
            f"{v.vid_w}x{v.vid_h} @ {v.fps:.0f}fps, {v.duration:.1f}s loop"
            + (" — synced to mesh world clock" if self.mesh_node else
               " — local clock"))
        self.btn_video_play.setEnabled(False)
        self.btn_video_stop.setEnabled(True)
        self.btn_paint_stop.setEnabled(True)
        self.btn_paint_start.setEnabled(False)
        self.status.showMessage("Video wall playing")

    def _stop_video(self):
        self.paint_engine.clear_video()
        self.config.setdefault("video", {})["source"] = ""
        self.video_status.setText("")
        self.btn_video_play.setEnabled(True)
        self.btn_video_stop.setEnabled(False)
        self.status.showMessage("Video wall stopped")

    def _on_video_brightness(self, value):
        self.config.setdefault("video", {})["brightness"] = value / 10.0
        if self.paint_engine.video is not None:
            self.paint_engine.video.set_brightness(value / 10.0)

    def _on_mesh_changed(self, *_):
        mcfg = self.config.setdefault("mesh", {})
        mcfg["enabled"] = self.mesh_enabled.isChecked()
        mcfg["position"] = self.mesh_position.value()
        if not mcfg["enabled"]:
            self._stop_mesh()

    # ── Settings Tab: the making of the scene ───────────────────

    def _build_settings_tab(self):
        outer = QScrollArea()
        outer.setWidgetResizable(True)
        tab = QWidget()
        outer.setWidget(tab)
        layout = QVBoxLayout(tab)

        # ── Scenes: save/restore a whole physical setup ──
        scene_group = QGroupBox("Scenes")
        scene_layout = QVBoxLayout(scene_group)

        scene_desc = QLabel(
            "A scene is one physical setup: displays, camera, projector, the calibration\n"
            "scan, and the feel you dialed in. Save it once at a venue; it all comes back."
        )
        scene_desc.setStyleSheet("color: #aaa; font-size: 11px;")
        scene_desc.setWordWrap(True)
        scene_layout.addWidget(scene_desc)

        save_row = QHBoxLayout()
        self.scene_name_edit = QLineEdit()
        self.scene_name_edit.setPlaceholderText("Scene name (e.g. living_room, wedding_wall)")
        save_row.addWidget(self.scene_name_edit)
        btn_scene_save = QPushButton("Save Scene")
        btn_scene_save.clicked.connect(self._save_scene)
        save_row.addWidget(btn_scene_save)
        scene_layout.addLayout(save_row)

        load_row = QHBoxLayout()
        self.scene_combo = QComboBox()
        load_row.addWidget(self.scene_combo, stretch=1)
        btn_scene_load = QPushButton("Load")
        btn_scene_load.clicked.connect(self._load_scene)
        load_row.addWidget(btn_scene_load)
        btn_scene_delete = QPushButton("Delete")
        btn_scene_delete.clicked.connect(self._delete_scene)
        load_row.addWidget(btn_scene_delete)
        scene_layout.addLayout(load_row)

        self.scene_info_label = QLabel("")
        self.scene_info_label.setStyleSheet("color: #667; font-size: 11px;")
        scene_layout.addWidget(self.scene_info_label)
        self.scene_combo.currentIndexChanged.connect(self._update_scene_info)

        layout.addWidget(scene_group)

        # ── Displays ──
        disp_group = QGroupBox("Displays")
        disp_layout = QVBoxLayout(disp_group)

        disp_row = QHBoxLayout()
        disp_row.addWidget(QLabel("Control panel:"))
        self.settings_control_display = QComboBox()
        disp_row.addWidget(self.settings_control_display)
        disp_row.addWidget(QLabel("Projector:"))
        self.settings_projector_display = QComboBox()
        disp_row.addWidget(self.settings_projector_display)
        disp_layout.addLayout(disp_row)

        disp_btn_row = QHBoxLayout()
        btn_identify = QPushButton("Identify Displays")
        btn_identify.clicked.connect(self._identify_displays)
        disp_btn_row.addWidget(btn_identify)
        btn_apply_disp = QPushButton("Apply Displays")
        btn_apply_disp.clicked.connect(self._apply_displays)
        disp_btn_row.addWidget(btn_apply_disp)
        disp_layout.addLayout(disp_btn_row)

        layout.addWidget(disp_group)

        # ── Camera ──
        cam_group = QGroupBox("Camera")
        cam_layout = QVBoxLayout(cam_group)

        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel("Device:"))
        self.settings_camera_device = QComboBox()
        cam_row.addWidget(self.settings_camera_device)
        btn_detect_cams = QPushButton("Detect Cameras")
        btn_detect_cams.clicked.connect(self._detect_cameras)
        cam_row.addWidget(btn_detect_cams)
        cam_layout.addLayout(cam_row)

        cam_res_row = QHBoxLayout()
        cam_res_row.addWidget(QLabel("Resolution:"))
        self.settings_camera_res = QComboBox()
        self.settings_camera_res.addItems(
            ["640 x 480", "854 x 480", "1280 x 720", "1920 x 1080"])
        cam_res_row.addWidget(self.settings_camera_res)
        btn_apply_cam = QPushButton("Apply Camera")
        btn_apply_cam.clicked.connect(self._apply_camera)
        cam_res_row.addWidget(btn_apply_cam)
        cam_layout.addLayout(cam_res_row)

        layout.addWidget(cam_group)

        # ── Projector ──
        proj_group = QGroupBox("Projector")
        proj_layout = QVBoxLayout(proj_group)

        proj_row = QHBoxLayout()
        proj_row.addWidget(QLabel("Resolution:"))
        self.settings_proj_w = QSpinBox()
        self.settings_proj_w.setRange(320, 3840)
        proj_row.addWidget(self.settings_proj_w)
        proj_row.addWidget(QLabel("x"))
        self.settings_proj_h = QSpinBox()
        self.settings_proj_h.setRange(240, 2160)
        proj_row.addWidget(self.settings_proj_h)
        btn_apply_proj = QPushButton("Apply Projector")
        btn_apply_proj.clicked.connect(self._apply_projector)
        proj_row.addWidget(btn_apply_proj)
        proj_layout.addLayout(proj_row)

        proj_note = QLabel("Changing projector resolution invalidates the calibration scan.")
        proj_note.setStyleSheet("color: #aaa; font-size: 11px;")
        proj_layout.addWidget(proj_note)

        layout.addWidget(proj_group)

        # ── Calibration timing ──
        cal_group = QGroupBox("Calibration Timing")
        cal_layout = QHBoxLayout(cal_group)

        cal_layout.addWidget(QLabel("Pattern wait (ms):"))
        self.settings_cal_wait = QSpinBox()
        self.settings_cal_wait.setRange(100, 3000)
        self.settings_cal_wait.setSingleStep(50)
        self.settings_cal_wait.valueChanged.connect(
            lambda v: self.config["calibration"].__setitem__("gray_code_wait_ms", v))
        cal_layout.addWidget(self.settings_cal_wait)

        cal_layout.addWidget(QLabel("Capture delay (ms):"))
        self.settings_cal_delay = QSpinBox()
        self.settings_cal_delay.setRange(50, 2000)
        self.settings_cal_delay.setSingleStep(50)
        self.settings_cal_delay.valueChanged.connect(
            lambda v: self.config["calibration"].__setitem__("capture_delay_ms", v))
        cal_layout.addWidget(self.settings_cal_delay)
        cal_layout.addStretch()

        layout.addWidget(cal_group)

        # ── Persist ──
        save_cfg_row = QHBoxLayout()
        btn_save_cfg = QPushButton("Save Settings to config.json")
        btn_save_cfg.clicked.connect(self._save_config)
        save_cfg_row.addWidget(btn_save_cfg)
        save_cfg_row.addStretch()
        layout.addLayout(save_cfg_row)

        layout.addStretch()

        self._populate_display_combos()
        self._populate_camera_devices([self.config["camera"]["device_index"]])
        self._refresh_scene_list()
        self._sync_ui_from_config()

        return outer

    # ── Settings: sync helpers ──────────────────────────────────

    def _sync_ui_from_config(self):
        """Push config values into every settings + painting widget."""
        cfg = self.config

        # Displays
        n_screens = self.settings_control_display.count()
        ci = min(cfg["display"]["control_display"], max(0, n_screens - 1))
        self.settings_control_display.setCurrentIndex(ci)
        pi = cfg["display"]["projector_display"]
        # Last item of the projector combo is the windowed fallback
        if pi >= self.settings_projector_display.count() - 1:
            pi = self.settings_projector_display.count() - 1
        self.settings_projector_display.setCurrentIndex(pi)

        # Camera
        dev = cfg["camera"]["device_index"]
        idx = self.settings_camera_device.findData(dev)
        if idx < 0:
            self.settings_camera_device.addItem(f"Camera {dev}", dev)
            idx = self.settings_camera_device.findData(dev)
        self.settings_camera_device.setCurrentIndex(idx)
        res_txt = f"{cfg['camera']['width']} x {cfg['camera']['height']}"
        ri = self.settings_camera_res.findText(res_txt)
        if ri < 0:
            self.settings_camera_res.addItem(res_txt)
            ri = self.settings_camera_res.findText(res_txt)
        self.settings_camera_res.setCurrentIndex(ri)

        # Projector
        self.settings_proj_w.setValue(cfg["projector"]["width"])
        self.settings_proj_h.setValue(cfg["projector"]["height"])

        # Calibration timing
        self.settings_cal_wait.setValue(cfg["calibration"]["gray_code_wait_ms"])
        self.settings_cal_delay.setValue(cfg["calibration"]["capture_delay_ms"])

        # Director
        dcfg = cfg.get("director", {})
        bi = self.director_backend.findData(dcfg.get("backend", "instinct"))
        self.director_backend.setCurrentIndex(max(0, bi))
        self.director_url.setText(dcfg.get("url", "http://127.0.0.1:11434"))
        self.director_model.setText(dcfg.get("model", "moondream"))
        self.director_interval.setValue(int(dcfg.get("redirect_interval_sec", 180)))

        # Mesh
        mcfg = cfg.get("mesh", {})
        self.mesh_enabled.setChecked(bool(mcfg.get("enabled", False)))
        self.mesh_position.setValue(int(mcfg.get("position", 0)))

        # Video wall
        vcfg = cfg.get("video", {})
        self.video_source.setText(vcfg.get("source", ""))
        self.video_brightness.setValue(
            int(float(vcfg.get("brightness", 1.0)) * 10))

        # Painting feel
        pcfg = cfg.get("painting", {})
        mood = pcfg.get("mood", "aurora")
        if mood in MOOD_NAMES:
            self.paint_mood.setCurrentIndex(MOOD_NAMES.index(mood))
        self.paint_memory.setValue(int(pcfg.get("memory_seconds", 12)))
        self.paint_sensitivity.setValue(int(pcfg.get("sensitivity", 1.0) * 10))
        self.paint_mist_toggle.setChecked(bool(pcfg.get("motion_mist", True)))
        self.paint_dreams_toggle.setChecked(bool(pcfg.get("dreams", True)))

    def _populate_display_combos(self):
        screens = QApplication.screens()
        self.settings_control_display.clear()
        self.settings_projector_display.clear()
        for i, s in enumerate(screens):
            geo = s.geometry()
            label = f"[{i}] {s.name()} ({geo.width()}x{geo.height()})"
            self.settings_control_display.addItem(label)
            self.settings_projector_display.addItem(label)
        self.settings_projector_display.addItem("Windowed (no projector display)")

    def _populate_camera_devices(self, indices):
        self.settings_camera_device.clear()
        for i in sorted(set(indices)):
            self.settings_camera_device.addItem(f"Camera {i}", i)

    # ── Settings: actions ───────────────────────────────────────

    def _identify_displays(self):
        """Flash each display's index number on it for a moment."""
        self._ident_windows = []
        for i, screen in enumerate(QApplication.screens()):
            w = QLabel(str(i))
            w.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
            w.setAlignment(Qt.AlignCenter)
            w.setStyleSheet(
                "background-color: #101024; color: #7df;"
                "font-size: 140px; font-weight: bold;"
            )
            geo = screen.geometry()
            w.resize(260, 220)
            w.move(geo.x() + (geo.width() - 260) // 2,
                   geo.y() + (geo.height() - 220) // 2)
            w.show()
            self._ident_windows.append(w)
        QTimer.singleShot(2500, self._close_ident_windows)

    def _close_ident_windows(self):
        for w in getattr(self, "_ident_windows", []):
            w.close()
        self._ident_windows = []

    def _apply_displays(self):
        screens = QApplication.screens()
        ci = self.settings_control_display.currentIndex()
        pi = self.settings_projector_display.currentIndex()

        self.config["display"]["control_display"] = ci
        self.config["display"]["projector_display"] = pi

        if 0 <= ci < len(screens):
            self.target_screen = screens[ci]
            self._position_on_screen()

        if pi >= len(screens):        # windowed fallback entry
            self.projector.set_target_screen(None)
            self.status.showMessage("Projector output windowed")
        else:
            self.projector.set_target_screen(screens[pi])
            self.status.showMessage(f"Projector output on display {pi}")

    def _detect_cameras(self):
        """Probe /dev/video indices for openable cameras."""
        self.status.showMessage("Probing cameras...")
        QApplication.processEvents()
        found = []
        current = self.config["camera"]["device_index"]
        for i in range(6):
            if self.camera is not None and i == current:
                found.append(i)     # in use by us right now — clearly valid
                continue
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                found.append(i)
            cap.release()
        if not found:
            found = [current]
            self.status.showMessage("No cameras found — keeping current setting")
        else:
            self.status.showMessage(f"Found camera(s): {found}")
        self._populate_camera_devices(found)
        idx = self.settings_camera_device.findData(current)
        if idx >= 0:
            self.settings_camera_device.setCurrentIndex(idx)

    def _apply_camera(self):
        dev = self.settings_camera_device.currentData()
        if dev is None:
            dev = self.config["camera"]["device_index"]
        try:
            w_txt, h_txt = self.settings_camera_res.currentText().split("x")
            cam_w, cam_h = int(w_txt.strip()), int(h_txt.strip())
        except ValueError:
            cam_w = self.config["camera"]["width"]
            cam_h = self.config["camera"]["height"]

        self.config["camera"]["device_index"] = int(dev)
        self.config["camera"]["width"] = cam_w
        self.config["camera"]["height"] = cam_h

        was_running = self.camera is not None
        if was_running:
            self._stop_camera()
        # Tracker dimensions depend on camera resolution
        self._recreate_engines()
        if was_running:
            self._start_camera()
        self.status.showMessage(
            f"Camera set to device {dev} at {cam_w}x{cam_h}")

    def _apply_projector(self):
        w = self.settings_proj_w.value()
        h = self.settings_proj_h.value()
        if (w == self.config["projector"]["width"]
                and h == self.config["projector"]["height"]):
            self.status.showMessage("Projector resolution unchanged")
            return
        self.projector.set_resolution(w, h)   # also updates config
        self._recreate_engines()
        self.status.showMessage(
            f"Projector set to {w}x{h} — run a new calibration scan")

    def _save_config(self):
        scenes_cfg = self.config.setdefault("scenes", {})
        scenes_cfg.setdefault("dir", "scenes")
        try:
            with open(CONFIG_PATH, "w") as f:
                json.dump(self.config, f, indent=4)
            self.status.showMessage(f"Settings saved to {CONFIG_PATH}")
        except OSError as e:
            self.status.showMessage(f"ERROR saving config: {e}")

    # ── Settings: scenes ────────────────────────────────────────

    def _refresh_scene_list(self):
        self.scene_combo.clear()
        for name in self.scenes.list_scenes():
            self.scene_combo.addItem(name)
        self._update_scene_info()

    def _update_scene_info(self):
        name = self.scene_combo.currentText()
        info = self.scenes.scene_info(name) if name else None
        if info is None:
            self.scene_info_label.setText("")
            return
        cal = "with calibration scan" if info["has_calibration"] else "no calibration scan"
        proj = info.get("projector", {})
        res = f"{proj.get('width', '?')}x{proj.get('height', '?')}"
        self.scene_info_label.setText(
            f"Saved {info['saved_at']} — projector {res}, {cal}")

    def _save_scene(self):
        name = self.scene_name_edit.text().strip() or self.scene_combo.currentText()
        if not name:
            self.status.showMessage("Give the scene a name first")
            return
        try:
            safe = self.scenes.save_scene(name, self.config)
        except (ValueError, OSError) as e:
            self.status.showMessage(f"ERROR saving scene: {e}")
            return
        self.config.setdefault("scenes", {})["current"] = safe
        self._save_config()
        self._refresh_scene_list()
        idx = self.scene_combo.findText(safe)
        if idx >= 0:
            self.scene_combo.setCurrentIndex(idx)
        self.status.showMessage(f"Scene '{safe}' saved — it will load on next startup")

    def _load_scene(self):
        name = self.scene_combo.currentText()
        if not name:
            return
        if not self.scenes.load_scene(name, self.config):
            self.status.showMessage(f"Could not load scene '{name}'")
            return
        # Apply everything the scene describes
        self.projector.set_resolution(self.config["projector"]["width"],
                                      self.config["projector"]["height"])
        self._recreate_engines()
        self._sync_ui_from_config()
        self._apply_displays()
        self.config.setdefault("scenes", {})["current"] = name
        self._save_config()
        self.status.showMessage(f"Scene '{name}' loaded")

    def _delete_scene(self):
        name = self.scene_combo.currentText()
        if not name:
            return
        answer = QMessageBox.question(
            self, "Delete scene",
            f"Delete scene '{name}'? Its calibration scan goes with it.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return
        self.scenes.delete_scene(name)
        if self.config.get("scenes", {}).get("current") == name:
            self.config["scenes"]["current"] = ""
        self._refresh_scene_list()
        self.status.showMessage(f"Scene '{name}' deleted")

    def closeEvent(self, event):
        if self.autopilot.running:
            self.autopilot.stop()
        self._stop_mesh()
        if self.calibration.is_scanning:
            self.calibration.cancel_scan()
        self.paint_engine.stop()
        self._stop_camera()
        self.projector.close()
        event.accept()
