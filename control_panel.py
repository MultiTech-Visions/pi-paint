from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QPushButton, QLabel, QStatusBar,
    QProgressBar, QGroupBox, QComboBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QColor
import cv2
import numpy as np
from calibration import CalibrationEngine


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

        # Calibration engine
        self.calibration = CalibrationEngine(
            config, self._grab_camera_frame, projector_win
        )
        self.calibration.progress.connect(self._on_calibration_progress)
        self.calibration.scan_finished.connect(self._on_calibration_finished)
        self.calibration.preview_ready.connect(self._on_calibration_preview)

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
        self.tabs.addTab(self._build_scene_tab(), "Scene Analysis")
        self.tabs.addTab(self._build_projection_tab(), "Projection")
        self.tabs.addTab(self._build_settings_tab(), "Settings")

        self._position_on_screen()

    def _position_on_screen(self):
        if self.target_screen is not None:
            geo = self.target_screen.geometry()
            self.move(geo.x(), geo.y())

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

    # ── Scene Analysis Tab (placeholder) ────────────────────────

    def _build_scene_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        label = QLabel("Scene analysis will go here.\n\nHailo-8L will run YOLOv8-seg\nto identify objects and surfaces.")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("color: #888; font-size: 14px;")
        layout.addWidget(label)
        return tab

    # ── Projection Tab (placeholder) ────────────────────────────

    def _build_projection_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        label = QLabel("Projection controls will go here.\n\nAnimation engine + agent brain\nwill drive the projector output.")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("color: #888; font-size: 14px;")
        layout.addWidget(label)
        return tab

    # ── Settings Tab (placeholder) ──────────────────────────────

    def _build_settings_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        label = QLabel("Settings will go here.\n\nDisplay configuration, camera selection,\nprojector resolution, API keys.")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("color: #888; font-size: 14px;")
        layout.addWidget(label)
        return tab

    def closeEvent(self, event):
        if self.calibration.is_scanning:
            self.calibration.cancel_scan()
        self._stop_camera()
        self.projector.close()
        event.accept()
