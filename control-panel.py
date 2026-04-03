from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QPushButton, QLabel, QStatusBar
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import cv2


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
        btn_black.clicked.connect(lambda: self.projector.show_solid(
            __import__('PyQt5').QtGui.QColor(0, 0, 0)
        ))
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

    # ── Calibration Tab (placeholder) ───────────────────────────

    def _build_calibration_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        label = QLabel("Calibration controls will go here.\n\nThis will run structured light scanning\nto map camera ↔ projector coordinates.")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("color: #888; font-size: 14px;")
        layout.addWidget(label)
        return tab

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
        self._stop_camera()
        self.projector.close()
        event.accept()
