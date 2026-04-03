from PyQt5.QtWidgets import QMainWindow, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QColor, QPainter
import numpy as np


class ProjectorWindow(QMainWindow):
    def __init__(self, config, screen=None):
        super().__init__()
        self.config = config
        self.proj_w = config["projector"]["width"]
        self.proj_h = config["projector"]["height"]
        self.target_screen = screen

        self.setWindowTitle("Projector Output")
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.setCursor(Qt.BlankCursor)

        self.canvas = QLabel(self)
        self.canvas.setAlignment(Qt.AlignCenter)
        self.canvas.setStyleSheet("background-color: black;")
        self.setCentralWidget(self.canvas)

        self._position_on_screen()
        self.show_solid(QColor(0, 0, 0))

    def _position_on_screen(self):
        if self.target_screen is not None:
            geo = self.target_screen.geometry()
            self.setGeometry(geo)
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
            self.showFullScreen()
        else:
            # Windowed fallback for single-display testing
            self.setFixedSize(self.proj_w, self.proj_h)
            self.setWindowFlags(Qt.Window)

    def show_solid(self, color):
        """Fill the projector with a solid color."""
        img = QImage(self.proj_w, self.proj_h, QImage.Format_RGB888)
        img.fill(color)
        self.canvas.setPixmap(QPixmap.fromImage(img))

    def show_image(self, np_array):
        """Display a numpy array (H, W, 3) uint8 RGB on the projector."""
        h, w, ch = np_array.shape
        bytes_per_line = ch * w
        img = QImage(np_array.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.canvas.setPixmap(QPixmap.fromImage(img).scaled(
            self.proj_w, self.proj_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
        ))

    def show_pattern(self, pattern_array):
        """Display a calibration pattern. Input: numpy (H, W) or (H, W, 3) uint8."""
        if pattern_array.ndim == 2:
            rgb = np.stack([pattern_array] * 3, axis=-1)
        else:
            rgb = pattern_array
        self.show_image(rgb)

    def show_test_pattern(self):
        """Display a color gradient test pattern to verify projector output."""
        img = np.zeros((self.proj_h, self.proj_w, 3), dtype=np.uint8)
        # Red gradient left to right
        img[:, :, 0] = np.tile(np.linspace(0, 255, self.proj_w, dtype=np.uint8), (self.proj_h, 1))
        # Green gradient top to bottom
        img[:, :, 1] = np.tile(np.linspace(0, 255, self.proj_h, dtype=np.uint8).reshape(-1, 1), (1, self.proj_w))
        # Blue diagonal
        for y in range(self.proj_h):
            for x in range(self.proj_w):
                img[y, x, 2] = int(255 * ((x + y) / (self.proj_w + self.proj_h)))
        self.show_image(img)
