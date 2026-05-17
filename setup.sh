#!/bin/bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv"
DESKTOP_DIR="$HOME/Desktop"

echo "============================================"
echo "  Light Painter - Setup"
echo "============================================"
echo ""

# System dependencies
echo "[1/5] Installing system packages..."
sudo apt update
sudo apt install -y \
    python3-full \
    python3-venv \
    python3-dev \
    python3-pip \
    python3-pyqt5 \
    libatlas-base-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libgl1-mesa-dev \
    libglib2.0-0 \
    v4l-utils \
    python3-picamera2 \
    python3-libcamera

# Hailo runtime
echo "[2/5] Ensuring Hailo runtime is installed..."
sudo apt install -y hailo-all || echo "WARNING: hailo-all not available. Hailo may already be installed or needs manual setup."

# Python virtual environment
echo "[3/5] Setting up Python virtual environment..."
python3 -m venv --system-site-packages "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --upgrade pip
pip install \
    opencv-python-headless \
    numpy \
    anthropic \
    Pillow

# Verify
echo "[4/5] Verifying installations..."
echo ""
echo "Python: $(python3 --version)"
echo -n "PyQt5: "
python3 -c "from PyQt5.QtWidgets import QApplication; print('OK')"
echo -n "OpenCV: "
python3 -c "import cv2; print(cv2.__version__)"
echo -n "NumPy: "
python3 -c "import numpy; print(numpy.__version__)"
echo -n "Hailo: "
python3 -c "import hailo_platform; print('OK')" 2>/dev/null || echo "NOT FOUND (install separately if needed)"
echo -n "Anthropic SDK: "
python3 -c "import anthropic; print(anthropic.__version__)"
echo -n "Picamera2: "
python3 -c "from picamera2 import Picamera2; print('OK')" 2>/dev/null || echo "NOT FOUND (Pi camera will not work; USB camera fallback still available)"

# Create double-clickable desktop shortcut
echo ""
echo "[5/5] Creating desktop shortcut..."
mkdir -p "$DESKTOP_DIR"
cat > "$DESKTOP_DIR/LightPainter.desktop" << EOF
[Desktop Entry]
Name=Light Painter
Comment=Launch Light Painter
Exec=bash -c 'cd $PROJECT_DIR && source venv/bin/activate && python main.py'
Path=$PROJECT_DIR
Type=Application
Terminal=true
Icon=applications-graphics
Categories=Graphics;
EOF
chmod +x "$DESKTOP_DIR/LightPainter.desktop"

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "A 'Light Painter' icon is now on your desktop."
echo "Double-click it to launch the app."
echo ""
echo "Press Enter to close..."
read
