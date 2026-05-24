#!/bin/bash
# pi-paint VJ — windowed launcher.
# Double-click in the file manager and choose "Execute".
# Runs the VJ app in a small 854x480 window. Good for testing on the Pi
# itself without a projector connected.

cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
  zenity --error --text="Setup hasn't been run yet.\n\nDouble-click setup.sh first." 2>/dev/null \
    || (echo "ERROR: venv not found — please run setup.sh first." && read -p "Press Enter to close...")
  exit 1
fi

./venv/bin/python main.py
