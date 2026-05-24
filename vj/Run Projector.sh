#!/bin/bash
# pi-paint VJ — fullscreen launcher (projector on display 1).
# Double-click in the file manager and choose "Execute".
#
# Assumes your projector is the SECOND display (display index 1).
# If your projector is the only display, use "Run Fullscreen.sh" instead.

cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
  zenity --error --text="Setup hasn't been run yet.\n\nDouble-click setup.sh first." 2>/dev/null \
    || (echo "ERROR: venv not found — please run setup.sh first." && read -p "Press Enter to close...")
  exit 1
fi

./venv/bin/python main.py --fullscreen --display 1
