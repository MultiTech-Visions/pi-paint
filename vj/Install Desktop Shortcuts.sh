#!/bin/bash
# Installs clickable launcher icons on the Pi's desktop.
# Double-click in the file manager and choose "Execute in Terminal".

set -e
cd "$(dirname "$0")"
VJ_DIR="$(pwd)"
DESKTOP_DIR="${XDG_DESKTOP_DIR:-$HOME/Desktop}"

echo ""
echo "Installing VJ launcher icons to: $DESKTOP_DIR"
echo ""

mkdir -p "$DESKTOP_DIR"

write_launcher() {
  local name="$1"
  local script="$2"
  local comment="$3"
  local file="$DESKTOP_DIR/$name.desktop"
  cat > "$file" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=$name
Comment=$comment
Exec=bash "$VJ_DIR/$script"
Icon=video-display
Path=$VJ_DIR
Terminal=false
Categories=AudioVideo;
StartupNotify=true
EOF
  chmod +x "$file"
  # Mark as trusted (Raspberry Pi OS / GNOME Files)
  gio set "$file" metadata::trusted true 2>/dev/null || true
  echo "  ✓ $name"
}

write_launcher "VJ — Windowed"    "Run Windowed.sh"    "pi-paint VJ in a window"
write_launcher "VJ — Fullscreen"  "Run Fullscreen.sh"  "pi-paint VJ fullscreen on primary display"
write_launcher "VJ — Projector"   "Run Projector.sh"   "pi-paint VJ fullscreen on display 1 (projector)"

echo ""
echo "Done. Look for the icons on your desktop."
echo "If they show a '?' icon, right-click and choose 'Allow Launching'."
echo ""
read -p "Press Enter to close..."
