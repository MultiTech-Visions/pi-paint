#!/bin/bash
# fix-taskbar-dual-display.sh
# Fixes the disappearing taskbar when connecting a projector (dual display)
# on Raspberry Pi OS Bookworm (Wayfire/Wayland).
#
# The wf-panel-pi panel only appears on one output by default.
# This script configures it to appear on the correct display and
# restarts the panel.

set -e

PANEL_CFG="$HOME/.config/wf-panel-pi.ini"
WAYFIRE_CFG="$HOME/.config/wayfire.ini"

# --- Detect desktop environment ---
if [ "$XDG_SESSION_TYPE" = "wayland" ] || pgrep -x wayfire >/dev/null 2>&1; then
    echo "Detected: Wayfire (Wayland) — Pi OS Bookworm"

    # Ensure wf-panel-pi.ini exists
    if [ ! -f "$PANEL_CFG" ]; then
        echo "Panel config not found at $PANEL_CFG"
        echo "Copying default config..."
        cp /etc/xdg/wf-panel-pi.ini "$PANEL_CFG" 2>/dev/null || true
    fi

    if [ -f "$PANEL_CFG" ]; then
        # Check if an output key already exists; if so, update it, otherwise add it
        if grep -q "^output=" "$PANEL_CFG"; then
            # Remove the fixed output assignment so the panel appears on all outputs
            sed -i 's/^output=.*/#output=/' "$PANEL_CFG"
            echo "Commented out fixed output assignment in wf-panel-pi.ini"
        fi

        # Ensure the panel is set to appear (autohide off)
        if grep -q "^autohide=" "$PANEL_CFG"; then
            sed -i 's/^autohide=.*/autohide=0/' "$PANEL_CFG"
        fi
    fi

    # Restart the panel
    echo "Restarting wf-panel-pi..."
    killall wf-panel-pi 2>/dev/null || true
    sleep 1
    nohup wf-panel-pi >/dev/null 2>&1 &
    echo "Panel restarted."

    # Also ensure wayfire.ini has mirror/extend configured properly
    if [ -f "$WAYFIRE_CFG" ]; then
        echo ""
        echo "Tip: If the panel shows on the projector instead of your main screen,"
        echo "edit $WAYFIRE_CFG and set the output section:"
        echo ""
        echo "  [output:HDMI-A-1]"
        echo "  mode = auto"
        echo "  position = 0,0"
        echo ""
        echo "  [output:HDMI-A-2]"
        echo "  mode = 854x480"
        echo "  position = auto"
        echo ""
        echo "Then restart wayfire or log out and log back in."
    fi

elif pgrep -x lxpanel >/dev/null 2>&1; then
    echo "Detected: LXDE (X11) — Pi OS Bullseye or Legacy"

    LXPANEL_DIR="$HOME/.config/lxpanel/LXDE-pi/panels"

    if [ -d "$LXPANEL_DIR" ]; then
        # Reset panel to primary monitor
        echo "Restarting lxpanel..."
        lxpanelctl restart 2>/dev/null || true
        echo "Panel restarted."
    else
        echo "lxpanel config directory not found. Trying restart anyway..."
        lxpanelctl restart 2>/dev/null || true
    fi

    echo ""
    echo "If the panel is still missing, try:"
    echo "  1. Right-click desktop → Desktop Preferences → check panel settings"
    echo "  2. Or run: lxpanelctl run"

else
    echo "Could not detect desktop environment."
    echo ""
    echo "Quick fixes to try:"
    echo "  - If Wayfire:  killall wf-panel-pi; wf-panel-pi &"
    echo "  - If LXDE:     lxpanelctl restart"
    echo "  - If labwc:    killall sfwbar; sfwbar &"
fi

echo ""
echo "Done. If the taskbar still doesn't appear, try logging out and back in."
