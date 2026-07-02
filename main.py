import sys
import json
from PyQt5.QtWidgets import QApplication
from control_panel import ControlPanel
from projector_window import ProjectorWindow
from scene_manager import SceneManager


def load_config(path="config.json"):
    with open(path, "r") as f:
        return json.load(f)


def apply_startup_scene(config):
    """Restore the last-used scene so the box powers on ready to paint."""
    scenes_cfg = config.get("scenes", {})
    current = scenes_cfg.get("current") or ""
    if not current:
        return
    manager = SceneManager(scenes_cfg.get("dir", "scenes"))
    if manager.load_scene(current, config):
        print(f"Restored scene '{current}'")
    else:
        print(f"WARNING: saved scene '{current}' not found — using config.json as-is")


def main():
    config = load_config()
    apply_startup_scene(config)
    app = QApplication(sys.argv)

    screens = app.screens()
    print(f"Detected {len(screens)} display(s):")
    for i, screen in enumerate(screens):
        geo = screen.geometry()
        print(f"  [{i}] {screen.name()} - {geo.width()}x{geo.height()} at ({geo.x()},{geo.y()})")

    control_idx = config["display"]["control_display"]
    projector_idx = config["display"]["projector_display"]

    if projector_idx >= len(screens):
        print(f"WARNING: projector_display index {projector_idx} not found. Only {len(screens)} display(s) detected.")
        print("Running projector window on primary display in windowed mode.")
        projector_screen = None
    else:
        projector_screen = screens[projector_idx]

    if control_idx >= len(screens):
        print(f"WARNING: control_display index {control_idx} not found. Using primary.")
        control_screen = screens[0]
    else:
        control_screen = screens[control_idx]

    # Launch projector output window
    projector_win = ProjectorWindow(config, projector_screen)
    projector_win.show()

    # Launch control panel
    control_panel = ControlPanel(config, control_screen, projector_win)
    control_panel.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
