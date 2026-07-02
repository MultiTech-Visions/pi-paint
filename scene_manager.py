"""Scene profiles — the whole making of a scene, saved and restored.

A *scene* is one physical setup: where the projector points, which
camera watches it, the structured light calibration for that exact
geometry, and the feel you dialed in (mood, memory, sensitivity...).

Setting up at a venue is real work — aim the projector, run the scan,
tune the feel.  Scenes make that work durable: save it under a name,
and the next time the box powers on it's all back, ready to paint.

Each scene lives in its own directory:

    scenes/<name>/
        scene.json              # config sections for this setup
        calibration_data.npz    # the structured light scan (if done)

Pure Python (no Qt) so it's testable headless.
"""

import json
import os
import re
import shutil
import time


# Config sections that belong to a scene (everything that describes
# the physical setup + the experience settings)
SCENE_SECTIONS = ["display", "projector", "camera", "calibration", "painting"]


class SceneManager:
    def __init__(self, base_dir="scenes", calibration_file="calibration_data.npz"):
        self.base_dir = base_dir
        self.calibration_file = calibration_file

    # ── Naming / paths ──────────────────────────────────────────────────

    @staticmethod
    def sanitize(name):
        """Turn a scene name into a safe directory name."""
        name = name.strip().replace(" ", "_")
        name = re.sub(r"[^A-Za-z0-9_\-]", "", name)
        return name[:64]

    def scene_dir(self, name):
        return os.path.join(self.base_dir, self.sanitize(name))

    def _scene_json(self, name):
        return os.path.join(self.scene_dir(name), "scene.json")

    def _scene_calibration(self, name):
        return os.path.join(self.scene_dir(name), self.calibration_file)

    # ── Queries ─────────────────────────────────────────────────────────

    def list_scenes(self):
        """Names of saved scenes, most recently saved first."""
        if not os.path.isdir(self.base_dir):
            return []
        scenes = []
        for entry in os.listdir(self.base_dir):
            path = os.path.join(self.base_dir, entry, "scene.json")
            if os.path.isfile(path):
                scenes.append((os.path.getmtime(path), entry))
        return [name for _, name in sorted(scenes, reverse=True)]

    def exists(self, name):
        return os.path.isfile(self._scene_json(name))

    def scene_info(self, name):
        """Metadata for display: saved time, calibration presence."""
        if not self.exists(name):
            return None
        try:
            with open(self._scene_json(name), "r") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        return {
            "name": name,
            "saved_at": data.get("_meta", {}).get("saved_at", "unknown"),
            "has_calibration": os.path.isfile(self._scene_calibration(name)),
            "projector": data.get("projector", {}),
        }

    # ── Save / load / delete ────────────────────────────────────────────

    def save_scene(self, name, config):
        """Snapshot the scene sections of config + the current calibration.

        Returns the sanitized scene name.
        """
        safe = self.sanitize(name)
        if not safe:
            raise ValueError("Scene name is empty after sanitizing")
        os.makedirs(self.scene_dir(safe), exist_ok=True)

        snapshot = {k: config[k] for k in SCENE_SECTIONS if k in config}
        snapshot["_meta"] = {"saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(self._scene_json(safe), "w") as f:
            json.dump(snapshot, f, indent=4)

        # Bundle the calibration scan with the scene it belongs to
        if os.path.isfile(self.calibration_file):
            shutil.copy2(self.calibration_file, self._scene_calibration(safe))
        return safe

    def load_scene(self, name, config):
        """Merge a saved scene into config and restore its calibration.

        Mutates config in place.  Returns True on success.
        """
        if not self.exists(name):
            return False
        try:
            with open(self._scene_json(name), "r") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return False

        for section in SCENE_SECTIONS:
            if section in data:
                config.setdefault(section, {}).update(data[section])

        # Restore this scene's calibration as the working scan
        cal = self._scene_calibration(name)
        if os.path.isfile(cal):
            shutil.copy2(cal, self.calibration_file)
        return True

    def delete_scene(self, name):
        """Remove a saved scene from disk."""
        d = self.scene_dir(name)
        if os.path.isdir(d) and self.exists(name):
            shutil.rmtree(d)
            return True
        return False
