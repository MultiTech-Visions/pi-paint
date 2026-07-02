"""The scene director — looks at the physical scene, decides the show.

Set the box down, and this is the mind that takes over:

1. analyze_surfaces(): the camera's view is warped into projector
   space (so we reason about exactly what the projector can touch)
   and segmented into a handful of surfaces — each with a footprint
   mask, centroid, brightness, color, and texture.

2. A director turns that into a *program*: a mood, a tempo, a theme,
   and a cast of performers assigned to surfaces.

Two directors are provided:

* InstinctDirector — built-in, pure CV rules, always available,
  fully offline.  Dark expanses become aquariums, textured shapes
  get their outlines traced, bright walls host auroras.

* VLMDirector — sends the scene snapshot to a local vision-language
  model (Ollama or any OpenAI-compatible server, e.g. llama.cpp) and
  asks it to direct.  Falls back to instinct on any failure, so
  autonomy never blocks on a model.

Pure Python + numpy/cv2 (urllib for the VLM) — no Qt, testable headless.
"""

import base64
import json
import urllib.request

import numpy as np
import cv2

from paint_canvas import MOOD_NAMES
from performers import BEHAVIOR_NAMES, PERSPECTIVES


# ── Scene analysis ──────────────────────────────────────────────────────

def analyze_surfaces(frame_bgr, proj_w, proj_h, inv_x=None, inv_y=None,
                     max_regions=4):
    """Segment the projectable scene into surfaces.

    Returns (surfaces, view_rgb):
      surfaces: list of dicts sorted largest-first:
        {id, area_frac, centroid (x,y), bbox (x0,y0,x1,y1), brightness
         0..1, edge_density 0..1, color_rgb, mask (bool, quarter res),
         mask_scale}
      view_rgb: the camera's view warped into projector space (RGB u8),
        i.e. what the projector's canvas is physically lying on.
    """
    if inv_x is not None:
        view = cv2.remap(frame_bgr, inv_x, inv_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
        view = cv2.resize(view, (proj_w, proj_h))
    else:
        view = cv2.resize(frame_bgr, (proj_w, proj_h))
    view_rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)

    sw, sh = proj_w // 4, proj_h // 4
    small = cv2.resize(view, (sw, sh), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 1.5)

    # Features: color (Lab) + position, so regions are coherent patches
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2Lab).astype(np.float32)
    ys, xs = np.mgrid[0:sh, 0:sw].astype(np.float32)
    feats = np.stack([
        lab[:, :, 0],
        lab[:, :, 1] * 1.6,
        lab[:, :, 2] * 1.6,
        xs / sw * 90.0,
        ys / sh * 90.0,
    ], axis=-1).reshape(-1, 5)

    k = max(2, min(max_regions, 4))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
    _, labels, _ = cv2.kmeans(feats, k, None, criteria, 3,
                              cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(sh, sw)

    gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_small, 60, 140)

    surfaces = []
    for i in range(k):
        mask = labels == i
        area_frac = float(mask.mean())
        if area_frac < 0.03:
            continue
        # Keep only the largest connected patch so a surface is a place,
        # not a scattering of same-colored fragments
        n, comp = cv2.connectedComponents(mask.astype(np.uint8))
        if n > 2:
            sizes = np.bincount(comp.ravel())
            sizes[0] = 0
            mask = comp == int(sizes.argmax())
            area_frac = float(mask.mean())
            if area_frac < 0.03:
                continue
        mys, mxs = np.nonzero(mask)
        cx, cy = float(mxs.mean()) * 4, float(mys.mean()) * 4
        bbox = (int(mxs.min()) * 4, int(mys.min()) * 4,
                int(mxs.max()) * 4, int(mys.max()) * 4)
        mean_bgr = small[mask].mean(axis=0)
        surfaces.append({
            "id": len(surfaces),
            "area_frac": area_frac,
            "centroid": (cx, cy),
            "bbox": bbox,
            "brightness": float(gray_small[mask].mean()) / 255.0,
            "edge_density": float((edges[mask] > 0).mean()),
            "color_rgb": [int(mean_bgr[2]), int(mean_bgr[1]), int(mean_bgr[0])],
            "mask": mask,
            "mask_scale": 4,
        })

    surfaces.sort(key=lambda s: -s["area_frac"])
    for new_id, s in enumerate(surfaces):
        s["id"] = new_id
    return surfaces, view_rgb


def surfaces_summary(surfaces):
    """Compact JSON-safe description of surfaces (for prompts and UI)."""
    return [{
        "id": s["id"],
        "area_pct": round(s["area_frac"] * 100, 1),
        "center": [round(s["centroid"][0]), round(s["centroid"][1])],
        "brightness": round(s["brightness"], 2),
        "texture": round(s["edge_density"], 2),
        "color_rgb": s["color_rgb"],
    } for s in surfaces]


def draw_surface_overlay(view_rgb, surfaces):
    """The director's-eye view: surfaces outlined and numbered."""
    out = view_rgb.copy()
    colors = [(120, 220, 255), (255, 170, 120), (170, 255, 140), (240, 140, 255)]
    for s in surfaces:
        c = colors[s["id"] % len(colors)]
        m = (s["mask"].astype(np.uint8)) * 255
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cont in contours:
            cv2.drawContours(out, [cont * s["mask_scale"]], -1, c, 2)
        cx, cy = int(s["centroid"][0]), int(s["centroid"][1])
        cv2.putText(out, str(s["id"]), (cx - 8, cy + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, c, 2, cv2.LINE_AA)
    return out


# ── Directors ───────────────────────────────────────────────────────────

class InstinctDirector:
    """Built-in direction from CV heuristics — no model required."""

    source = "instinct"

    def direct(self, surfaces, view_rgb):
        gray = cv2.cvtColor(view_rgb, cv2.COLOR_RGB2GRAY)
        scene_brightness = float(gray.mean()) / 255.0
        warmth = 0.0
        if surfaces:
            r, g, b = np.mean([s["color_rgb"] for s in surfaces], axis=0)
            warmth = (r - b) / 255.0

        if scene_brightness < 0.18:
            mood = "biolume"
        elif scene_brightness < 0.32:
            mood = "moonlight"
        elif warmth > 0.12:
            mood = "embers"
        else:
            mood = "aurora"

        tempo = float(np.clip(0.3 + gray.std() / 128.0, 0.2, 0.85))

        behaviors = []
        if surfaces:
            big = surfaces[0]
            behaviors.append({"type": "aurora_drift", "region": big["id"]})
            darkest = min(surfaces, key=lambda s: s["brightness"])
            if darkest["brightness"] < 0.35 and darkest["area_frac"] > 0.08:
                behaviors.append({"type": "fish_tank", "region": darkest["id"]})
            textured = max(surfaces, key=lambda s: s["edge_density"])
            if textured["edge_density"] > 0.06:
                behaviors.append({"type": "contour_trace", "region": textured["id"]})
            second = surfaces[1] if len(surfaces) > 1 else big
            behaviors.append({"type": "fireflies", "region": second["id"]})
            # A compact, clean patch earns the 3D box illusion; otherwise
            # the brightest small surface breathes instead
            boxy = [s for s in surfaces
                    if 0.04 < s["area_frac"] < 0.30 and s["edge_density"] < 0.05]
            if boxy:
                pick = min(boxy, key=lambda s: s["edge_density"])
                behaviors.append({"type": "perspective_box",
                                  "region": pick["id"], "perspective": "auto"})
            else:
                small_bright = max(surfaces, key=lambda s: s["brightness"])
                if small_bright["area_frac"] < 0.25:
                    behaviors.append({"type": "breathing_glow",
                                      "region": small_bright["id"]})
        else:
            behaviors = [{"type": "aurora_drift", "region": None},
                         {"type": "fireflies", "region": None}]

        theme = (f"a {'dim' if scene_brightness < 0.3 else 'lit'} scene of "
                 f"{len(surfaces)} surfaces — {mood} tonight")
        return {
            "theme": theme,
            "mood": mood,
            "tempo": tempo,
            "behaviors": behaviors[:5],
            "source": self.source,
            "notes": "",
        }


PROMPT = """You are the director of a projection-mapped light show. A projector \
paints living light onto the physical scene in the attached photo (the photo is \
already aligned to the projector's canvas; coordinates are projector pixels).

Detected surfaces (id, area %, center, brightness 0-1, texture 0-1, color):
{surfaces}

Choose a show. Available behaviors: {behaviors}.
Available moods: {moods}.

Respond with ONLY a JSON object:
{{"theme": "<one poetic sentence naming what you see and the feeling you chose>",
 "mood": "<one mood>",
 "tempo": <0.0-1.0>,
 "behaviors": [{{"type": "<behavior>", "region": <surface id>}}, ...]}}

Pick 2-4 behaviors. Match them to what the surfaces are: dark expanses suit \
fish_tank, distinct objects suit contour_trace, big open walls suit \
aurora_drift, warm corners suit breathing_glow or fireflies, and a clean \
patch suits perspective_box (a breathing 3D box illusion). A perspective_box \
behavior may add "perspective": "left"|"right"|"up"|"down"|"center" — the \
direction its depth recedes; pick what matches the angle that surface is \
seen from (omit for auto, which recedes toward the canvas center)."""


class VLMDirector:
    """Direction by a local vision-language model (Ollama / OpenAI-style).

    Any failure — server down, bad JSON, hallucinated fields — falls
    back to InstinctDirector so the show always goes on.
    """

    def __init__(self, backend="ollama", url="http://127.0.0.1:11434",
                 model="moondream", timeout=45):
        self.backend = backend
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.fallback = InstinctDirector()
        self.source = f"vlm:{model}"

    def direct(self, surfaces, view_rgb):
        try:
            program = self._ask_model(surfaces, view_rgb)
            return self._validate(program, surfaces)
        except Exception as e:      # noqa: BLE001 — any failure means fallback
            program = self.fallback.direct(surfaces, view_rgb)
            program["notes"] = f"VLM unavailable ({type(e).__name__}: {e}) — instinct took over"
            return program

    # ── Model I/O ──

    def _snapshot_b64(self, view_rgb):
        h, w = view_rgb.shape[:2]
        scale = 512.0 / max(w, 1)
        img = cv2.resize(view_rgb, (512, max(1, int(h * scale))))
        ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return base64.b64encode(jpg.tobytes()).decode("ascii")

    def _prompt(self, surfaces):
        return PROMPT.format(
            surfaces=json.dumps(surfaces_summary(surfaces)),
            behaviors=", ".join(BEHAVIOR_NAMES),
            moods=", ".join(MOOD_NAMES),
        )

    def _post_json(self, path, payload):
        req = urllib.request.Request(
            self.url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode())

    def _ask_model(self, surfaces, view_rgb):
        b64 = self._snapshot_b64(view_rgb)
        prompt = self._prompt(surfaces)
        if self.backend == "ollama":
            data = self._post_json("/api/generate", {
                "model": self.model,
                "prompt": prompt,
                "images": [b64],
                "stream": False,
                "format": "json",
            })
            text = data.get("response", "")
        else:                       # openai-compatible (llama.cpp, vLLM, ...)
            data = self._post_json("/v1/chat/completions", {
                "model": self.model,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}"}},
                ]}],
                "max_tokens": 400,
            })
            text = data["choices"][0]["message"]["content"]
        return self._extract_json(text)

    @staticmethod
    def _extract_json(text):
        """Pull the first balanced JSON object out of model output."""
        start = text.find("{")
        if start < 0:
            raise ValueError("no JSON in model output")
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start:i + 1])
        raise ValueError("unbalanced JSON in model output")

    # ── Validation: models dream; the show must be real ──

    def _validate(self, program, surfaces):
        valid_ids = {s["id"] for s in surfaces}
        mood = program.get("mood", "")
        if mood not in MOOD_NAMES:
            mood = "aurora"
        behaviors = []
        for b in program.get("behaviors", []):
            if not isinstance(b, dict) or b.get("type") not in BEHAVIOR_NAMES:
                continue
            region = b.get("region")
            if region not in valid_ids:
                region = surfaces[0]["id"] if surfaces else None
            entry = {"type": b["type"], "region": region}
            if b.get("perspective") in PERSPECTIVES:
                entry["perspective"] = b["perspective"]
            behaviors.append(entry)
        if not behaviors:
            raise ValueError("model chose no valid behaviors")
        try:
            tempo = float(np.clip(float(program.get("tempo", 0.5)), 0.0, 1.0))
        except (TypeError, ValueError):
            tempo = 0.5
        return {
            "theme": str(program.get("theme", ""))[:200] or "an untitled night",
            "mood": mood,
            "tempo": tempo,
            "behaviors": behaviors[:5],
            "source": self.source,
            "notes": "",
        }


def make_director(config):
    """Build the director described by config['director']."""
    dcfg = config.get("director", {})
    backend = dcfg.get("backend", "instinct")
    if backend in ("ollama", "openai"):
        return VLMDirector(
            backend=backend,
            url=dcfg.get("url", "http://127.0.0.1:11434"),
            model=dcfg.get("model", "moondream"),
            timeout=int(dcfg.get("timeout_sec", 45)),
        )
    return InstinctDirector()
