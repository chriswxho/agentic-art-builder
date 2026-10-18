#!/usr/bin/env python3
"""test_3: one shape per call, parallel region workers.

Each round, one worker per canvas region (3x2 grid, 6 regions) runs at the same time. Every worker
sees the full canvas and reference with its region outlined, zoomed crops of its region, measured
color errors for its sub-cells, and a memory of its own last shapes. It returns an AgentProposal
that adds exactly ONE shape whose center lies in its region. The prompt stresses that this is one
small step of a long process. The orchestrator validates each proposal against the round's base
context, merges the round's shapes (largest first) into one add_objects revision, renders and
records it, and the next round sees the new canvas.

Mark rules match test_2 (local override, shared policy untouched): CMYK inks on white, opacity
0.1-1.0, marks may span the canvas, strokes up to 48 wide.

Outputs: test_3/runs/<run_id>/ (steps, handoffs incl. per-worker inference logs, views/ images sent
to the model, comparison.png, progress.png, milestones.png, evals.txt, handoffs/evaluation.json).

usage (from the repo root):
    drawing-sandbox/.venv/bin/python test_3/run_test_3.py --level easy [--only RUN_ID] [--rounds 40]
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait
import contextlib
from copy import deepcopy
import http.client
import io
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SANDBOX = ROOT / "drawing-sandbox"
sys.path[:0] = [str(SANDBOX), str(SANDBOX / "scripts")]

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from drawing_sandbox import iteration_context, nemotron_inference, policy, recording  # noqa: E402
from drawing_sandbox.iteration_context import (  # noqa: E402
    PictureContextError, apply_and_record, apply_proposal, create_context, initialize_recorded_context,
    read_context,
)
from drawing_sandbox.nemotron_inference import (  # noqa: E402
    NemotronInferenceError, _call_crusoe, _data_url, _response_text, infer_evaluation,
    load_crusoe_api_key, parse_proposal, write_evaluation_artifacts,
)
from drawing_sandbox.iteration_context import EVALUATOR_MODEL, PROTOCOL_VERSION  # noqa: E402
from nemotron_iteration import reference_record  # noqa: E402

# --------------------------------------------------------------------------- settings
ROUNDS = 40
CANVAS = (768, 512)
REGION_GRID = (3, 2)            # 3 across, 2 down -> 6 parallel workers of 256x256
MAX_SPAN = 820
MAX_STROKE = 48
OPACITY_RANGE = (0.1, 1.0)
CALL_TIMEOUT = 180              # seconds per HTTP call
ROUND_DEADLINE = 600            # seconds to wait for a whole round
MAX_ATTEMPTS = 3
MAX_TOKENS = 16384
WORKER_MODEL = EVALUATOR_MODEL  # Super-VL paints too; the Nano generator is not used in test_3
MEMORY = 6                      # how many of its own past shapes each worker is reminded of
MAX_REGIONS = 8
MAX_PAINTERS = 8                # parallel calls per round
REPLAN_EVERY = 10               # rounds between Super-VL region re-plans
INKS = {"cyan": (0, 1, 1), "magenta": (1, 0, 1), "yellow": (1, 1, 0), "black": (0, 0, 0)}
INK_NAMES = {"#00ffff": "cyan", "#ff00ff": "magenta", "#ffff00": "yellow", "#000000": "black"}
# Set from the command line: finish_layers tracks unfinished two-layer colors (test_3b);
# skills is text from a skills file appended to every painter's instructions.
OPTIONS = {"finish_layers": False, "skills": ""}
TONE_BLOCK = 8                  # same block size the offline evals use for skill_vs_blank

TARGETS = {
    "easy": [
        ("easy-sea-sunset", "landscape-paintings/01-sea-sunset.png"),
        ("easy-snowy-mountains", "landscape-paintings/02-snowy-mountains.png"),
        ("easy-green-hills", "landscape-paintings/03-green-hills.png"),
    ],
    "medium": [
        ("medium-great-wave", "medphotos/Katsushika_Hokusai_-_Thirty-Six_Views_of_Mount_Fuji-_The_Great_Wave_Off_the_Coast_of_Kanagawa_-_Google_Art_Project.jpg"),
        ("medium-starry-night", "medphotos/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"),
        ("medium-hay-wain", "medphotos/John_Constable_The_Hay_Wain.jpg"),
    ],
    "hard": [
        ("hard-grand-canyon", "hardphotos/01-grand-canyon.jpg"),
        ("hard-matterhorn", "hardphotos/04-matterhorn.jpg"),
        ("hard-moraine-lake", "hardphotos/08-moraine-lake.jpg"),
    ],
}
PROMPT = ("Recreate the reference image one shape at a time, with several painters working on "
          "different regions in parallel, using only cyan, magenta, yellow and black on white.")

# --------------------------------------------------------------------------- local rule override (same as test_2)
policy.CMYK_MARK_CONSTRAINTS.pop("required_opacity", None)
policy.CMYK_MARK_CONSTRAINTS["max_shape_span"] = MAX_SPAN
policy.CMYK_MARK_CONSTRAINTS["max_stroke_width"] = MAX_STROKE
_stock_instruction_set = policy.current_instruction_set


def _test3_instruction_set() -> dict:
    rules = _stock_instruction_set()
    marks = rules["marks"]
    marks.pop("opacity", None)
    marks["opacity_range"] = list(OPACITY_RANGE)
    marks["max_transformed_span"] = MAX_SPAN
    marks["max_transformed_stroke_width"] = MAX_STROKE
    marks["shapes_per_agent_call"] = 1
    rules["variant"] = "test_3 local override"
    return rules


for _module in (policy, iteration_context, recording):
    _module.current_instruction_set = _test3_instruction_set

# --------------------------------------------------------------------------- regions


def grid_regions() -> list[dict]:
    """Fallback when the planner fails: a plain 3x2 grid, one painter each."""
    cols, rows = REGION_GRID
    w, h = CANVAS[0] // cols, CANVAS[1] // rows
    names_x = ["left", "center", "right"] if cols == 3 else [f"col{i + 1}" for i in range(cols)]
    names_y = ["top", "bottom"] if rows == 2 else [f"row{j + 1}" for j in range(rows)]
    return [{"region_id": f"{names_y[j]}-{names_x[i]}", "label": f"{names_y[j]} {names_x[i]} of the canvas",
             "x0": i * w, "y0": j * h, "x1": (i + 1) * w, "y1": (j + 1) * h, "painters": 1}
            for j in range(rows) for i in range(cols)]


PLANNER_PROMPT = f"""You plan the work of a team of painters who recreate the reference picture on a
{CANVAS[0]}x{CANVAS[1]} canvas (x to the right, y down, origin top-left), one shape per painter per round.
Split the picture into 4-{MAX_REGIONS} FOCUS REGIONS that follow its content (for example: the sky, the sun,
distant mountains, a lake, a foreground tree), NOT a uniform grid. Each region is an axis-aligned box.
Regions may overlap a little and together should cover the whole canvas. Give a region 2 painters if it
holds the most important or most detailed content, otherwise 1; use at most {MAX_PAINTERS} painters in total.
When a current canvas is shown, put the regions and extra painters where the canvas differs most from
the reference. Return exactly one JSON object and nothing else:
{{"regions": [{{"region_id": "short-slug", "label": "what this region contains", "x0": 0, "y0": 0,
"x1": 768, "y1": 200, "painters": 1, "focus": "what painters here should get right (shapes, colors)"}}]}}"""


def _valid_regions(raw: object) -> list[dict]:
    if not isinstance(raw, dict) or not isinstance(raw.get("regions"), list):
        raise ValueError("planner must return {\"regions\": [...]}")
    out, seen, painters = [], set(), 0
    for item in raw["regions"][:MAX_REGIONS]:
        try:
            x0, y0, x1, y1 = (int(round(float(item[k]))) for k in ("x0", "y0", "x1", "y1"))
        except (KeyError, TypeError, ValueError):
            continue
        x0, x1 = max(0, min(x0, x1)), min(CANVAS[0], max(x0, x1))
        y0, y1 = max(0, min(y0, y1)), min(CANVAS[1], max(y0, y1))
        if x1 - x0 < 32 or y1 - y0 < 32:
            continue
        slug = "".join(c if c.isalnum() else "-" for c in str(item.get("region_id") or item.get("label") or "region").lower())[:24].strip("-") or "region"
        while slug in seen:
            slug += "x"
        seen.add(slug)
        count = 2 if item.get("painters") == 2 and painters + 2 <= MAX_PAINTERS else 1
        if painters + count > MAX_PAINTERS:
            break
        painters += count
        out.append({"region_id": slug, "label": str(item.get("label", slug))[:120],
                    "focus": str(item.get("focus", ""))[:300], "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                    "painters": count})
    if len(out) < 3:
        raise ValueError("planner returned fewer than 3 usable regions")
    return out


def _find_regions_json(text: str) -> dict | None:
    """Return the last JSON object in the text that has a "regions" list."""
    decoder, found = json.JSONDecoder(), None
    for index, char in enumerate(text or ""):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("regions"), list):
            found = value
    return found


def plan_regions(run_dir: Path, round_index: int, reference: Path, render_path: Path | None,
                 error_text: str, api_key: str, log) -> list[dict]:
    """Ask Super-VL for content-based focus regions; fall back to the grid if it fails."""
    if render_path is None:
        image = run_dir / "views" / "plan-reference.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.open(reference).convert("RGB").resize(CANVAS).save(image)
        text = "The image is the reference picture."
    else:
        image = labelled_sheet([("REFERENCE", reference), ("CURRENT CANVAS", render_path)],
                               run_dir / "views" / f"plan-r{round_index:03d}.png", columns=2)
        text = ("The image shows two panels: REFERENCE (left) and CURRENT CANVAS (right). Coordinates refer "
                f"to the {CANVAS[0]}x{CANVAS[1]} canvas.\n\nMeasured error by area, worst first:\n" + error_text)
    content = [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": _data_url(image)}}]
    body = {"model": EVALUATOR_MODEL, "temperature": 0.2, "max_tokens": 16384,
            "messages": [{"role": "system", "content": PLANNER_PROMPT}, {"role": "user", "content": content}]}
    record = {"round": round_index, "model": EVALUATOR_MODEL, "system_prompt": PLANNER_PROMPT,
              "user_prompt": content[0]["text"],
              "image_assets": [str(image)]}
    regions = None
    for attempt in range(1, 3):
        try:
            response = _call_crusoe(api_key, body, CALL_TIMEOUT)
            choice = (response.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            text = _response_text(message)
            reasoning = message.get("reasoning_content") or ""
            record[f"attempt_{attempt}"] = {"id": response.get("id"), "usage": response.get("usage", {}),
                                            "finish_reason": choice.get("finish_reason"),
                                            "content": text, "reasoning_content": reasoning}
            regions = _valid_regions(_find_regions_json(text) or _find_regions_json(reasoning))
            break
        except (NemotronInferenceError, http.client.HTTPException, OSError, ValueError) as error:
            record[f"attempt_{attempt}_error"] = f"{type(error).__name__}: {str(error)[:300]}"
    if regions is None:
        regions = grid_regions()
        record["fallback"] = "grid"
        log(f"round {round_index}: planner failed, using the grid")
    record["regions"] = regions
    out = run_dir / "handoffs" / f"region-plan-r{round_index:03d}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))
    log(f"round {round_index}: regions " + ", ".join(f"{r['region_id']}x{r['painters']}" for r in regions))
    return regions


def error_overview(render: np.ndarray, reference: np.ndarray, cols: int = 6, rows: int = 4) -> str:
    cw, ch = CANVAS[0] // cols, CANVAS[1] // rows
    cells = []
    for j in range(rows):
        for i in range(cols):
            err = float(np.abs(render[j * ch:(j + 1) * ch, i * cw:(i + 1) * cw] - reference[j * ch:(j + 1) * ch, i * cw:(i + 1) * cw]).mean())
            cells.append((err, f"  x {i * cw}-{(i + 1) * cw}, y {j * ch}-{(j + 1) * ch}: error {err:.2f}"))
    cells.sort(reverse=True)
    return "\n".join(text for _, text in cells[:8])


def phase(round_index: int, total: int) -> tuple[str, str]:
    if round_index <= max(1, round(total * 0.2)):
        return "block-in", ("Place one LARGE shape (about 150-500 units across) that sets the main color of a big "
                            "area in your region. It may extend well beyond your region.")
    if round_index <= round(total * 0.6):
        return "forms", ("Place one MEDIUM shape (about 30-150 units) that shapes an object, edge, or shadow in "
                         "your region, fixing the worst sub-cell first.")
    return "details", ("Place one SMALL shape (about 4-40 units): a highlight, edge, texture stroke, or accent "
                       "in the worst part of your region.")


SYSTEM_PROMPT = f"""You are one of several painters working AT THE SAME TIME on one shared canvas. Each painter
works on one region of the picture (for example the sky, a mountain, or a lake). The canvas is built over many rounds; in every round each painter adds exactly ONE
shape, so the finished picture will be made of hundreds of shapes.
Your turn is one small step in that long process. Do not try to paint the whole picture, and do not try to
finish your region in one turn. Add the single shape that most improves your region right now; later turns
(yours and the other painters') will add the rest.
Return exactly one JSON object and nothing else, with no Markdown fences: an AgentProposal using only the
add_objects operation with exactly one object.
Rules: the canvas is {CANVAS[0]}x{CANVAS[1]} units, white. Shapes are paths, polygons or ellipses. Colors are only
#00ffff (cyan), #ff00ff (magenta), #ffff00 (yellow) and #000000 (black). Choose any opacity from
{OPACITY_RANGE[0]} to {OPACITY_RANGE[1]}. Strokes are at most {MAX_STROKE} wide. The shape's center must lie inside your region.
Colors mix by layering: a shape of ink color I at opacity a turns the canvas color C into (1-a)*C + a*I.
So cyan over yellow gives green, magenta over yellow gives red/orange, cyan over magenta gives blue, and
black darkens. Low opacity (0.15-0.4) makes light tints and lets later shapes mix. You are given measured
"best ink" suggestions for each part of your region: they say which ink and opacity, painted over what is
there now, moves that spot closest to the reference color. Prefer them unless the shape clearly needs
another color. Blue, green, red and orange always need TWO layers of different inks (for example
magenta then cyan for blue), so lay the first layer now and expect the second on a later turn. If the spot
already has the first layer, add the second. Avoid large solid black unless the reference is truly black."""

SCHEMA_NOTE = ("Primitive schemas are strict. A path has type, points, smooth, closed, fill, stroke, width, opacity; "
               "a polygon has only type, points, fill, stroke, width, opacity; an ellipse has only type, cx, cy, rx, ry, "
               "fill, stroke, width, opacity. points is a JSON array of 3-256 [x, y] pairs of plain numbers. "
               "Filled shapes use stroke 'none' and width 0.")

# --------------------------------------------------------------------------- images


def outlined(image: Image.Image, region: dict, path: Path) -> Path:
    image = image.convert("RGB").resize(CANVAS)
    draw = ImageDraw.Draw(image)
    for k in range(3):
        draw.rectangle([region["x0"] + k, region["y0"] + k, region["x1"] - 1 - k, region["y1"] - 1 - k], outline=(255, 0, 0))
    image.save(path)
    return path


def crop(image: Image.Image, region: dict, path: Path) -> Path:
    image = image.convert("RGB").resize(CANVAS).crop((region["x0"], region["y0"], region["x1"], region["y1"]))
    scale = 384 / max(image.size)
    image.resize((max(1, round(image.size[0] * scale)), max(1, round(image.size[1] * scale)))).save(path)
    return path


def posterized(image: Image.Image, region: dict, path: Path, colors: int = 6) -> Path:
    """The region of the reference reduced to a few flat colors, so its shapes read clearly."""
    part = image.convert("RGB").resize(CANVAS).crop((region["x0"], region["y0"], region["x1"], region["y1"]))
    flat = part.quantize(colors=colors, method=Image.Quantize.MEDIANCUT).convert("RGB")
    scale = 384 / max(part.size)
    flat.resize((max(1, round(part.size[0] * scale)), max(1, round(part.size[1] * scale))), Image.NEAREST).save(path)
    return path


def to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB in 0-1 to CIE Lab (D65), so color distance tracks perceived hue, not just brightness."""
    rgb = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    xyz = rgb @ np.array([[0.4124, 0.2126, 0.0193], [0.3576, 0.7152, 0.1192], [0.1805, 0.0722, 0.9505]])
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def _delta(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(to_lab(a) - to_lab(b)))


def ink_recipes(current: np.ndarray, target: np.ndarray) -> str:
    """Best one-coat and two-coat ink/opacity choices over the current color (source-over blending),
    scored by perceptual (Lab) distance so hue matters as much as lightness."""
    alphas = np.round(np.arange(0.1, 1.0001, 0.05), 2)
    singles = []
    for name, ink in INKS.items():
        for a in alphas:
            result = (1 - a) * current + a * np.array(ink)
            singles.append((_delta(result, target), name, a, result))
    singles.sort(key=lambda item: item[0])
    best = singles[0]
    doubles = []
    for _, name1, a1, mid in singles[:40]:
        for name2, ink in INKS.items():
            for a in alphas[::2]:
                result = (1 - a) * mid + a * np.array(ink)
                doubles.append((_delta(result, target), name1, a1, name2, a))
    two = min(doubles, key=lambda item: item[0])
    now = _delta(current, target)
    if now < 6:
        return "already close; leave it"
    if two[0] < best[0] - 8:
        return (f"two layers: {two[1]} {two[2]:.2f} first, {two[3]} {two[4]:.2f} on top (color difference "
                f"{now:.0f} -> {two[0]:.0f}; one layer can only reach {best[0]:.0f})")
    return f"{best[1]} at opacity {best[2]:.2f} (color difference {now:.0f} -> {best[0]:.0f})"


def dominant_colors(pixels: np.ndarray, count: int = 2) -> list[tuple[float, np.ndarray]]:
    """The main colors of a patch with their area shares (median-cut quantize)."""
    patch = Image.fromarray((pixels * 255).astype("uint8"))
    flat = patch.quantize(colors=count, method=Image.Quantize.MEDIANCUT)
    palette = np.array(flat.getpalette()[:count * 3], dtype=float).reshape(-1, 3) / 255
    shares = np.bincount(np.asarray(flat).ravel(), minlength=count)[:count] / flat.size[0] / flat.size[1]
    return sorted(((float(shares[i]), palette[i]) for i in range(count) if shares[i] > 0.05),
                  key=lambda item: -item[0])


def labelled_sheet(panels: list[tuple[str, Path]], path: Path, columns: int) -> Path:
    """Stitch several images into one labelled sheet: Super-VL accepts one image per request."""
    images = []
    for label, source in panels:
        img = Image.open(source).convert("RGB")
        if img.size[0] > CANVAS[0]:
            img = img.resize((CANVAS[0], round(img.size[1] * CANVAS[0] / img.size[0])))
        images.append((label, img))
    cell_w = max(img.size[0] for _, img in images)
    cell_h = max(img.size[1] for _, img in images)
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * (cell_w + 12) + 12, rows * (cell_h + 34) + 6), (235, 235, 235))
    draw = ImageDraw.Draw(sheet)
    for i, (label, img) in enumerate(images):
        x = 12 + (i % columns) * (cell_w + 12)
        y = 6 + (i // columns) * (cell_h + 34)
        draw.text((x, y + 4), label, fill=(0, 0, 0))
        sheet.paste(img, (x, y + 22))
    sheet.save(path)
    return path


def region_feedback(render: np.ndarray, reference: np.ndarray, region: dict) -> str:
    lines = []
    width, height = region["x1"] - region["x0"], region["y1"] - region["y0"]
    cols = max(1, min(3, round(width / 128)))
    rows = max(1, min(3, round(height / 128)))
    w, h = width // cols, height // rows
    for j in range(rows):
        for i in range(cols):
            x0, y0 = region["x0"] + i * w, region["y0"] + j * h
            a = render[y0:y0 + h, x0:x0 + w]
            b = reference[y0:y0 + h, x0:x0 + w]
            err = float(np.abs(a - b).mean())
            current = a.reshape(-1, 3).mean(0)
            hexa = "#%02x%02x%02x" % tuple(int(v * 255) for v in current)
            parts = []
            for share, color in dominant_colors(b):
                hexb = "#%02x%02x%02x" % tuple(int(v * 255) for v in color)
                parts.append(f"{share:.0%} is {hexb} -> {ink_recipes(current, color)}")
            lines.append((err, f"  x {x0}-{x0 + w}, y {y0}-{y0 + h}: error {err:.2f}; canvas there is {hexa}; "
                               "reference there: " + "; ".join(parts)))
    lines.sort(reverse=True)
    return "\n".join(text for _, text in lines)

# --------------------------------------------------------------------------- one worker call


def _bbox(node: dict) -> tuple[float, float, float, float]:
    if node.get("type") == "ellipse":
        return node["cx"] - node["rx"], node["cy"] - node["ry"], node["cx"] + node["rx"], node["cy"] + node["ry"]
    xs = [p[0] for p in node["points"]]
    ys = [p[1] for p in node["points"]]
    return min(xs), min(ys), max(xs), max(ys)


def _normalize_and_check(proposal: dict, context: dict, region: dict, object_id: str, agent_id: str, mutation_id: str) -> dict:
    if not isinstance(proposal, dict) or not isinstance(proposal.get("add_objects"), dict):
        raise PictureContextError("return an AgentProposal with an add_objects operation")
    objects = proposal["add_objects"].get("objects")
    if not isinstance(objects, list) or len(objects) != 1:
        raise PictureContextError("add exactly ONE object per turn")
    item = objects[0]
    node = item.get("renderer_node") if isinstance(item, dict) else None
    if not isinstance(node, dict):
        raise PictureContextError("the object needs a renderer_node")
    if node.get("type") in ("polygon", "ellipse"):
        node.pop("smooth", None)
        node.pop("closed", None)
    opacity = node.get("opacity")
    if not isinstance(opacity, (int, float)) or not OPACITY_RANGE[0] <= opacity <= OPACITY_RANGE[1]:
        raise PictureContextError(f"opacity must be between {OPACITY_RANGE[0]} and {OPACITY_RANGE[1]}")
    try:
        x0, y0, x1, y1 = _bbox(node)
    except (KeyError, TypeError, IndexError):
        raise PictureContextError("shape geometry is malformed; check " + SCHEMA_NOTE)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if not (region["x0"] <= cx <= region["x1"] and region["y0"] <= cy <= region["y1"]):
        raise PictureContextError(
            f"the shape's center ({cx:.0f}, {cy:.0f}) is outside your region x {region['x0']}-{region['x1']}, "
            f"y {region['y0']}-{region['y1']}")
    item["object_id"] = object_id
    item.setdefault("intent", "")
    item.setdefault("tags", [])
    proposal.update(protocol_version=PROTOCOL_VERSION, agent_id=agent_id, mutation_id=mutation_id,
                    base_revision=context["revision"])
    proposal["add_objects"].update(layer_id="marks", before_object_id="")
    for key in ("critique", "rationale"):
        if not isinstance(proposal.get(key), str) or not proposal[key].strip():
            proposal[key] = "(none given)"
        proposal[key] = proposal[key][:1500]
    if len(proposal["critique"]) < 15:
        raise PictureContextError("critique must say what differs in your region")
    for extra in set(proposal) - {"protocol_version", "agent_id", "mutation_id", "base_revision", "critique", "rationale", "add_objects"}:
        proposal.pop(extra)
    apply_proposal(context, proposal)  # full policy + schema validation against this round's base
    return proposal


def system_prompt() -> str:
    if not OPTIONS["skills"].strip():
        return SYSTEM_PROMPT
    return SYSTEM_PROMPT + "\n\nLearned painting skills (follow these):\n" + OPTIONS["skills"].strip()


def shape_mask(node: dict) -> np.ndarray:
    """Pixels a primitive covers on the canvas (top-level nodes carry no transform)."""
    mask = Image.new("L", CANVAS, 0)
    draw = ImageDraw.Draw(mask)
    if node["type"] == "ellipse":
        draw.ellipse([node["cx"] - node["rx"], node["cy"] - node["ry"], node["cx"] + node["rx"], node["cy"] + node["ry"]], fill=255)
    else:
        points = [tuple(point) for point in node["points"]]
        if node.get("fill") not in (None, "none") and len(points) >= 3:
            draw.polygon(points, fill=255)
        if node.get("stroke") not in (None, "none") and node.get("width", 0) > 0:
            draw.line(points + ([points[0]] if node.get("closed") else []), fill=255, width=max(1, round(node["width"])))
    return np.asarray(mask) > 0


def unfinished_layer(node: dict, render: np.ndarray, reference: np.ndarray) -> str:
    """If the canvas under this shape would get much closer to the reference with one more layer of a
    different ink, describe that layer (the second half of a two-layer color)."""
    mask = shape_mask(node)
    if mask.sum() < 150:
        return ""
    current = render[mask].mean(0)
    target = dominant_colors(reference[mask].reshape(-1, 1, 3), count=2)[0][1]
    before = _delta(current, target)
    own = INK_NAMES.get(str(node.get("fill") if node.get("fill") not in (None, "none") else node.get("stroke")).lower())
    best = None
    for name, ink in INKS.items():
        if name == own:
            continue
        for a in np.round(np.arange(0.1, 1.0001, 0.05), 2):
            after = _delta((1 - a) * current + a * np.array(ink), target)
            if best is None or after < best[0]:
                best = (after, name, a)
    if best is None or before < 15 or before - best[0] < 10:
        return ""
    x0, y0, x1, y1 = _bbox(node)
    hext = "#%02x%02x%02x" % tuple(int(v * 255) for v in target)
    return (f"Unfinished color: your last shape ({own or 'shape'} {node['type']} at x {max(0, x0):.0f}-{min(CANVAS[0], x1):.0f}, "
            f"y {max(0, y0):.0f}-{min(CANVAS[1], y1):.0f}) still needs {best[1]} at opacity {best[2]:.2f} on top to reach "
            f"the reference color {hext} there (color difference {before:.0f} -> {best[0]:.0f}). Add that layer now "
            "with a shape covering the same area, unless something more important in your region changed.")


def worker(context: dict, region: dict, painter: int, round_index: int, total: int, run_dir: Path,
           render: np.ndarray, reference: np.ndarray, images: list[Path], memory: list[str], api_key: str,
           pending: str = "") -> dict:
    rid = f"{region['region_id']}-p{painter}"
    revision = context["revision"]
    agent_id = f"painter-{rid}-r{round_index:03d}"
    mutation_id = f"{context['run_id'][:30]}-{rid}-r{round_index:03d}"
    object_id = f"r{round_index:03d}-{rid}"
    phase_name, phase_text = phase(round_index, total)
    template = {
        "protocol_version": PROTOCOL_VERSION, "agent_id": agent_id, "mutation_id": mutation_id,
        "base_revision": revision,
        "critique": "what most differs in your region right now (one sentence)",
        "rationale": "why this one shape is the best next step",
        "add_objects": {"layer_id": "marks", "before_object_id": "", "objects": [{
            "object_id": object_id,
            "renderer_node": {"type": "polygon", "points": [[0, 0], [100, 0], [50, 80]], "fill": "#00ffff",
                              "stroke": "none", "width": 0, "opacity": 0.3},
            "intent": "what this shape represents", "tags": []}]},
    }
    user = "\n\n".join([
        f"Round {round_index} of {total} (phase: {phase_name}). Your region is '{region['label']}': "
        f"x {region['x0']}-{region['x1']}, y {region['y0']}-{region['y1']}."
        + (f" Focus: {region['focus']}" if region.get("focus") else "")
        + (f" You are painter {painter} of {region['painters']} in this region; work on a different part "
           "of it than the other painter (painter 1 takes the left/upper part, painter 2 the right/lower part)."
           if region["painters"] > 1 else ""),
        phase_text,
        *([pending] if pending else []),
        "The image is a sheet of five labelled panels: (1) CANVAS NOW, the whole canvas with your region "
        "outlined in red; (2) REFERENCE, the whole reference with the same outline; (3) REFERENCE ZOOM and "
        "(4) CANVAS ZOOM, your region enlarged; (5) REFERENCE SIMPLIFIED, your region reduced to a few flat "
        "colors to show its shapes clearly. The red outline is not part of either picture. Coordinates always "
        f"refer to the full {CANVAS[0]}x{CANVAS[1]} canvas, not to the panels.",
        "Measured error in your region, worst first, with the best ink for each spot:\n"
        + region_feedback(render, reference, region),
        "Your own last shapes:\n" + ("\n".join(memory[-MEMORY:]) if memory else "  (none yet)"),
        SCHEMA_NOTE,
        "Required output shape (replace example values; exactly one object):\n" + json.dumps(template, separators=(",", ":")),
    ])
    content = [{"type": "text", "text": user}] + [
        {"type": "image_url", "image_url": {"url": _data_url(path)}} for path in images]  # one sheet
    attempts, correction = [], ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        body = {"model": WORKER_MODEL, "temperature": 0.3, "max_tokens": MAX_TOKENS,
                "messages": [{"role": "system", "content": system_prompt()},
                             {"role": "user", "content": content if not correction else
                              [{"type": "text", "text": user + correction}] + content[1:]}]}
        log = {"attempt": attempt, "system_prompt": system_prompt(), "user_prompt": user + correction,
               "image_assets": [str(p.relative_to(run_dir)) for p in images]}
        started = time.monotonic()
        try:
            response = _call_crusoe(api_key, body, CALL_TIMEOUT)
        except (NemotronInferenceError, http.client.HTTPException, OSError) as error:
            log["error"] = f"{type(error).__name__}: {str(error)[:300]}"
            attempts.append(log)
            continue
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = _response_text(message)
        log["response"] = {"id": response.get("id"), "model": response.get("model"),
                           "finish_reason": choice.get("finish_reason"), "usage": response.get("usage", {}),
                           "content": text, "reasoning_content": message.get("reasoning_content") or "",
                           "elapsed_seconds": round(time.monotonic() - started, 2)}
        try:
            proposal = _normalize_and_check(parse_proposal(text or log["response"]["reasoning_content"]),
                                            context, region, object_id, agent_id, mutation_id)
        except (NemotronInferenceError, PictureContextError, ValueError, TypeError) as error:
            log["validation_error"] = str(error)[:500]
            attempts.append(log)
            correction = (f"\n\nYour previous answer was rejected: {str(error)[:300]}. "
                          "Return one corrected JSON object only.")
            continue
        log["accepted_proposal"] = deepcopy(proposal)
        attempts.append(log)
        return {"region_id": rid, "proposal": proposal, "attempts": attempts}
    return {"region_id": rid, "proposal": None, "attempts": attempts}

# --------------------------------------------------------------------------- rounds


def blank_context(run_id: str, reference: Path) -> dict:
    placeholder = {"type": "ellipse", "cx": 0, "cy": 0, "rx": 1, "ry": 1,
                   "fill": "none", "stroke": "none", "width": 0, "opacity": 0}
    scene = {"version": 1, "canvas": {"width": CANVAS[0], "height": CANVAS[1], "background": "#ffffff"},
             "motifs": {}, "nodes": [placeholder]}
    return create_context(
        scene, run_id=run_id, prompt_set=[PROMPT],
        composition={"title": f"test_3 parallel one-shape recreation of {reference.name}",
                     "approach": "Super-VL plans content-based focus regions (re-planned every "
                                 f"{REPLAN_EVERY} rounds); one painter call per region slot adds one shape; "
                                 "shapes merged per round",
                     "source_image": str(reference), "layers": [{"label": "marks"}],
                     "options": {"finish_layers": OPTIONS["finish_layers"], "skills_file": bool(OPTIONS["skills"])},
                     "region_plans": "handoffs/region-plan-rNNN.json"},
        layer_specs=[{"layer_id": "marks", "label": "marks", "end_index": 1}],
        references=[reference_record(reference)],
    )


def _area(proposal: dict) -> float:
    x0, y0, x1, y1 = _bbox(proposal["add_objects"]["objects"][0]["renderer_node"])
    return (x1 - x0) * (y1 - y0)


def run_round(context_path: Path, round_index: int, total: int, run_dir: Path, reference_path: Path,
              ref_image: Image.Image, ref_array: np.ndarray, plan: list[dict], memories: dict,
              pool: ThreadPoolExecutor, record_root: Path, api_key: str, log, pendings: dict | None = None) -> Path:
    pendings = {} if pendings is None else pendings
    context = read_context(context_path)
    revision = context["revision"]
    render_path = run_dir / "steps" / f"{revision:03d}-drawing.png"
    render_image = Image.open(render_path).convert("RGB").resize(CANVAS)
    render_array = np.asarray(render_image, dtype=float) / 255
    views = run_dir / "views" / f"r{round_index:03d}"
    views.mkdir(parents=True, exist_ok=True)
    futures = {}
    for region in plan:
        rid = region["region_id"]
        images = [
            outlined(render_image, region, views / f"{rid}-canvas.png"),
            outlined(ref_image, region, views / f"{rid}-reference.png"),
            crop(ref_image, region, views / f"{rid}-reference-zoom.png"),
            crop(render_image, region, views / f"{rid}-canvas-zoom.png"),
            posterized(ref_image, region, views / f"{rid}-reference-flat.png"),
        ]
        labels = ["1 CANVAS NOW", "2 REFERENCE", "3 REFERENCE ZOOM", "4 CANVAS ZOOM", "5 REFERENCE SIMPLIFIED"]
        images = [labelled_sheet(list(zip(labels, images)), views / f"{rid}-sheet.png", columns=2)]
        for painter in range(1, region["painters"] + 1):
            key = f"{rid}-p{painter}"
            futures[pool.submit(worker, deepcopy(context), region, painter, round_index, total, run_dir,
                                render_array, ref_array, images, memories.setdefault(key, []), api_key,
                                pendings.get(key, ""))] = key
    done, pending = wait(futures, timeout=ROUND_DEADLINE)
    for future in pending:
        log(f"round {round_index}: worker {futures[future]} timed out; continuing without it")
    results = [future.result() for future in done]
    worker_logs = run_dir / "handoffs" / "workers"
    worker_logs.mkdir(parents=True, exist_ok=True)
    accepted = []
    owner = {}
    for result in sorted(results, key=lambda item: item["region_id"]):
        (worker_logs / f"r{round_index:03d}-{result['region_id']}.json").write_text(json.dumps(
            {"round": round_index, "base_revision": revision, "region_id": result["region_id"],
             "model": WORKER_MODEL, "accepted": result["proposal"] is not None,
             "attempts": result["attempts"]}, indent=1, allow_nan=False))
        if result["proposal"] is not None:
            accepted.append(result["proposal"])
            owner[result["proposal"]["agent_id"]] = result["region_id"]
            node = result["proposal"]["add_objects"]["objects"][0]["renderer_node"]
            x0, y0, x1, y1 = _bbox(node)
            color = node.get("fill") if node.get("fill") not in (None, "none") else node.get("stroke")
            memories[result["region_id"]].append(
                f"  round {round_index}: {node['type']} {color} opacity {node.get('opacity')} "
                f"x {x0:.0f}-{x1:.0f} y {y0:.0f}-{y1:.0f} ({result['proposal']['add_objects']['objects'][0].get('intent', '')[:60]})")
    if not accepted:
        log(f"round {round_index}: no worker produced a valid shape")
        return context_path
    accepted.sort(key=_area, reverse=True)  # big shapes first so they never bury details from the same round
    merged = {
        "protocol_version": PROTOCOL_VERSION,
        "agent_id": f"round-merge-r{round_index:03d}",
        "mutation_id": f"{context['run_id'][:40]}-merge-r{round_index:03d}",
        "base_revision": revision,
        "critique": " | ".join(f"{p['agent_id']}: {p['critique']}" for p in accepted)[:3900],
        "rationale": f"round {round_index}: {len(accepted)} region painters each added one shape",
        "add_objects": {"layer_id": "marks", "before_object_id": "",
                        "objects": [p["add_objects"]["objects"][0] for p in accepted]},
    }
    usage = [a["response"]["usage"] for r in results for a in r["attempts"] if a.get("response")]
    provenance = {"model": WORKER_MODEL, "round": round_index, "workers": [r["region_id"] for r in results],
                  "accepted_regions": [p["agent_id"] for p in accepted],
                  "worker_logs": f"handoffs/workers/r{round_index:03d}-*.json",
                  "completion_tokens": sum(u.get("completion_tokens") or 0 for u in usage),
                  "prompt_tokens": sum(u.get("prompt_tokens") or 0 for u in usage)}
    _, next_path = apply_and_record(context, merged, root=record_root, final=round_index == total,
                                    inference_provenance=provenance)
    for key in list(pendings):
        pendings.pop(key)
    if OPTIONS["finish_layers"]:
        new_render = np.asarray(Image.open(run_dir / "steps" / f"{revision + 1:03d}-drawing.png").convert("RGB")
                                .resize(CANVAS), dtype=float) / 255
        for proposal in accepted:
            note = unfinished_layer(proposal["add_objects"]["objects"][0]["renderer_node"], new_render, ref_array)
            if note:
                pendings[owner[proposal["agent_id"]]] = note
        (run_dir / "handoffs" / f"pending-layers-r{round_index:03d}.json").write_text(json.dumps(pendings, indent=1))
    return next_path

# --------------------------------------------------------------------------- reporting and evals


def sheets(run_dir: Path, reference: Path, milestones=(1, 5, 10, 20, 40)) -> None:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    steps = manifest["steps"]
    w, h = CANVAS
    sheet = Image.new("RGB", (w * 2 + 30, h + 40), "white")
    draw = ImageDraw.Draw(sheet)
    sheet.paste(Image.open(reference).convert("RGB").resize((w, h)), (10, 30))
    sheet.paste(Image.open(run_dir / steps[-1]["png"]).convert("RGB").resize((w, h)), (w + 20, 30))
    draw.text((10, 8), f"reference: {reference.name}", fill="black")
    draw.text((w + 20, 8), f"final after {len(steps) - 1} rounds", fill="black")
    sheet.save(run_dir / "comparison.png")
    tw, th = 384, 256
    tiles = [("reference", Image.open(reference))]
    for target in milestones:
        index = min(target, len(steps) - 1)
        tiles.append((f"round {index}" if index == target else f"round {index} (asked {target})",
                      Image.open(run_dir / steps[index]["png"])))
    strip = Image.new("RGB", (len(tiles) * (tw + 8) + 8, th + 40), "white")
    draw = ImageDraw.Draw(strip)
    for i, (label, image) in enumerate(tiles):
        x = 8 + i * (tw + 8)
        strip.paste(image.convert("RGB").resize((tw, th)), (x, 30))
        draw.text((x, 10), label, fill="black")
    strip.save(run_dir / "milestones.png")


def skill_vs_blank(render: np.ndarray, reference: np.ndarray) -> float:
    """1 = matches the reference, 0 = no better than the white canvas (the evals' definition)."""
    from drawing_sandbox.evals.image_metrics import tone_mae
    blank = tone_mae(np.ones_like(reference), reference, TONE_BLOCK)
    return 1 - tone_mae(render, reference, TONE_BLOCK) / blank if blank > 0 else 1.0


def mark_final(run_dir: Path) -> None:
    """Promote the last recorded step to the run's final output (used when a run stops early)."""
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    last = manifest["steps"][-1]
    if manifest.get("final_step") == last["index"]:
        return
    (run_dir / "final-scene.json").write_text((run_dir / last["scene"]).read_text())
    (run_dir / "final-drawing.svg").write_text((run_dir / last["svg"]).read_text())
    (run_dir / "final-drawing.png").write_bytes((run_dir / last["png"]).read_bytes())
    last["final"] = True
    manifest["final_step"] = last["index"]
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")


def offline_evals(run_dir: Path, reference: Path) -> str:
    from drawing_sandbox.evals.__main__ import main as evals_main
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        evals_main(["run", str(run_dir), "--target", str(reference)])
    text = buffer.getvalue()
    (run_dir / "evals.txt").write_text(text)
    return " ".join(next((line.split() for line in text.splitlines() if line.strip().startswith("overall")), []))


def super_vl(run_dir: Path, final_path: Path, reference: Path, record_root: Path, api_key: str, rounds: int) -> str:
    base = read_context(run_dir / "handoffs" / "context-000.json")
    final = read_context(final_path)
    plan = {"plan_id": "dynamic-regions", "approach": f"{rounds} rounds of parallel region painters, one shape "
            f"each; Super-VL focus regions re-planned every {REPLAN_EVERY} rounds"}
    try:
        evaluation, log = infer_evaluation(
            base, final, plan, api_key=api_key, baseline_image=run_dir / "steps" / "000-drawing.png",
            final_image=run_dir / "steps" / f"{final['revision']:03d}-drawing.png",
            reference_images=[reference], timeout=300)
    except (NemotronInferenceError, http.client.HTTPException, OSError) as error:
        return f"Super-VL evaluation failed: {str(error)[:200]}"
    write_evaluation_artifacts(evaluation, log, root=record_root)
    return (f"reference_match {evaluation['reference_match_before']} -> {evaluation['reference_match_after']}, "
            f"composition {evaluation['composition_before']} -> {evaluation['composition_after']}: {evaluation['summary']}")


def run_one(run_id: str, reference: Path, rounds: int, record_root: Path, api_key: str, log,
            stop_after: int | None = None, resume: bool = False, target_score: float = 0.9,
            patience: int = 5, min_gain: float = 0.005) -> dict:
    started = time.time()
    run_dir = record_root / run_id
    state_path = run_dir / "state.json"
    if resume and state_path.exists():
        state = json.loads(state_path.read_text())
        path = sorted((run_dir / "handoffs").glob("context-*.json"))[-1]
        memories, plan, pendings = state["memories"], state["plan"], state["pendings"]
        scores = state.get("scores", [])
        first_round = state["round"] + 1
        log(f"{run_id}: resuming at round {first_round}")
    else:
        path = initialize_recorded_context(blank_context(run_id, reference), root=record_root)
        memories, plan, pendings, first_round = {}, [], {}, 1
        scores = []
    ref_image = Image.open(reference).convert("RGB").resize(CANVAS)
    ref_array = np.asarray(ref_image, dtype=float) / 255
    if not scores:
        scores = [{"round": 0, "score": round(skill_vs_blank(np.ones_like(ref_array), ref_array), 4)}]
    stop_reason = None
    empty_rounds = 0
    paused = False
    with ThreadPoolExecutor(max_workers=MAX_PAINTERS) as pool:
        for round_index in range(first_round, rounds + 1):
            if round_index == 1 or (round_index - 1) % REPLAN_EVERY == 0:
                revision = read_context(path)["revision"]
                render_path = run_dir / "steps" / f"{revision:03d}-drawing.png"
                render_array = np.asarray(Image.open(render_path).convert("RGB").resize(CANVAS), dtype=float) / 255
                plan = plan_regions(run_dir, round_index, reference,
                                    render_path if round_index > 1 else None,
                                    error_overview(render_array, ref_array), api_key,
                                    lambda message: log(f"{run_id}: {message}"))
            before = path
            path = run_round(path, round_index, rounds, run_dir, reference, ref_image, ref_array, plan,
                             memories, pool, record_root, api_key, log, pendings)
            revision = read_context(path)["revision"]
            render_now = np.asarray(Image.open(run_dir / "steps" / f"{revision:03d}-drawing.png").convert("RGB")
                                    .resize(CANVAS), dtype=float) / 255
            score = skill_vs_blank(render_now, ref_array)
            scores.append({"round": round_index, "score": round(score, 4)})
            (run_dir / "scores.json").write_text(json.dumps(scores, indent=1))
            state_path.write_text(json.dumps({"round": round_index, "plan": plan, "memories": memories,
                                              "pendings": pendings, "scores": scores}, indent=1))
            log(f"{run_id}: round {round_index} score {score:.3f}")
            if target_score and score >= target_score:
                stop_reason = f"reached target {target_score} (score {score:.3f}) at round {round_index}"
            elif patience and len(scores) > patience:
                best_before = max(item["score"] for item in scores[:-patience])
                if max(item["score"] for item in scores[-patience:]) < best_before + min_gain:
                    stop_reason = (f"plateau: no gain of {min_gain} in the last {patience} rounds "
                                   f"(best {max(item['score'] for item in scores):.3f}) at round {round_index}")
            if stop_reason:
                log(f"{run_id}: stopping, {stop_reason}")
                break
            empty_rounds = empty_rounds + 1 if path == before else 0
            if empty_rounds >= 3:
                log(f"{run_id}: stopping after 3 rounds without any valid shape")
                break
            if round_index % 5 == 0:
                manifest = json.loads((run_dir / "manifest.json").read_text())
                log(f"{run_id}: round {round_index}/{rounds}, {manifest['steps'][-1]['metrics'].get('expanded_shapes')} shapes")
            if stop_after is not None and round_index >= stop_after and round_index < rounds:
                paused = True
                log(f"{run_id}: paused after round {round_index} (resume with --resume)")
                break
    if not paused:
        mark_final(run_dir)
    sheets(run_dir, reference)
    eval_text = offline_evals(run_dir, reference)
    verdict = "skipped while paused" if paused else super_vl(run_dir, path, reference, record_root, api_key, rounds)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    return {"run_id": run_id, "reference": str(reference.relative_to(ROOT)), "rounds": len(manifest["steps"]) - 1,
            "shapes": manifest["steps"][-1]["metrics"].get("expanded_shapes"), "evals": eval_text,
            "score": scores[-1]["score"], "best_score": max(item["score"] for item in scores),
            "stop_reason": stop_reason, "super_vl": verdict, "paused": paused, "seconds": round(time.time() - started)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--level", choices=sorted(TARGETS), required=True)
    parser.add_argument("--only", help="run just this run_id")
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--name", default="test_3", help="output folder under the repo root, e.g. test_3b")
    parser.add_argument("--finish-layers", action="store_true",
                        help="tell each painter when its last shape still needs a second ink layer")
    parser.add_argument("--skills", type=Path, help="text file of learned skills appended to painter instructions")
    parser.add_argument("--stop-after", type=int, help="pause after this round (the round plan still uses --rounds)")
    parser.add_argument("--resume", action="store_true", help="continue paused runs from their saved state")
    parser.add_argument("--target-score", type=float, default=0.9,
                        help="stop once skill_vs_blank reaches this (0 disables)")
    parser.add_argument("--patience", type=int, default=5,
                        help="stop after this many rounds without --min-gain improvement (0 disables)")
    parser.add_argument("--min-gain", type=float, default=0.005)
    args = parser.parse_args()
    OPTIONS["finish_layers"] = args.finish_layers
    if args.skills is not None and args.skills.exists():
        OPTIONS["skills"] = args.skills.read_text()
    out_root = ROOT / args.name
    record_root = out_root / "runs"
    record_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / f"log-{args.level}.txt"

    def log(message: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {message}"
        print(line, flush=True)
        with log_path.open("a") as handle:
            handle.write(line + "\n")

    api_key = load_crusoe_api_key(ROOT / ".env")
    for run_id, rel in TARGETS[args.level]:
        if args.only and run_id != args.only:
            continue
        if (record_root / run_id).exists() and not args.resume:
            log(f"{run_id}: already exists, skipping")
            continue
        log(f"{run_id}: starting ({rel})")
        try:
            result = run_one(run_id, ROOT / rel, args.rounds, record_root, api_key, log,
                             stop_after=args.stop_after, resume=args.resume, target_score=args.target_score,
                             patience=args.patience, min_gain=args.min_gain)
        except Exception as error:  # keep the other targets going
            result = {"run_id": run_id, "reference": rel, "error": f"{type(error).__name__}: {str(error)[:300]}"}
            log(f"{run_id}: FAILED: {result['error']}")
        log(f"{run_id}: finished: {json.dumps(result)}")
        out = out_root / f"results-{args.level}.json"
        previous = json.loads(out.read_text()) if out.exists() else []
        out.write_text(json.dumps(previous + [result], indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
