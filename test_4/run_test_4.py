#!/usr/bin/env python3
"""test_4: a tree of Super-VL calls (see test_4/DESIGN.md).

DIRECTOR (1 call) -> AREA LEADS (parallel) -> PATCH PAINTERS (3 candidates each, best kept only if it
improves the canvas) -> REFINERS where error remains (recursing to MAX_DEPTH). The model draws every
outline itself; the harness only offers measuring tools (`measure`, `recipe`, `preview`) as JSON actions,
and commits with deterministic keep-if-better checks. Stops at the Lab target, on budget, or when done.

usage (from the repo root):
    drawing-sandbox/.venv/bin/python test_4/run_test_4.py --only easy-sea-sunset [--budget 400] [--target 0.9]
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import html
import http.client
import itertools
import json
from pathlib import Path
import sys
import threading
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "test_3")]
import run_test_3 as t3  # noqa: E402  (applies the test_3 mark rules: CMYK, opacity 0.1-1, big marks)

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from drawing_sandbox import policy  # noqa: E402
from drawing_sandbox.iteration_context import (  # noqa: E402
    EVALUATOR_MODEL, PROTOCOL_VERSION, apply_and_record, create_context, initialize_recorded_context,
    materialize_scene, read_context,
)
from drawing_sandbox.nemotron_inference import (  # noqa: E402
    NemotronInferenceError, _call_crusoe, _data_url, _response_text, infer_evaluation, load_crusoe_api_key,
    write_evaluation_artifacts,
)
from drawing_sandbox.renderer import check_scene, render  # noqa: E402
from nemotron_iteration import reference_record  # noqa: E402

MODEL = EVALUATOR_MODEL           # Super-VL for every node
CANVAS = t3.CANVAS                # 768 x 512
MAX_DEPTH = 4                     # director 0, area 1, patch 2, refiner 3, sub-patch 4
CANDIDATES = 2
TOOL_TURNS = 2
FINAL_RETRIES = 2
IN_FLIGHT = 24
REFINE_DE = 8.0                   # refine a patch whose mean Lab color difference stays above this
LAB_BLOCK = 4
MAX_TOKENS = 16384
CALL_TIMEOUT = 240
# Mutable run settings; --deadline switches to the fast profile (see apply_deadline_profile).
SETTINGS = {"tool_turns": TOOL_TURNS, "max_tokens": MAX_TOKENS, "call_timeout": CALL_TIMEOUT, "in_flight": IN_FLIGHT}
SLICES = {"director": 0.20, "leads": 0.40, "painters": 0.70, "refiners": 0.93}  # fractions of the deadline


def apply_deadline_profile() -> None:
    SETTINGS.update(tool_turns=0, max_tokens=8192, call_timeout=90, in_flight=64)
_MEDIUM_NAMES = {
    "A_Sunday_on_La_Grande_Jatte,_Georges_Seurat,_1884.jpg": "medium-grande-jatte",
    "Claude_Monet,_Impression,_soleil_levant.jpg": "medium-impression-sunrise",
    "Claude_Monet_-_Waterlilies_-_Nympheas_(1908).jpg": "medium-waterlilies",
    "John_Constable_The_Hay_Wain.jpg": "medium-hay-wain",
    "Katsushika_Hokusai_-_Thirty-Six_Views_of_Mount_Fuji-_The_Great_Wave_Off_the_Coast_of_Kanagawa_-_Google_Art_Project.jpg": "medium-great-wave",
    "Paul_Cézanne_-_Mont_Sainte-Victoire_-_Google_Art_Project.jpg": "medium-mont-sainte-victoire",
    "The_Fighting_Temeraire,_JMW_Turner,_National_Gallery.jpg": "medium-fighting-temeraire",
    "Thomas_Cole_-_The_Oxbow_(Met).jpg": "medium-oxbow",
    "Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg": "medium-starry-night",
}


def _all_targets() -> dict:
    """Every reference image: easy-*, medium-*, hard-* run ids -> repo-relative paths."""
    targets = {}
    for level, folder, pattern in (("easy", "landscape-paintings", "*.png"), ("hard", "hardphotos", "*.jpg")):
        for path in sorted((ROOT / folder).glob(pattern)):
            targets[f"{level}-{path.stem.split('-', 1)[1]}"] = f"{folder}/{path.name}"
    for path in sorted((ROOT / "medphotos").glob("*.jpg")):
        name = unicodedata.normalize("NFC", path.name)  # macOS may store "é" decomposed
        targets[_MEDIUM_NAMES.get(name, "medium-" + path.stem.lower()[:30])] = f"medphotos/{path.name}"
    return targets


TARGETS = _all_targets()

# --------------------------------------------------------------------------- metrics


def _block(image: np.ndarray, size: int) -> np.ndarray:
    h, w = image.shape[0] // size * size, image.shape[1] // size * size
    return image[:h, :w].reshape(h // size, size, w // size, size, 3).mean((1, 3))


def lab_skill(render_img: np.ndarray, reference: np.ndarray) -> float:
    """Stricter score: 1 - mean Lab color difference (4x4 blocks) / that of a white canvas."""
    ref = t3.to_lab(_block(reference, LAB_BLOCK))
    blank = np.linalg.norm(t3.to_lab(_block(np.ones_like(reference), LAB_BLOCK)) - ref, axis=-1).mean()
    now = np.linalg.norm(t3.to_lab(_block(render_img, LAB_BLOCK)) - ref, axis=-1).mean()
    return float(1 - now / blank) if blank > 0 else 1.0


def mask_delta(render_img: np.ndarray, reference_lab: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return 0.0
    return float(np.linalg.norm(t3.to_lab(render_img[mask]) - reference_lab[mask], axis=-1).mean())


def polygon_mask(points: list) -> np.ndarray:
    return t3.shape_mask({"type": "polygon", "points": points, "fill": "#000000", "stroke": "none", "width": 0})


def hexcolor(rgb) -> str:
    return "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in rgb)

# --------------------------------------------------------------------------- shared canvas


class Canvas:
    """The one committed picture. Commits are serialized; reads take a consistent snapshot."""

    def __init__(self, run_dir: Path, record_root: Path, context_path: Path, reference: np.ndarray):
        self.run_dir, self.record_root = run_dir, record_root
        self.context = read_context(context_path)
        self.path = context_path
        self.reference = reference
        self.reference_lab = t3.to_lab(reference)
        self.lock = threading.Lock()
        self.commits = 0
        self.image = self._render(materialize_scene(self.context))
        self.timeline = [self._score_entry()]

    def _render(self, scene: dict) -> np.ndarray:
        return np.asarray(render(scene).convert("RGB").resize(CANVAS), dtype=float) / 255

    def _score_entry(self) -> dict:
        return {"commit": self.commits, "time": round(time.time(), 1),
                "lab": round(lab_skill(self.image, self.reference), 4),
                "lenient": round(t3.skill_vs_blank(self.image, self.reference), 4)}

    def snapshot(self) -> tuple[dict, np.ndarray]:
        with self.lock:
            return materialize_scene(self.context), self.image.copy()

    def preview(self, shapes: list[dict], base: tuple[dict, np.ndarray] | None = None) -> np.ndarray:
        scene, _ = base or self.snapshot()
        scene = deepcopy(scene)
        scene["nodes"].extend(deepcopy(shapes))
        return self._render(scene)

    def try_commit(self, shapes: list[dict], mask: np.ndarray, node_id: str, label: str, note: str) -> dict | None:
        """Commit only if, against the canvas as it is now, the patch improves and the picture doesn't worsen."""
        with self.lock:
            scene = materialize_scene(self.context)
            before_patch = mask_delta(self.image, self.reference_lab, mask)
            before_global = lab_skill(self.image, self.reference)
            trial = deepcopy(scene)
            trial["nodes"].extend(deepcopy(shapes))
            after_image = self._render(trial)
            after_patch = mask_delta(after_image, self.reference_lab, mask)
            after_global = lab_skill(after_image, self.reference)
            if not (after_patch < before_patch - 0.5 and after_global >= before_global - 0.0005):
                return None
            self.commits += 1
            objects = [{"object_id": f"{node_id[:50]}-{index}", "renderer_node": deepcopy(shape),
                        "intent": label[:200], "tags": [f"node:{node_id[:40]}"]}
                       for index, shape in enumerate(shapes, start=1)]
            proposal = {
                "protocol_version": PROTOCOL_VERSION,
                "agent_id": f"tree-{node_id}"[:64],
                "mutation_id": f"c{self.commits:04d}-{node_id}"[:64],
                "base_revision": self.context["revision"],
                "critique": f"{label}: patch color difference {before_patch:.1f} before"[:3900],
                "rationale": (note or "patch painter commit")[:3900],
                "add_objects": {"layer_id": "marks", "before_object_id": "", "objects": objects},
            }
            self.context, self.path = apply_and_record(
                self.context, proposal, root=self.record_root,
                inference_provenance={"model": MODEL, "tree_node": node_id, "commit": self.commits,
                                      "patch_delta_e": [round(before_patch, 2), round(after_patch, 2)],
                                      "lab_skill": [round(before_global, 4), round(after_global, 4)]})
            self.image = after_image
            self.timeline.append(self._score_entry())
            (self.run_dir / "scores.json").write_text(json.dumps(self.timeline, indent=1))
            return {"patch_before": before_patch, "patch_after": after_patch,
                    "global_before": before_global, "global_after": after_global, "commit": self.commits}

# --------------------------------------------------------------------------- run state (budget, tree log)


class Run:
    def __init__(self, run_id: str, run_dir: Path, canvas: Canvas, api_key: str, budget: int, target: float, log):
        self.run_id, self.run_dir, self.canvas, self.api_key = run_id, run_dir, canvas, api_key
        self.budget, self.target, self.log = budget, target, log
        self.calls = 0
        self.lock = threading.Lock()
        self.semaphore = threading.Semaphore(SETTINGS["in_flight"])
        self.t0 = time.time()
        self.deadline_seconds: float | None = None
        self.nodes: list[dict] = []
        self.ids = itertools.count(1)
        self.stop = threading.Event()
        self.stop_reason = None
        (run_dir / "tree" / "nodes").mkdir(parents=True, exist_ok=True)
        (run_dir / "tree" / "sheets").mkdir(parents=True, exist_ok=True)

    def new_node(self, kind: str, parent: str | None, depth: int, label: str, **extra) -> dict:
        with self.lock:
            node = {"id": f"{kind[0]}{next(self.ids):04d}", "kind": kind, "parent": parent, "depth": depth,
                    "label": label, "status": "running", "calls": 0, "tool_calls": [], "started": time.time(), **extra}
            self.nodes.append(node)
        return node

    def until(self, phase: str) -> float | None:
        """Absolute time by which calls of this phase must end (None without a deadline)."""
        if self.deadline_seconds is None:
            return None
        return self.t0 + SLICES[phase] * self.deadline_seconds

    def take_call(self) -> bool:
        with self.lock:
            if self.calls >= self.budget:
                self.halt(f"budget of {self.budget} calls used")
                return False
            self.calls += 1
            return True

    def halt(self, reason: str) -> None:
        if not self.stop.is_set():
            self.stop_reason = reason
            self.stop.set()
            self.log(f"{self.run_id}: stopping, {reason}")

    def check_target(self) -> None:
        score = self.canvas.timeline[-1]["lab"]
        if score >= self.target:
            self.halt(f"reached target {self.target} (lab {score:.3f})")

    def save_tree(self) -> None:
        with self.lock:
            nodes = deepcopy(self.nodes)
        (self.run_dir / "tree.json").write_text(json.dumps(
            {"run_id": self.run_id, "calls": self.calls, "budget": self.budget, "stop_reason": self.stop_reason,
             "nodes": nodes}, indent=1, default=str))

# --------------------------------------------------------------------------- tools


def _region_mask(args: dict) -> np.ndarray:
    if isinstance(args.get("polygon"), list) and len(args["polygon"]) >= 3:
        return polygon_mask([[float(p[0]), float(p[1])] for p in args["polygon"]])
    box = args.get("box") or [0, 0, CANVAS[0], CANVAS[1]]
    x0, y0, x1, y1 = (int(round(float(v))) for v in box)
    mask = np.zeros((CANVAS[1], CANVAS[0]), dtype=bool)
    mask[max(0, y0):min(CANVAS[1], y1), max(0, x0):min(CANVAS[0], x1)] = True
    return mask


def tool_measure(run: Run, args: dict, base) -> str:
    mask = _region_mask(args)
    if mask.sum() < 4:
        return "That area covers no pixels."
    _, image = base
    parts = [f"{share:.0%} {hexcolor(color)}" for share, color in t3.dominant_colors(run.canvas.reference[mask].reshape(-1, 1, 3), 3)]
    current = image[mask].mean(0)
    return (f"area {int(mask.sum())} px; reference colors: {', '.join(parts)}; canvas average {hexcolor(current)}; "
            f"mean color difference (Lab dE) {mask_delta(image, run.canvas.reference_lab, mask):.1f}")


def tool_recipe(run: Run, args: dict, base) -> str:
    target = str(args.get("target", "")).strip().lstrip("#")
    if len(target) != 6:
        return "recipe needs target as a #rrggbb hex color"
    rgb = np.array([int(target[i:i + 2], 16) for i in (0, 2, 4)]) / 255
    _, image = base
    current = image[_region_mask(args)].mean(0) if (args.get("polygon") or args.get("box")) else np.ones(3)
    return f"over current {hexcolor(current)} to reach #{target}: {t3.ink_recipes(current, rgb)}"


def tool_preview(run: Run, args: dict, base, mask: np.ndarray | None) -> str:
    try:
        shapes = validate_shapes(args.get("shapes"), base)
    except ValueError as error:
        return f"preview rejected the shapes: {error}"
    after = run.canvas.preview(shapes, base)
    _, image = base
    text = (f"whole picture score {lab_skill(image, run.canvas.reference):.3f} -> {lab_skill(after, run.canvas.reference):.3f}")
    if mask is not None:
        text = (f"inside the patch: color difference {mask_delta(image, run.canvas.reference_lab, mask):.1f} -> "
                f"{mask_delta(after, run.canvas.reference_lab, mask):.1f}; " + text)
    return text + " (nothing was committed)"


def validate_shapes(shapes, base) -> list[dict]:
    if not isinstance(shapes, list) or not 1 <= len(shapes) <= 3:
        raise ValueError("give 1 to 3 shapes")
    clean = []
    for shape in shapes:
        if not isinstance(shape, dict) or shape.get("type") not in ("polygon", "ellipse", "path"):
            raise ValueError("each shape must be a polygon, ellipse or path object")
        shape = deepcopy(shape)
        if shape["type"] in ("polygon", "ellipse"):
            shape.pop("smooth", None)
            shape.pop("closed", None)
        shape.setdefault("stroke", "none")
        shape.setdefault("width", 0)
        shape.setdefault("fill", "none")
        opacity = shape.get("opacity")
        if not isinstance(opacity, (int, float)) or not t3.OPACITY_RANGE[0] <= opacity <= t3.OPACITY_RANGE[1]:
            raise ValueError(f"opacity must be between {t3.OPACITY_RANGE[0]} and {t3.OPACITY_RANGE[1]}")
        clean.append(shape)
    scene = deepcopy(base[0])
    scene["nodes"].extend(clean)
    checks = check_scene(scene, policy.composition_constraints())
    if not checks.valid:
        raise ValueError("; ".join(checks.violations)[:400])
    return clean

# --------------------------------------------------------------------------- one agent call with JSON tool actions


# Super-VL reasons at length by default and can spend its whole output budget before answering;
# this switch (honored by Nemotron chat templates) makes it answer directly.
REASONING_OFF = "detailed thinking off\n"


def system_text() -> str:
    if SETTINGS["tool_turns"] == 0:
        return REASONING_OFF + _SYSTEM_BASE + ("\nTools are not available in this run: everything you need is measured for you in the "
                               "prompt. Reply with exactly ONE JSON object {\"final\": { ... }} and nothing else, no Markdown.")
    return REASONING_OFF + _SYSTEM_BASE + _TOOLS_TEXT.format(turns=SETTINGS["tool_turns"])


_SYSTEM_BASE = f"""You are one node in a large tree of painting agents that together recreate a reference picture
on a {CANVAS[0]}x{CANVAS[1]} white canvas (x right, y down, origin top-left). Parent nodes split the picture into
parts; leaf nodes paint shapes; every committed shape is measured, and only shapes that make the canvas
closer to the reference are kept. Your job is one small, well-defined part of that work.
Shapes are polygons, ellipses or paths. Colors are only #00ffff (cyan), #ff00ff (magenta), #ffff00 (yellow)
and #000000 (black); each shape has its own opacity from {t3.OPACITY_RANGE[0]} to {t3.OPACITY_RANGE[1]}. A shape of ink I at
opacity a turns canvas color C into (1-a)*C + a*I, so blue, green, red and orange need two layers of
different inks. Nothing can be made lighter than what is under it except by covering it with ink.
{t3.SCHEMA_NOTE}"""
_TOOLS_TEXT = """
Reply with exactly ONE JSON object and nothing else, no Markdown. Either call a tool:
  {{"tool": "measure", "args": {{"polygon": [[x,y],...]}} or {{"box": [x0,y0,x1,y1]}}}}  -> reference colors there, canvas color, error
  {{"tool": "recipe", "args": {{"target": "#rrggbb", "polygon": [[x,y],...]}}}}       -> best ink layers over what is there now
  {{"tool": "preview", "args": {{"shapes": [ ...renderer shapes... ]}}}}              -> score change if those shapes were added
or give your answer: {{"final": {{ ... }}}}. You may use at most {turns} tool calls, then you must answer."""


def _parse_action(text: str) -> dict | None:
    decoder, found = json.JSONDecoder(), None
    for index, char in enumerate(text or ""):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and ("final" in value or "tool" in value):
            found = value
    return found


def agent(run: Run, node: dict, user_text: str, image: Path, finalize, tools: dict, temperature: float = 0.3):
    """Run one node's conversation: optional tool turns, then a validated final answer (or None)."""
    system = system_text()
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": [{"type": "text", "text": user_text},
                                             {"type": "image_url", "image_url": {"url": _data_url(image)}}]}]
    transcript = {"node": node["id"], "model": MODEL, "system_prompt": system, "user_prompt": user_text,
                  "image": str(image.relative_to(run.run_dir)), "turns": []}
    tool_turns, retries = 0, 0
    result = None
    limits = [v for v in (node.get("until"), run.deadline_seconds and run.t0 + run.deadline_seconds) if v]
    while not run.stop.is_set():
        time_left = min(limits) - time.time() if limits else None
        if time_left is not None and time_left < 8:
            node["timed_out"] = True
            break
        if not run.take_call():
            break
        node["calls"] += 1
        body = {"model": MODEL, "temperature": temperature, "max_tokens": SETTINGS["max_tokens"], "messages": messages}
        call_timeout = SETTINGS["call_timeout"] if time_left is None else max(5, min(SETTINGS["call_timeout"], time_left - 2))
        turn = {"request_messages": len(messages)}
        with run.semaphore:
            started = time.monotonic()
            try:
                response = _call_crusoe(run.api_key, body, call_timeout)
            except (NemotronInferenceError, http.client.HTTPException, OSError) as error:
                turn["error"] = f"{type(error).__name__}: {str(error)[:300]}"
                transcript["turns"].append(turn)
                retries += 1
                if retries > FINAL_RETRIES:
                    break
                continue
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = _response_text(message)
        reasoning = message.get("reasoning_content") or ""
        turn.update(id=response.get("id"), usage=response.get("usage", {}), finish_reason=choice.get("finish_reason"),
                    content=content, reasoning_content=reasoning, seconds=round(time.monotonic() - started, 1))
        action = _parse_action(content) or _parse_action(reasoning)
        reply = content.strip() or (json.dumps(action) if action else "(empty)")
        messages.append({"role": "assistant", "content": reply})
        if action and "tool" in action and "final" not in action and tool_turns < SETTINGS["tool_turns"]:
            name, args = str(action.get("tool")), action.get("args") if isinstance(action.get("args"), dict) else {}
            tool_turns += 1
            if name in tools:
                try:
                    output = tools[name](args)
                except Exception as error:  # a bad tool argument is the model's problem, not the run's
                    output = f"tool error: {type(error).__name__}: {str(error)[:200]}"
            else:
                output = f"unknown tool {name!r}; available: {', '.join(tools)}"
            turn["tool"] = {"name": name, "args": args, "result": output}
            node["tool_calls"].append({"name": name, "result": output[:300]})
            left = SETTINGS["tool_turns"] - tool_turns
            messages.append({"role": "user", "content": f"TOOL RESULT ({name}): {output}\n"
                             + (f"You have {left} tool call(s) left." if left else "No tool calls left: give your final answer now.")})
        elif action and "final" in action:
            try:
                result = finalize(action["final"])
                turn["accepted"] = True
                transcript["turns"].append(turn)
                break
            except (ValueError, TypeError, KeyError) as error:
                turn["validation_error"] = str(error)[:400]
                retries += 1
                if retries > FINAL_RETRIES:
                    transcript["turns"].append(turn)
                    break
                messages.append({"role": "user", "content": f"Your final answer was rejected: {str(error)[:300]}. "
                                 "Reply with one corrected JSON object {\"final\": {...}}."})
        else:
            retries += 1
            if retries > FINAL_RETRIES:
                transcript["turns"].append(turn)
                break
            messages.append({"role": "user", "content": "Reply with exactly one JSON object: a tool call or {\"final\": {...}}."
                             + (" No tool calls left." if tool_turns >= SETTINGS["tool_turns"] else "")})
        transcript["turns"].append(turn)
    (run.run_dir / "tree" / "nodes" / f"{node['id']}.json").write_text(json.dumps(transcript, indent=1, default=str))
    return result

# --------------------------------------------------------------------------- images for nodes


def zoom(image: np.ndarray | Image.Image, box: tuple, outline: list | None, path: Path, pad: int = 16) -> Path:
    img = image if isinstance(image, Image.Image) else Image.fromarray((image * 255).astype("uint8"))
    img = img.convert("RGB").resize(CANVAS)
    x0, y0, x1, y1 = max(0, box[0] - pad), max(0, box[1] - pad), min(CANVAS[0], box[2] + pad), min(CANVAS[1], box[3] + pad)
    part = img.crop((x0, y0, x1, y1))
    scale = 384 / max(part.size)
    part = part.resize((max(1, round(part.size[0] * scale)), max(1, round(part.size[1] * scale))))
    if outline:
        draw = ImageDraw.Draw(part)
        pts = [((px - x0) * scale, (py - y0) * scale) for px, py in outline]
        draw.line(pts + [pts[0]], fill=(255, 0, 0), width=2)
    part.save(path)
    return path


def flat(image: Image.Image) -> Image.Image:
    return image.convert("RGB").resize(CANVAS).quantize(colors=8, method=Image.Quantize.MEDIANCUT).convert("RGB")


def bbox_of(points: list) -> tuple:
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return int(min(xs)), int(min(ys)), int(max(xs)) + 1, int(max(ys)) + 1

# --------------------------------------------------------------------------- node behaviours


def director(run: Run, reference_img: Image.Image) -> list[dict]:
    node = run.new_node("director", None, 0, "whole picture", until=run.until("director"))
    image = run.run_dir / "tree" / "sheets" / f"{node['id']}.png"
    reference_img.convert("RGB").resize(CANVAS).save(image)
    base = run.canvas.snapshot()
    text = (
        "You are the DIRECTOR (depth 0). The image is the reference picture. Split it into 3-8 AREAS that follow "
        "its content (sky, sun, mountains, lake, a tree...), each an axis-aligned box in canvas coordinates. Give "
        "each area an order: 0 = background painted first (sky, water, large fields), higher = painted later on "
        "top (objects, details). Areas together must cover the whole canvas and may overlap a little. Give each "
        "area its main color as base_color (#rrggbb): order-0 areas are filled with it immediately, for free, so "
        "draw background boxes to match where that color really is.\n"
        f"Total budget is {run.budget} model calls; give each area a budget share (numbers summing to at most "
        f"{int(run.budget * 0.9)}). Answer: {{\"final\": {{\"subject\": \"...\", \"areas\": [{{\"area_id\": \"sky\", "
        "\"label\": \"what is in it\", \"box\": [x0, y0, x1, y1], \"order\": 0, \"base_color\": \"#a8d0f0\", "
        "\"budget\": 60}]}}")

    def finalize(final: dict) -> list[dict]:
        areas = []
        for item in final.get("areas", [])[:8]:
            box = [int(round(float(v))) for v in item["box"]]
            x0, y0 = max(0, min(box[0], box[2])), max(0, min(box[1], box[3]))
            x1, y1 = min(CANVAS[0], max(box[0], box[2])), min(CANVAS[1], max(box[1], box[3]))
            if x1 - x0 < 24 or y1 - y0 < 24:
                continue
            areas.append({"area_id": str(item.get("area_id", "area"))[:24], "label": str(item.get("label", ""))[:120],
                          "box": [x0, y0, x1, y1], "order": int(item.get("order", 0)), "budget": int(item.get("budget", 40)),
                          "base_color": str(item.get("base_color", ""))[:7]})
        if len(areas) < 2:
            raise ValueError("need at least 2 usable areas with boxes inside the canvas")
        return areas

    tools = {"measure": lambda a: tool_measure(run, a, base), "recipe": lambda a: tool_recipe(run, a, base)}
    areas = agent(run, node, text, image, finalize, tools)
    if not areas:
        areas = []
        for area_id, box in (("top", [0, 0, CANVAS[0], CANVAS[1] // 2]), ("bottom", [0, CANVAS[1] // 2, CANVAS[0], CANVAS[1]])):
            x0, y0, x1, y1 = box
            main = t3.dominant_colors(run.canvas.reference[y0:y1, x0:x1], 2)[0][1]
            areas.append({"area_id": area_id, "label": f"{area_id} half", "box": box, "order": 0, "budget": 100,
                          "base_color": hexcolor(main)})
        node["fallback"] = True
    node.update(status="done", result=areas)
    run.log(f"{run.run_id}: director -> " + ", ".join(f"{a['area_id']}(o{a['order']})" for a in areas))
    return areas, node["id"]


def best_layers(current: np.ndarray, target: np.ndarray) -> list[tuple[str, float]]:
    """Best one or two ink layers (ink hex, opacity) over `current` toward `target`, by Lab distance."""
    inks = {"#00ffff": np.array([0, 1, 1.]), "#ff00ff": np.array([1, 0, 1.]), "#ffff00": np.array([1, 1, 0.]),
            "#000000": np.zeros(3)}
    alphas = np.round(np.arange(0.1, 1.0001, 0.05), 2)
    one = [(t3._delta((1 - a) * current + a * ink, target), [(name, float(a))], (1 - a) * current + a * ink)
           for name, ink in inks.items() for a in alphas]
    one.sort(key=lambda item: item[0])
    best = one[0][:2]
    for _, layers, mid in one[:30]:
        for name, ink in inks.items():
            for a in alphas[::2]:
                d = t3._delta((1 - a) * mid + a * ink, target)
                if d < best[0] - 2:
                    best = (d, layers + [(name, float(a))])
    return best[1]


def base_fills(run: Run, areas: list[dict]) -> None:
    """Free backgrounds: fill each lowest-order area's box with its base_color (no model calls)."""
    lowest = min(a["order"] for a in areas)
    for area in [a for a in areas if a["order"] == lowest]:
        color = area.get("base_color", "").lstrip("#")
        if len(color) != 6:
            continue
        try:
            target = np.array([int(color[i:i + 2], 16) for i in (0, 2, 4)]) / 255
        except ValueError:
            continue
        x0, y0, x1, y1 = area["box"]
        points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        mask = polygon_mask(points)
        node = run.new_node("background", None, 1, f"{area['area_id']} base fill", box=area["box"], target=f"#{color}")
        _, image = run.canvas.snapshot()
        layers = best_layers(image[mask].mean(0), target)
        shapes = [{"type": "polygon", "points": points, "fill": ink, "stroke": "none", "width": 0, "opacity": round(a, 2)}
                  for ink, a in layers]
        committed = run.canvas.try_commit(shapes, mask, node["id"], node["label"], f"free base fill #{color}")
        node.update(status="committed" if committed else "no-gain", layers=layers,
                    remaining_delta_e=round(mask_delta(run.canvas.snapshot()[1], run.canvas.reference_lab, mask), 2))
    run.log(f"{run.run_id}: base fills -> {run.canvas.commits} commits, lab {run.canvas.timeline[-1]['lab']:.3f}, 0 calls")


def lead(run: Run, parent: str, depth: int, label: str, box: list, reference_img: Image.Image,
         within: list | None = None, until: float | None = None) -> tuple[list[dict], str]:
    """Area lead (depth 1) or refiner (depth 3): the model draws patch outlines inside the box."""
    kind = "area" if within is None else "refiner"
    node = run.new_node(kind, parent, depth, label, box=box, until=until)
    base = run.canvas.snapshot()
    sheets = run.run_dir / "tree" / "sheets"
    panels = [("1 REFERENCE (area outlined)", t3.outlined(reference_img, {"x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3]}, sheets / f"{node['id']}-a.png")),
              ("2 REFERENCE ZOOM", zoom(reference_img, box, within, sheets / f"{node['id']}-b.png")),
              ("3 FLAT COLORS ZOOM", zoom(flat(reference_img), box, within, sheets / f"{node['id']}-c.png")),
              ("4 CANVAS ZOOM", zoom(base[1], box, within, sheets / f"{node['id']}-d.png"))]
    image = t3.labelled_sheet(panels, sheets / f"{node['id']}.png", columns=2)
    region = f"box x {box[0]}-{box[2]}, y {box[1]}-{box[3]}"
    scope = (f"You are an AREA LEAD (depth {depth}) for '{label}' ({region})." if within is None else
             f"You are a REFINER (depth {depth}) for the patch '{label}' (outlined in red, {region}); it is still "
             "off after painting. Split it into smaller sub-patches that capture what is still wrong.")
    text = (scope + " Draw PATCHES: flat-color pieces of the reference inside this part, each as a polygon outline "
            "in canvas coordinates that traces the real edges you see (use 4-30 points; follow curves with more points). "
            "Give each patch its target color (the reference color inside it, #rrggbb) and an order (0 = paint first). "
            "Use 2-12 patches; cover the important color areas; small details may be separate patches. "
            "Panels 2-4 are zoomed: convert to canvas coordinates using the box above. "
            + ("Use measure to check a patch's colors before answering if unsure. " if SETTINGS["tool_turns"] else "")
            + f"Measured in this part now: {tool_measure(run, {'polygon': within} if within else {'box': box}, base)}\n"
            "Answer: {\"final\": {\"patches\": [{\"patch_id\": \"sky-upper\", \"label\": \"pale blue upper sky\", "
            "\"points\": [[x,y],...], \"target_color\": \"#a8d0f0\", \"order\": 0}]}}")

    def finalize(final: dict) -> list[dict]:
        patches = []
        for item in final.get("patches", [])[:12]:
            points = [[min(CANVAS[0], max(0, float(p[0]))), min(CANVAS[1], max(0, float(p[1])))] for p in item["points"]]
            if len(points) < 3:
                continue
            mask = polygon_mask(points)
            if mask.sum() < 120:
                continue
            patches.append({"patch_id": str(item.get("patch_id", "patch"))[:24], "label": str(item.get("label", ""))[:120],
                            "points": points, "target_color": str(item.get("target_color", ""))[:7],
                            "order": int(item.get("order", 0))})
        if not patches:
            raise ValueError("no usable patches: each needs 3+ points and a visible area of at least ~120 pixels")
        return patches

    tools = {"measure": lambda a: tool_measure(run, a, base), "recipe": lambda a: tool_recipe(run, a, base)}
    patches = agent(run, node, text, image, finalize, tools) or []
    node.update(status="done" if patches else "failed", result=[{k: v for k, v in p.items() if k != "points"} for p in patches])
    return patches, node["id"]


def prepare(run: Run, parent: str, depth: int, patch: dict, reference_img: Image.Image, pool: ThreadPoolExecutor,
            until: float | None = None) -> dict:
    """Leaf, part 1: CANDIDATES painters propose shapes in parallel (nothing is committed yet)."""
    node = run.new_node("patch", parent, depth, patch["label"] or patch["patch_id"], target=patch["target_color"])
    mask = polygon_mask(patch["points"])
    box = bbox_of(patch["points"])
    base = run.canvas.snapshot()
    sheets = run.run_dir / "tree" / "sheets"
    panels = [("1 REFERENCE ZOOM (patch outlined)", zoom(reference_img, box, patch["points"], sheets / f"{node['id']}-a.png")),
              ("2 FLAT COLORS ZOOM", zoom(flat(reference_img), box, patch["points"], sheets / f"{node['id']}-b.png")),
              ("3 CANVAS ZOOM", zoom(base[1], box, patch["points"], sheets / f"{node['id']}-c.png"))]
    image = t3.labelled_sheet(panels, sheets / f"{node['id']}.png", columns=3)
    measured = tool_measure(run, {"polygon": patch["points"]}, base)
    text = (f"You are a PATCH PAINTER (depth {depth}). Paint the patch '{patch['label']}', outline (canvas coordinates): "
            f"{json.dumps([[round(x) for x in p] for p in patch['points']])}. Target color {patch['target_color']}.\n"
            f"Measured now: {measured}\n"
            "Return 1-3 shapes that make this patch match the reference: usually a polygon tracing the outline "
            "(adjust points where the outline is off), plus a second layer of another ink if the color needs it. "
            "Stay inside the patch. "
            + ("Use recipe for the ink layers and preview to check before answering.\n" if SETTINGS["tool_turns"] else
               f"Ink recipe for the target color over what is there now: "
               f"{tool_recipe(run, {'target': patch['target_color'], 'polygon': patch['points']}, base)}\n")
            + "Answer: {\"final\": {\"shapes\": [ ...renderer shapes... ], \"note\": \"what you did\"}}")
    tools = {"measure": lambda a: tool_measure(run, a, base), "recipe": lambda a: tool_recipe(run, a, base),
             "preview": lambda a: tool_preview(run, a, base, mask)}

    def one_candidate(index: int):
        child = run.new_node("candidate", node["id"], depth + 1, f"candidate {index}", until=until)
        shapes = agent(run, child, text, image, lambda final: (validate_shapes(final.get("shapes"), base), str(final.get("note", ""))[:300]),
                       tools, temperature=0.3 + 0.25 * index)
        if not shapes:
            child.update(status="failed")
            return None
        after = run.canvas.preview(shapes[0], base)
        gain = mask_delta(base[1], run.canvas.reference_lab, mask) - mask_delta(after, run.canvas.reference_lab, mask)
        child.update(status="proposed", gain=round(gain, 2), shapes=shapes[0])
        return {"shapes": shapes[0], "note": shapes[1], "gain": gain, "node": child}

    candidates = [c for c in pool.map(one_candidate, range(CANDIDATES)) if c]
    candidates.sort(key=lambda c: -c["gain"])
    return {"node": node, "patch": patch, "mask": mask, "box": box, "depth": depth, "candidates": candidates}


def commit_best(run: Run, state: dict) -> None:
    """Leaf, part 2: commit the best candidate that still improves the canvas as it is now."""
    node, patch, mask, depth = state["node"], state["patch"], state["mask"], state["depth"]
    committed = None
    for candidate in state["candidates"]:
        if candidate["gain"] <= 0.5:
            break
        committed = run.canvas.try_commit(candidate["shapes"], mask, node["id"], node["label"], candidate["note"])
        if committed:
            candidate["node"]["status"] = "committed"
            with (run.run_dir / "skills-log.jsonl").open("a") as handle:
                handle.write(json.dumps({"patch": node["label"], "target": patch["target_color"],
                                         "shapes": [{k: s.get(k) for k in ("type", "fill", "stroke", "opacity")} for s in candidate["shapes"]],
                                         "delta_e": [round(committed["patch_before"], 1), round(committed["patch_after"], 1)],
                                         "depth": depth}) + "\n")
            break
    remaining = mask_delta(run.canvas.snapshot()[1], run.canvas.reference_lab, mask)
    node.update(status="committed" if committed else "no-gain", remaining_delta_e=round(remaining, 2),
                commit=committed and committed["commit"])
    state["remaining"] = remaining
    run.check_target()


def refine(run: Run, state: dict, reference_img: Image.Image, pool) -> None:
    """Leaf, part 3: if the patch is still far off, a refiner splits it and recurses."""
    node, patch, depth = state["node"], state["patch"], state["depth"]
    if state.get("remaining", 0) > REFINE_DE and depth + 1 < MAX_DEPTH and not run.stop.is_set():
        sub, refiner_id = lead(run, node["id"], depth + 1, patch["label"] or patch["patch_id"], list(state["box"]),
                               reference_img, within=patch["points"], until=run.until("refiners"))
        run_patches(run, refiner_id, depth + 2, sub, reference_img, pool, until=run.until("refiners"))


def run_patches(run: Run, parent: str, depth: int, patches: list[dict], reference_img: Image.Image, pool,
                wait_turn=None, first_pass_done=None, until: float | None = None) -> None:
    """All patches' candidates are generated in parallel; commits then go back to front by order;
    refinements run in parallel afterwards."""
    try:
        if run.stop.is_set() or not patches:
            return
        states = list(pool.map(lambda p: prepare(run, parent, depth, p, reference_img, pool, until), patches))
        if wait_turn:
            wait_turn()
        for state in sorted(states, key=lambda st: st["patch"]["order"]):
            if run.stop.is_set():
                break
            commit_best(run, state)
        run.save_tree()
    finally:
        if first_pass_done:
            first_pass_done()
    if not run.stop.is_set():
        list(pool.map(lambda st: refine(run, st, reference_img, pool), [st for st in states if "remaining" in st]))
        run.save_tree()

# --------------------------------------------------------------------------- reporting


def write_tree_html(run: Run) -> None:
    nodes = {n["id"]: n for n in run.nodes}
    children: dict = {}
    for n in run.nodes:
        children.setdefault(n["parent"], []).append(n["id"])

    def item(node_id: str) -> str:
        n = nodes[node_id]
        bits = [f"<b>{html.escape(n['kind'])}</b> {html.escape(str(n['label']))}", f"status {html.escape(str(n['status']))}",
                f"calls {n['calls']}"]
        if "gain" in n:
            bits.append(f"gain {n['gain']}")
        if "remaining_delta_e" in n:
            bits.append(f"remaining dE {n['remaining_delta_e']}")
        sheet = f"tree/sheets/{node_id}.png"
        link = f' · <a href="{sheet}">sheet</a>' if (run.run_dir / sheet).exists() else ""
        link += f' · <a href="tree/nodes/{node_id}.json">calls</a>' if (run.run_dir / "tree" / "nodes" / f"{node_id}.json").exists() else ""
        kids = "".join(item(c) for c in children.get(node_id, []))
        return (f"<li><details{' open' if n['depth'] < 2 else ''}><summary>{' · '.join(bits)}{link}</summary>"
                f"<ul>{kids}</ul></details></li>")

    final_png = json.loads((run.run_dir / "manifest.json").read_text())["steps"][-1]["png"]
    page = (f"<!doctype html><meta charset=utf-8><title>{run.run_id} tree</title>"
            "<style>body{font:14px system-ui;margin:16px}ul{list-style:none;padding-left:18px}"
            "summary{cursor:pointer}img{max-width:48%}</style>"
            f"<h1>{run.run_id}</h1><p>{run.calls} calls · stop: {html.escape(str(run.stop_reason))} · "
            f"final lab {run.canvas.timeline[-1]['lab']} · lenient {run.canvas.timeline[-1]['lenient']}</p>"
            f'<p><img src="comparison.png"></p><ul>{"".join(item(r) for r in children.get(None, []))}</ul>')
    (run.run_dir / "tree.html").write_text(page)


def milestone_sheet(run_dir: Path, reference: Path) -> None:
    steps = json.loads((run_dir / "manifest.json").read_text())["steps"]
    picks = sorted({0, *[round((len(steps) - 1) * f) for f in (0.25, 0.5, 0.75, 1.0)]})
    tiles = [("reference", Image.open(reference))] + [(f"commit {i}", Image.open(run_dir / steps[i]["png"])) for i in picks]
    w, h = 384, 256
    sheet = Image.new("RGB", (len(tiles) * (w + 8) + 8, h + 40), "white")
    draw = ImageDraw.Draw(sheet)
    for k, (label, img) in enumerate(tiles):
        sheet.paste(img.convert("RGB").resize((w, h)), (8 + k * (w + 8), 30))
        draw.text((8 + k * (w + 8), 10), label, fill="black")
    sheet.save(run_dir / "milestones.png")

# --------------------------------------------------------------------------- run one image


def run_one(run_id: str, reference: Path, budget: int, target: float, log, deadline: float | None = None,
            runs_name: str = "runs") -> dict:
    started = time.time()
    record_root = ROOT / "test_4" / runs_name
    run_dir = record_root / run_id
    placeholder = {"type": "ellipse", "cx": 0, "cy": 0, "rx": 1, "ry": 1, "fill": "none", "stroke": "none", "width": 0, "opacity": 0}
    context = create_context(
        {"version": 1, "canvas": {"width": CANVAS[0], "height": CANVAS[1], "background": "#ffffff"}, "motifs": {}, "nodes": [placeholder]},
        run_id=run_id, prompt_set=["Recreate the reference with a tree of Super-VL agents: director, area leads, patch painters, refiners."],
        composition={"title": f"test_4 tree recreation of {reference.name}", "source_image": str(reference),
                     "approach": "see test_4/DESIGN.md", "layers": [{"label": "marks"}]},
        layer_specs=[{"layer_id": "marks", "label": "marks", "end_index": 1}], references=[reference_record(reference)])
    path = initialize_recorded_context(context, root=record_root)
    reference_img = Image.open(reference).convert("RGB").resize(CANVAS)
    canvas = Canvas(run_dir, record_root, path, np.asarray(reference_img, dtype=float) / 255)
    run = Run(run_id, run_dir, canvas, load_crusoe_api_key(ROOT / ".env"), budget, target, log)
    timer = None
    if deadline:
        run.t0, run.deadline_seconds = started, deadline
        timer = threading.Timer(max(1, deadline - (time.time() - started)), lambda: run.halt(f"deadline of {deadline:.0f}s"))
        timer.daemon = True
        timer.start()
    with ThreadPoolExecutor(max_workers=1024) as pool:
        areas, director_id = director(run, reference_img)
        base_fills(run, areas)
        # Every area works at once; an area commits only after all lower-order areas finished their first pass.
        pending = {}
        for area in areas:
            pending[area["order"]] = pending.get(area["order"], 0) + 1
        barrier = threading.Condition()

        def wait_turn(order):
            with barrier:
                while not run.stop.is_set() and any(n > 0 for o, n in pending.items() if o < order):
                    barrier.wait(timeout=5)

        def first_pass_done(order):
            with barrier:
                pending[order] -= 1
                barrier.notify_all()
            log(f"{run_id}: an order-{order} area finished its first pass; {canvas.commits} commits, "
                f"lab {canvas.timeline[-1]['lab']:.3f}, {run.calls} calls")

        def do_area(area):
            done = []

            def mark():
                if not done:
                    done.append(True)
                    first_pass_done(area["order"])
            try:
                patches, lead_id = lead(run, director_id, 1, area["label"] or area["area_id"], area["box"], reference_img,
                                        until=run.until("leads"))
                run_patches(run, lead_id, 2, patches, reference_img, pool,
                            wait_turn=lambda: wait_turn(area["order"]), first_pass_done=mark, until=run.until("painters"))
            finally:
                mark()

        list(pool.map(do_area, areas))

        # Deadline mode: spend any time left on more passes over the worst-matching areas.
        extra = 0
        while run.deadline_seconds and not run.stop.is_set():
            end = run.t0 + run.deadline_seconds
            left = end - time.time()
            if left < 45:
                break
            _, image = canvas.snapshot()
            scored = []
            for area in areas:
                x0, y0, x1, y1 = area["box"]
                mask = np.zeros((CANVAS[1], CANVAS[0]), dtype=bool)
                mask[y0:y1, x0:x1] = True
                scored.append((mask_delta(image, canvas.reference_lab, mask), area))
            worst = [a for d, a in sorted(scored, key=lambda item: -item[0]) if d > REFINE_DE][:4]
            if not worst:
                break
            extra += 1
            log(f"{run_id}: extra pass {extra} with {left:.0f}s left on " + ", ".join(a["area_id"] for a in worst))

            def extra_area(area):
                patches, lead_id = lead(run, director_id, 1, f"{area['label'] or area['area_id']} (pass {extra + 1})",
                                        area["box"], reference_img, until=time.time() + 0.4 * left)
                run_patches(run, lead_id, 2, patches, reference_img, pool, until=end - 10)

            list(pool.map(extra_area, worst))
    if not run.stop_reason:
        run.stop_reason = "tree finished"
    t3.mark_final(run_dir)
    t3.sheets(run_dir, reference, milestones=(1, 5, 10, 20, 40))
    milestone_sheet(run_dir, reference)
    if timer:
        timer.cancel()
    painting_seconds = round(time.time() - started)
    eval_text = t3.offline_evals(run_dir, reference)
    if deadline:
        verdict = "skipped in deadline mode"
    else:
        try:
            evaluation, elog = infer_evaluation(
                read_context(run_dir / "handoffs" / "context-000.json"), read_context(canvas.path),
                {"plan_id": "tree", "approach": "director -> area leads -> patch painters (best of 3, keep if better) -> refiners"},
                api_key=run.api_key, baseline_image=run_dir / "steps" / "000-drawing.png",
                final_image=run_dir / json.loads((run_dir / "manifest.json").read_text())["steps"][-1]["png"],
                reference_images=[reference], timeout=300)
            write_evaluation_artifacts(evaluation, elog, root=record_root)
            verdict = f"reference_match {evaluation['reference_match_after']}, composition {evaluation['composition_after']}: {evaluation['summary']}"
        except (NemotronInferenceError, http.client.HTTPException, OSError) as error:
            verdict = f"Super-VL evaluation failed: {str(error)[:200]}"
    run.save_tree()
    write_tree_html(run)
    final = canvas.timeline[-1]
    return {"run_id": run_id, "painting_seconds": painting_seconds, "lab": final["lab"], "lenient": final["lenient"], "commits": canvas.commits,
            "calls": run.calls, "nodes": len(run.nodes), "stop_reason": run.stop_reason, "evals": eval_text,
            "super_vl": verdict, "seconds": round(time.time() - started)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=sorted(TARGETS), required=True)
    parser.add_argument("--budget", type=int, default=400)
    parser.add_argument("--target", type=float, default=0.9, help="stop when the Lab score reaches this")
    parser.add_argument("--deadline", type=float, help="hard wall-clock limit in seconds (fast profile: no tool "
                        "turns, 4k-token answers, 60 s calls, 64 in flight, per-level time slices)")
    parser.add_argument("--name", default="runs", help="subfolder of test_4 for run output (e.g. runs-5min)")
    args = parser.parse_args()
    out = ROOT / "test_4"
    (out / args.name).mkdir(parents=True, exist_ok=True)
    if (out / args.name / args.only).exists():
        print(f"{args.only} already exists; delete test_4/{args.name}/{args.only} to redo it")
        return 1
    suffix = "" if args.name == "runs" else f"-{args.name}"
    log_path = out / f"log-{args.only}{suffix}.txt"

    def log(message: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {message}"
        print(line, flush=True)
        with log_path.open("a") as handle:
            handle.write(line + "\n")

    log(f"{args.only}: starting (budget {args.budget}, target lab {args.target})")
    try:
        if args.deadline:
            apply_deadline_profile()
        result = run_one(args.only, ROOT / TARGETS[args.only], args.budget, args.target, log, deadline=args.deadline,
                            runs_name=args.name)
    except Exception as error:
        import traceback
        result = {"run_id": args.only, "error": f"{type(error).__name__}: {str(error)[:300]}"}
        log(traceback.format_exc())
    log(f"{args.only}: finished: {json.dumps(result)}")
    (out / f"result-{args.only}{suffix}.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
