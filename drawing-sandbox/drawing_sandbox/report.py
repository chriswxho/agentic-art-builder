"""Plain-language views of scenes, checks, and recorded runs.

The renderer and recorder speak JSON so agents can parse them.  These helpers
turn the same data into text a person can skim: what the model drew, whether
it passed, and how a run progressed step by step.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from .policy import CMYK_INKS
from .renderer import SceneValidationError, expand_scene


_INK_NAMES = {value: name for name, value in CMYK_INKS.items()} | {"#ffffff": "white"}
MAX_OUTLINE_LINES = 40


def color_name(value: Any) -> str:
    """Return ``cyan`` for ``#00FFFF`` and friends, otherwise the raw value."""
    if not isinstance(value, str):
        return str(value)
    lowered = value.lower()
    if len(lowered) == 4 and lowered.startswith("#"):
        lowered = "#" + "".join(ch * 2 for ch in lowered[1:])
    return _INK_NAMES.get(lowered, value)


def _num(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:g}"
    return str(value)


def _paint(node: dict) -> str:
    parts = []
    fill, stroke = node.get("fill", "none"), node.get("stroke", "none")
    if fill not in (None, "none"):
        parts.append(f"{color_name(fill)} fill")
    if stroke not in (None, "none") and node.get("width", 0):
        parts.append(f"{color_name(stroke)} stroke {_num(node.get('width'))}")
    return ", ".join(parts) or "no paint"


def _transform_text(transform: Any) -> str:
    if not isinstance(transform, dict):
        return ""
    bits = []
    if transform.get("x", 0) or transform.get("y", 0):
        bits.append(f"at ({_num(transform.get('x', 0))}, {_num(transform.get('y', 0))})")
    if transform.get("scale", 1) != 1:
        bits.append(f"scale {_num(transform.get('scale'))}")
    if transform.get("rotate", 0):
        bits.append(f"rotated {_num(transform.get('rotate'))}°")
    return " " + ", ".join(bits) if bits else ""


def describe_node(node: Any) -> str:
    """One line for one top-level scene node."""
    if not isinstance(node, dict):
        return f"invalid node {node!r}"
    kind = node.get("type")
    if kind == "ellipse":
        rx, ry = node.get("rx"), node.get("ry")
        size = f"radius {_num(rx)}" if rx == ry else f"radii {_num(rx)}×{_num(ry)}"
        return f"ellipse at ({_num(node.get('cx'))}, {_num(node.get('cy'))}), {size}, {_paint(node)}"
    if kind in ("path", "polygon"):
        points = node.get("points") or []
        shape = "smooth path" if kind == "path" and node.get("smooth") else kind
        if kind == "path" and node.get("closed"):
            shape = "closed " + shape
        start = f" from ({_num(points[0][0])}, {_num(points[0][1])})" if points and isinstance(points[0], list) and len(points[0]) == 2 else ""
        return f"{shape} with {len(points)} points{start}, {_paint(node)}"
    if kind == "group":
        children = node.get("children") or []
        return f"group of {len(children)} shapes{_transform_text(node.get('transform'))}"
    if kind == "use":
        args = node.get("args") or {}
        arg_text = " with " + ", ".join(f"{k}={_num(v)}" for k, v in args.items()) if args else ""
        return f"motif '{node.get('motif')}'{arg_text}{_transform_text(node.get('transform'))}"
    return f"unknown node type {kind!r}"


def describe_scene(scene: Any) -> str:
    """Summarize what a scene draws: canvas, mark counts, inks, and an outline."""
    if not isinstance(scene, dict):
        return "Scene is not a JSON object."
    lines = []
    canvas = scene.get("canvas") or {}
    lines.append(
        f"Canvas: {_num(canvas.get('width'))}×{_num(canvas.get('height'))}, "
        f"background {color_name(canvas.get('background', '#ffffff'))}"
    )
    motifs = scene.get("motifs") or {}
    if motifs:
        lines.append("Motifs: " + ", ".join(sorted(motifs)))
    try:
        shapes = expand_scene(scene)
    except SceneValidationError:
        shapes = None
    if shapes is not None:
        kinds = Counter(shape["type"] for shape in shapes)
        inks: Counter = Counter()
        for shape in shapes:
            if shape["fill"] != "none":
                inks[color_name(shape["fill"])] += 1
            if shape["stroke"] != "none" and shape["width"] > 0:
                inks[color_name(shape["stroke"])] += 1
        lines.append(
            f"Marks: {len(shapes)} total ("
            + ", ".join(f"{count} {kind}" for kind, count in sorted(kinds.items())) + ")"
        )
        if inks:
            lines.append("Ink use: " + ", ".join(f"{name} {count}" for name, count in inks.most_common()))
    nodes = scene.get("nodes") or []
    lines.append(f"Top-level nodes ({len(nodes)}), in paint order:")
    for index, node in enumerate(nodes[:MAX_OUTLINE_LINES]):
        lines.append(f"  {index + 1:>3}. {describe_node(node)}")
    if len(nodes) > MAX_OUTLINE_LINES:
        lines.append(f"  ... and {len(nodes) - MAX_OUTLINE_LINES} more")
    return "\n".join(lines)


def describe_checks(checks: dict) -> str:
    """Summarize a checks dict (as returned by ``render_drawing``)."""
    hard = checks.get("hard_checks", {})
    metrics = checks.get("metrics", {})
    failed = [name for name, ok in hard.items() if not ok]
    lines = [
        ("PASS" if checks.get("valid") else "FAIL")
        + f": {len(hard) - len(failed)} of {len(hard)} rules passed"
    ]
    if failed:
        lines.append("Failed rules: " + ", ".join(failed))
    for violation in checks.get("violations", []):
        lines.append(f"  - {violation}")
    if metrics:
        picked = [
            ("marks", metrics.get("expanded_shapes")),
            ("inks used", metrics.get("shape_palette_color_count")),
            ("widest mark", metrics.get("max_shape_span")),
            ("thickest stroke", metrics.get("max_transformed_stroke_width")),
        ]
        lines.append("Stats: " + ", ".join(f"{label} {_num(value)}" for label, value in picked if value is not None))
    return "\n".join(lines)


def summarize_render(scene: Any, checks: dict) -> str:
    """The text block attached to every render result."""
    return describe_checks(checks) + "\n\n" + describe_scene(scene)


def describe_plan(composition: Any) -> list[str]:
    """Render a composition plan as Markdown bullets, falling back to JSON."""
    if not isinstance(composition, dict):
        return ["```json", json.dumps(composition, indent=2), "```"]
    lines = []
    for key, value in composition.items():
        label = str(key).replace("_", " ").capitalize()
        if isinstance(value, (str, int, float)):
            lines.append(f"- **{label}:** {value}")
        elif isinstance(value, list) and all(isinstance(item, (str, dict)) for item in value):
            lines.append(f"- **{label}:**")
            for item in value:
                if isinstance(item, str):
                    lines.append(f"  - {item}")
                    continue
                text = item.get("intent") or item.get("name") or json.dumps(item)
                span = item.get("range")
                if isinstance(span, list) and len(span) == 2:
                    text = f"nodes {span[0] + 1}–{span[1]}: {text}"
                lines.append(f"  - {text}")
        else:
            lines.append(f"- **{label}:** `{json.dumps(value)}`")
    return lines or ["(empty plan)"]


def write_run_report(run_dir: str | Path) -> Path:
    """Write ``report.md`` for a recording folder and return its path."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    out = [f"# Run: {manifest['run_id']}", ""]
    out.append(f"Started {manifest.get('created_at', '?')}, last updated {manifest.get('updated_at', '?')}.")
    final = manifest.get("final_step")
    out.append(f"{len(manifest['steps'])} steps recorded; "
               + (f"step {final + 1} is the accepted final." if final is not None else "no final yet."))
    out += ["", "## Prompts", ""]
    out += [f"{index + 1}. {prompt}" for index, prompt in enumerate(manifest["prompt_set"])]
    out += ["", "## Plan", ""] + describe_plan(manifest["composition"])
    for step in manifest["steps"]:
        out += ["", f"## Step {step['index'] + 1}: {step['label']}" + (" (final)" if step.get("final") else ""), ""]
        out.append(f"![{step['label']}]({step['png']})")
        out.append("")
        checks_path = run_dir / step["checks"]
        scene_path = run_dir / step["scene"]
        if checks_path.is_file():
            out += ["```text", describe_checks(json.loads(checks_path.read_text())), "```"]
        if step.get("model_output") and (run_dir / step["model_output"]).is_file():
            text = (run_dir / step["model_output"]).read_text()
            out += ["", "**Model output:**", "", "```text", text.rstrip(), "```"]
        if scene_path.is_file():
            out += ["", "**What was drawn:**", "", "```text", describe_scene(json.loads(scene_path.read_text())), "```"]
        out.append(f"\nFiles: [scene]({step['scene']}) · [svg]({step['svg']}) · [checks]({step['checks']})")
    path = run_dir / "report.md"
    path.write_text("\n".join(out) + "\n")
    return path


__all__ = [
    "color_name", "describe_checks", "describe_plan", "describe_node", "describe_scene",
    "summarize_render", "write_run_report",
]
