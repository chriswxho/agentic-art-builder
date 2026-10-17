"""A graded set of drawing tasks, from a single dot to a full scene.

Each task gives an agent a prompt, the skills it exercises, any stricter
constraints on top of the CMYK policy, and plain-language success criteria a
person or VLM judge can score.  Every task also ships a reference solution
built here from the same primitives, so the dataset doubles as a set of worked
examples.  References are deterministic: the random generator is seeded.
"""

from __future__ import annotations

import math
import random

C, M, Y, K = "#00ffff", "#ff00ff", "#ffff00", "#000000"
WHITE = "#ffffff"


def _ellipse(cx, cy, rx, ry=None, fill=C):
    return {"type": "ellipse", "cx": round(cx, 2), "cy": round(cy, 2), "rx": round(rx, 2),
            "ry": round(rx if ry is None else ry, 2), "fill": fill, "stroke": "none",
            "width": 0, "opacity": 0.5}


def _poly(points, fill):
    return {"type": "polygon", "points": [[round(x, 2), round(y, 2)] for x, y in points],
            "fill": fill, "stroke": "none", "width": 0, "opacity": 0.5}


def _line(points, stroke=K, width=3, smooth=False):
    return {"type": "path", "points": [[round(x, 2), round(y, 2)] for x, y in points],
            "smooth": smooth, "closed": False, "fill": "none", "stroke": stroke,
            "width": width, "opacity": 0.5}


def _use(motif, x, y, scale=1, rotate=0, **args):
    return {"type": "use", "motif": motif, "args": args,
            "transform": {"x": round(x, 2), "y": round(y, 2), "scale": scale, "rotate": rotate}}


def _scene(width, height, nodes, motifs=None):
    return {"version": 1, "canvas": {"width": width, "height": height, "background": WHITE},
            "motifs": motifs or {}, "nodes": nodes}


def _rect(x, y, w, h, fill):
    return _poly([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], fill)


def _tiled_rect(x, y, w, h, fill, tile=60):
    """Cover a large rectangle with abutting tiles so no mark exceeds the span limit."""
    nodes = []
    cols, rows = math.ceil(w / tile), math.ceil(h / tile)
    for row in range(rows):
        for col in range(cols):
            tx, ty = x + col * w / cols, y + row * h / rows
            nodes.append(_rect(tx, ty, w / cols, h / rows, fill))
    return nodes


def _sky(width, top, bottom, bands, rng, cell=34):
    """Soft horizontal color bands from small overlapping ellipses."""
    nodes = []
    rows = max(1, round((bottom - top) / (cell * 0.75)))
    for row in range(rows):
        t = row / max(1, rows - 1)
        color = bands[min(len(bands) - 1, int(t * len(bands)))]
        y = top + (row + 0.5) * (bottom - top) / rows
        offset = (row % 2) * cell / 2
        x = -cell / 2 + offset
        while x < width + cell / 2:
            nodes.append(_ellipse(x + rng.uniform(-3, 3), y + rng.uniform(-2, 2), cell * 0.7, cell * 0.42, color))
            x += cell
    return nodes


# --- Level 1-3: single marks and overprinting ------------------------------

def _single_dot():
    return _scene(256, 256, [_ellipse(128, 128, 40, fill=C)])


def _two_ink_overprint():
    return _scene(256, 256, [_ellipse(108, 128, 44, fill=C), _ellipse(148, 128, 44, fill=M)])


def _three_ink_venn():
    nodes = []
    for angle, color in ((-90, C), (30, M), (150, Y)):
        rad = math.radians(angle)
        nodes.append(_ellipse(128 + 26 * math.cos(rad), 132 + 26 * math.sin(rad), 44, fill=color))
    return _scene(256, 256, nodes)


# --- Level 4: building tone by stacking ------------------------------------

def _tonal_ramp():
    nodes = []
    for column in range(5):
        for _ in range(column + 1):
            nodes.append(_rect(22 + column * 70, 66, 60, 60, K))
    return _scene(384, 192, nodes)


# --- Level 5: a simple object with a reusable motif ------------------------

def _flower():
    motifs = {"petal": {"params": {}, "body": [_ellipse(0, -30, 13, 26, M), _ellipse(0, -24, 7, 16, Y)]}}
    nodes = [
        _line([(128, 140), (125, 192)], C, 6), _line([(125, 192), (130, 244)], C, 6),
        _line([(128, 140), (125, 192)], Y, 6), _line([(125, 192), (130, 244)], Y, 6),
        _poly([(126, 200), (96, 184), (84, 198), (110, 210)], C),
        _poly([(126, 200), (96, 184), (84, 198), (110, 210)], Y),
        _poly([(128, 218), (160, 200), (172, 214), (146, 226)], C),
        _poly([(128, 218), (160, 200), (172, 214), (146, 226)], Y),
    ]
    nodes += [_use("petal", 128, 110, rotate=angle) for angle in range(0, 360, 45)]
    nodes += [_ellipse(128, 110, 14, fill=Y), _ellipse(128, 110, 9, fill=K)]
    return _scene(256, 256, nodes, motifs)


# --- Level 6: a composed icon scene ----------------------------------------

def _sunny_house():
    rng = random.Random(6)
    nodes = []
    # ground: green from cyan+yellow tiles
    nodes += _tiled_rect(0, 300, 384, 84, C, tile=64)
    nodes += _tiled_rect(0, 300, 384, 84, Y, tile=64)
    # sun with rays
    nodes += [_ellipse(310, 76, 30, fill=Y), _ellipse(310, 76, 30, fill=Y), _ellipse(310, 76, 18, fill=M)]
    for angle in range(0, 360, 30):
        rad = math.radians(angle)
        nodes.append(_line([(310 + 38 * math.cos(rad), 76 + 38 * math.sin(rad)),
                            (310 + 54 * math.cos(rad), 76 + 54 * math.sin(rad))], Y, 5))
    # walls (yellow) split into tiles, roof (magenta) as two triangles
    nodes += _tiled_rect(110, 200, 160, 110, Y, tile=80)
    nodes += [_poly([(100, 204), (190, 130), (190, 204)], M), _poly([(190, 130), (280, 204), (190, 204)], M),
              _poly([(100, 204), (190, 130), (190, 204)], M), _poly([(190, 130), (280, 204), (190, 204)], M)]
    # door, windows with crossbars
    nodes += [_rect(172, 248, 36, 62, K), _rect(172, 248, 36, 62, M)]
    for wx in (126, 226):
        nodes += [_rect(wx, 222, 32, 32, C), _line([(wx + 16, 222), (wx + 16, 254)], K, 3),
                  _line([(wx, 238), (wx + 32, 238)], K, 3)]
    # grass tufts
    for _ in range(14):
        x = rng.uniform(10, 374)
        y = rng.uniform(320, 370)
        nodes.append(_line([(x - 5, y + 6), (x, y - 6), (x + 5, y + 6)], C, 3))
    return _scene(384, 384, nodes)


# --- Level 7: organic form from many small marks ---------------------------

def _fruit_tree():
    rng = random.Random(7)
    motifs = {"leaves": {"params": {}, "body": [_ellipse(0, 0, 22, 18, C), _ellipse(4, -3, 18, 14, Y)]}}
    nodes = _tiled_rect(0, 330, 384, 54, Y, tile=64)
    # trunk and branches
    for ink in (K, M):
        nodes += [_poly([(180, 332), (204, 332), (201, 280), (183, 280)], ink),
                  _poly([(183, 282), (201, 282), (198, 230), (186, 230)], ink)]
    for points in ([(192, 250), (150, 200), (120, 180)], [(192, 240), (238, 196), (270, 178)],
                   [(192, 236), (190, 180)], [(190, 180), (196, 130)]):
        nodes.append(_line(points, K, 8, smooth=True))
    # canopy: overlapping leaf clusters inside an ellipse
    placed = 0
    while placed < 46:
        x, y = rng.uniform(80, 304), rng.uniform(70, 250)
        if ((x - 192) / 112) ** 2 + ((y - 160) / 90) ** 2 <= 1:
            nodes.append(_use("leaves", x, y, scale=rng.uniform(0.8, 1.2), rotate=rng.uniform(0, 360)))
            placed += 1
    # shadow side of the canopy
    shaded = 0
    while shaded < 14:
        x, y = rng.uniform(200, 290), rng.uniform(170, 240)
        if ((x - 192) / 95) ** 2 + ((y - 160) / 75) ** 2 <= 1:
            nodes.append(_ellipse(x, y, rng.uniform(8, 12), fill=C))
            shaded += 1
    # fruit: magenta+yellow = red, with a black highlight dot
    for _ in range(11):
        x, y = rng.uniform(110, 280), rng.uniform(100, 240)
        if ((x - 192) / 100) ** 2 + ((y - 165) / 80) ** 2 <= 1:
            nodes += [_ellipse(x, y, 8, fill=M), _ellipse(x, y, 8, fill=Y), _ellipse(x + 2, y - 2, 2.5, fill=K)]
    # fallen fruit and grass
    for x in (140, 252):
        nodes += [_ellipse(x, 346, 8, fill=M), _ellipse(x, 346, 8, fill=Y)]
    for _ in range(18):
        x = rng.uniform(5, 379)
        nodes.append(_line([(x, 384), (x + rng.uniform(-4, 4), 330 + rng.uniform(0, 30))], C, 3))
    return _scene(384, 384, nodes, motifs)


# --- Level 8: strict radial symmetry ---------------------------------------

def _mandala():
    motifs = {
        "spoke": {"params": {}, "body": [
            _ellipse(0, -60, 10, 26, C), _ellipse(0, -60, 6, 16, M),
            _ellipse(0, -112, 14, 14, Y), _ellipse(0, -112, 6, 6, K),
            _poly([(0, -140), (12, -164), (0, -186), (-12, -164)], M),
            _ellipse(0, -205, 6, 6, C),
        ]},
        "petal": {"params": {}, "body": [_ellipse(0, -150, 16, 34, C), _ellipse(0, -150, 10, 24, Y)]},
        "tick": {"params": {}, "body": [_line([(0, -226), (0, -240)], K, 4)]},
    }
    nodes = []
    for angle in range(0, 360, 30):
        nodes.append(_use("petal", 256, 256, rotate=angle + 15))
    for angle in range(0, 360, 30):
        nodes.append(_use("spoke", 256, 256, rotate=angle))
    for angle in range(0, 360, 10):
        nodes.append(_use("tick", 256, 256, rotate=angle))
    for angle in range(0, 360, 45):
        rad = math.radians(angle)
        nodes.append(_ellipse(256 + 30 * math.cos(rad), 256 + 30 * math.sin(rad), 14, fill=M))
    nodes += [_ellipse(256, 256, 22, fill=Y), _ellipse(256, 256, 12, fill=C), _ellipse(256, 256, 5, fill=K)]
    return _scene(512, 512, nodes, motifs)


# --- Level 9: landscape with depth and reflection ---------------------------

def _mountain_lake():
    rng = random.Random(9)
    nodes = _sky(512, 0, 280, [C, C, M, M, Y], rng)
    nodes += [_ellipse(330, 168, 26, fill=Y), _ellipse(330, 168, 26, fill=Y), _ellipse(330, 168, 16, fill=M)]
    # far range (cyan) then near range (cyan+black), peaks as narrow triangles
    for layer, (base, color, extra) in enumerate(((280, C, None), (290, C, K))):
        x = -20 + layer * 30
        while x < 512:
            w = rng.uniform(70, 90)
            h = rng.uniform(80, 92) if layer == 0 else rng.uniform(55, 80)
            peak = (x + w / 2 + rng.uniform(-10, 10), base - h)
            tri = [(x, base), peak, (x + w, base)]
            nodes.append(_poly(tri, color))
            if extra:
                nodes.append(_poly(tri, extra))
            x += w * 0.7
    # shoreline
    nodes += [_line([(x, 292), (x + 64, 292)], K, 6) for x in range(0, 512, 64)]
    # lake: cyan field with reflected sunset streaks
    nodes += _tiled_rect(0, 294, 512, 218, C, tile=90)
    for row in range(16):
        y = 302 + row * 13
        for _ in range(4):
            x = rng.uniform(250, 410) if row % 2 == 0 else rng.uniform(0, 500)
            color = Y if row % 2 == 0 else M
            nodes.append(_line([(x, y), (x + rng.uniform(20, 50), y)], color, 4))
    # foreground pines
    pine = {"params": {}, "body": [
        _line([(0, 0), (0, -64)], K, 4),
        _poly([(0, -70), (-18, -40), (18, -40)], K), _poly([(0, -54), (-22, -20), (22, -20)], K),
        _poly([(0, -36), (-26, 0), (26, 0)], K), _poly([(0, -36), (-26, 0), (26, 0)], C),
    ]}
    for x, s in ((30, 1.2), (66, 0.9), (452, 1.1), (488, 1.3)):
        nodes.append(_use("pine", x, 300, scale=s))
    return _scene(512, 512, nodes, {"pine": pine})


# --- Level 10: a busy narrative scene ---------------------------------------

def _harbor_sunset():
    rng = random.Random(10)
    nodes = _sky(512, 0, 250, [M, M, Y, Y], rng, cell=40)
    # sun and its glow
    nodes += [_ellipse(256, 226, 36, fill=Y), _ellipse(256, 226, 36, fill=Y), _ellipse(256, 226, 24, fill=M)]
    # distant headland
    for x in range(-10, 200, 40):
        nodes.append(_poly([(x, 252), (x + 30, 232 - rng.uniform(0, 10)), (x + 60, 252)], K))
    # sea: cyan+magenta (blue) tiles with wave strokes
    nodes += _tiled_rect(0, 250, 512, 262, C, tile=88)
    nodes += _tiled_rect(0, 250, 512, 262, M, tile=88)
    for row in range(10):
        y = 270 + row * 24
        for col in range(7):
            x = col * 78 + (row % 2) * 36 + rng.uniform(-6, 6)
            nodes.append(_line([(x, y), (x + 14, y - 5), (x + 28, y), (x + 42, y - 5)], C, 3, smooth=True))
    # sun reflection column
    for row in range(9):
        y = 258 + row * 12
        half = 30 - row * 2.5
        nodes.append(_line([(256 - half, y), (256 + half, y)], Y, 5))
    # sailboat motif: hull, mast, two sails, flag
    boat = {"params": {}, "body": [
        _poly([(-40, 0), (40, 0), (28, 16), (-28, 16)], K),
        _line([(0, 0), (0, -80)], K, 3),
        _poly([(3, -78), (3, -6), (44, -6)], Y),
        _poly([(-3, -64), (-3, -6), (-34, -6)], M),
        _poly([(0, -80), (14, -76), (0, -72)], M),
    ]}
    for x, y, s in ((110, 330, 1.0), (380, 300, 0.7), (440, 410, 1.15)):
        nodes.append(_use("boat", x, y, scale=s))
        for i in range(3):  # wake
            nodes.append(_line([(x - 46 * s - i * 14, y + 18 * s + i * 4), (x - 30 * s - i * 14, y + 18 * s + i * 4)], C, 3))
    # pier with posts and lamp
    for x in range(0, 192, 48):
        nodes.append(_rect(x, 450, 48, 12, K))
    for x in range(10, 192, 36):
        nodes.append(_rect(x, 462, 8, 50, K))
    nodes += [_line([(176, 450), (176, 396)], K, 4), _ellipse(176, 392, 10, fill=Y), _ellipse(176, 392, 6, fill=Y)]
    # gulls
    for _ in range(7):
        x, y = rng.uniform(60, 470), rng.uniform(40, 170)
        w = rng.uniform(8, 14)
        nodes.append(_line([(x - w, y), (x - w / 3, y - w / 2), (x, y), (x + w / 3, y - w / 2), (x + w, y)], K, 2.5, smooth=True))
    return _scene(512, 512, nodes, {"boat": boat})


_TASKS = [
    dict(id="01-single-dot", level=1, tier="beginner", title="Single dot", build=_single_dot,
         prompt="Draw one cyan circle in the center of the canvas.",
         skills=["place one primitive", "use the required opacity"],
         constraints={"max_shapes": 1},
         success_criteria=["Exactly one cyan circle", "It sits roughly in the middle of the canvas"]),
    dict(id="02-two-ink-overprint", level=2, tier="beginner", title="Two-ink overprint", build=_two_ink_overprint,
         prompt="Overlap a cyan circle and a magenta circle so that a blue-violet lens appears where they cross.",
         skills=["layering", "subtractive mixing of two inks"],
         constraints={"max_shapes": 2},
         success_criteria=["Two circles, one cyan and one magenta", "A visible darker overlap region between them"]),
    dict(id="03-three-ink-venn", level=3, tier="beginner", title="Three-ink Venn", build=_three_ink_venn,
         prompt="Make a three-circle Venn diagram in cyan, magenta and yellow so that green, red, blue and a dark center all appear.",
         skills=["three-way overprint", "symmetric placement"],
         constraints={"max_shapes": 3},
         success_criteria=["Three circles, one per CMY ink", "Pairwise overlaps show green, red and blue",
                           "A darker region where all three meet", "Roughly triangular, balanced arrangement"]),
    dict(id="04-tonal-ramp", level=4, tier="intermediate", title="Tonal ramp", build=_tonal_ramp,
         prompt="Make a row of five equal squares that go from light grey to near-black, using only black at 50% opacity.",
         skills=["building value by stacking identical marks", "even spacing"],
         constraints={"max_shapes": 20},
         success_criteria=["Five squares of equal size in a row", "Each square is darker than the one before",
                           "Only black ink is used"]),
    dict(id="05-flower", level=5, tier="intermediate", title="Flower", build=_flower,
         prompt="Draw a single flower with eight petals, a dark center, a green stem and two leaves.",
         skills=["defining and reusing a motif", "rotation", "mixing green from cyan and yellow"],
         constraints={"max_shapes": 40},
         success_criteria=["Eight evenly rotated petals", "A distinct center", "A stem below the bloom",
                           "Two leaves on the stem, reading as green"]),
    dict(id="06-sunny-house", level=6, tier="intermediate", title="Sunny house", build=_sunny_house,
         prompt="Draw a simple house on green grass under a sun: walls, pitched roof, a door, two windows, and a sun with rays.",
         skills=["composing several objects", "tiling large areas under the 96-unit span limit", "spatial layout"],
         constraints={"max_shapes": 120},
         success_criteria=["Recognizable house with roof, door and two windows", "Grass along the bottom",
                           "Sun with rays in the sky", "Objects do not awkwardly overlap"]),
    dict(id="07-fruit-tree", level=7, tier="advanced", title="Fruit tree", build=_fruit_tree,
         prompt="Draw a fruit tree: a trunk with branches, a full leafy canopy built from many small clusters, red fruit in the canopy and two fallen fruit on the ground.",
         skills=["organic texture from many small marks", "random but controlled placement", "light and shadow"],
         constraints={"max_shapes": 220},
         success_criteria=["Trunk and visible branches", "Canopy reads as a mass of leaves, not one blob",
                           "Shadowed side of the canopy is darker", "Red fruit inside the canopy and on the ground"]),
    dict(id="08-mandala", level=8, tier="advanced", title="Twelve-fold mandala", build=_mandala,
         prompt="Design a mandala with exact twelve-fold rotational symmetry: an inner ring, a ring of petals, a ring of spokes with ornaments, and a fine outer ring of ticks.",
         skills=["exact rotational symmetry", "multiple motifs", "nested rings"],
         constraints={"max_shapes": 200},
         success_criteria=["Twelve-fold symmetry holds for every ring", "At least four distinct concentric rings",
                           "Centerd on the canvas", "Uses all four inks deliberately"]),
    dict(id="09-mountain-lake", level=9, tier="expert", title="Mountain lake at dusk", build=_mountain_lake,
         prompt="Paint a mountain lake at dusk: a sky graded from blue at the top to warm at the horizon, a low sun, two mountain ranges for depth, a lake that reflects the sunset, and pine trees in the foreground.",
         skills=["color gradients from small marks", "atmospheric depth", "reflections", "working near the 512-mark budget"],
         constraints={},
         success_criteria=["Sky gradient from cool to warm without visible gaps",
                           "Far range lighter than near range", "Lake reflects sunset colors",
                           "Foreground pines frame the scene", "Clear horizon line"]),
    dict(id="10-harbor-sunset", level=10, tier="expert", title="Harbor at sunset", build=_harbor_sunset,
         prompt="Paint a harbor at sunset: warm sky, setting sun with a reflection on the water, a dark headland, three sailboats at different distances with wakes, a wooden pier with a lit lamp, and gulls overhead.",
         skills=["many interacting objects", "scale for distance", "reflections", "narrative composition",
                 "budgeting marks across the whole image"],
         constraints={},
         success_criteria=["Sun with a vertical reflection on the water", "Three sailboats of different sizes suggesting depth",
                           "Each boat has hull, mast and sails", "Pier with posts and a lit lamp",
                           "Gulls in the sky", "Sea shows wave texture", "Composition is balanced, not cluttered"]),
]


def tasks() -> list[dict]:
    """Return every task with its reference scene, easiest first."""
    out = []
    for task in _TASKS:
        entry = {key: value for key, value in task.items() if key != "build"}
        entry["reference_scene"] = task["build"]()
        out.append(entry)
    return out


def task_card(task: dict) -> dict:
    """The part of a task an agent is given: everything but the reference answer."""
    return {key: value for key, value in task.items() if key != "reference_scene"}


def export_dataset(folder) -> list[dict]:
    """Write the dataset to ``folder``: one subfolder per task plus an index.

    Each task folder holds ``task.json`` (the card an agent sees),
    ``reference/`` (scene, SVG, PNG, checks, readable summary), and the
    folder root gets ``tasks.jsonl``, ``index.json`` and a ``README.md``.
    """
    import json
    from dataclasses import asdict
    from pathlib import Path

    from .policy import composition_constraints
    from .renderer import check_scene, render, to_svg
    from .report import summarize_render

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    index = []
    for task in tasks():
        scene = task["reference_scene"]
        checks = check_scene(scene, composition_constraints())
        if checks.valid and task["constraints"]:
            extra = check_scene(scene, task["constraints"])
            checks.hard_checks.update(extra.hard_checks)
            checks.violations.extend(extra.violations)
            checks.valid = extra.valid
        if not checks.valid:
            raise ValueError(f"{task['id']} reference fails: {'; '.join(checks.violations)}")
        task_dir = folder / task["id"]
        ref = task_dir / "reference"
        ref.mkdir(parents=True, exist_ok=True)
        card = task_card(task)
        (task_dir / "task.json").write_text(json.dumps(card, indent=2) + "\n")
        (ref / "scene.json").write_text(json.dumps(scene, indent=2) + "\n")
        (ref / "drawing.svg").write_text(to_svg(scene))
        render(scene).save(ref / "drawing.png")
        checks_dict = asdict(checks)
        (ref / "checks.json").write_text(json.dumps(checks_dict, indent=2) + "\n")
        (ref / "summary.txt").write_text(summarize_render(scene, checks_dict) + "\n")
        index.append({**card, "reference_marks": checks.metrics["expanded_shapes"],
                      "path": task["id"]})
    (folder / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    (folder / "tasks.jsonl").write_text("".join(json.dumps(task_card(t)) + "\n" for t in tasks()))
    lines = [
        "# Drawing task dataset",
        "",
        f"{len(index)} drawing tasks, easiest first. Every task must follow the CMYK rules "
        "(white canvas, cyan/magenta/yellow/black only, 50% opacity, marks at most 96 units "
        "across, at most 512 marks), plus any extra constraints listed for it.",
        "",
        "Give an agent `task.json` (or a line of `tasks.jsonl`). Score its drawing with the "
        "renderer's checks for the hard rules and the success criteria for the rest. "
        "`reference/` holds one passing solution for comparison, not the only right answer.",
        "",
        "Regenerate with `python -m drawing_sandbox dataset --out dataset`.",
        "",
        "| Level | Task | Tier | Extra limits | Reference marks | Reference |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for entry in index:
        limits = ", ".join(f"{k} {v}" for k, v in entry["constraints"].items()) or "none"
        lines.append(
            f"| {entry['level']} | [{entry['title']}]({entry['path']}/task.json) | {entry['tier']} | "
            f"{limits} | {entry['reference_marks']} | "
            f"<img src=\"{entry['path']}/reference/drawing.png\" width=\"120\"> |"
        )
    lines.append("")
    for entry in index:
        lines += [f"## {entry['level']}. {entry['title']}", "", f"> {entry['prompt']}", "",
                  "Skills: " + ", ".join(entry["skills"]), "", "Success criteria:"]
        lines += [f"- {item}" for item in entry["success_criteria"]]
        lines.append("")
    (folder / "README.md").write_text("\n".join(lines))
    return index


__all__ = ["export_dataset", "task_card", "tasks"]
