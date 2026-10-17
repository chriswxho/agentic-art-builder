#!/usr/bin/env python3
"""Build and record a Lake Tahoe study from constrained CMYK marks."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from drawing_sandbox.recording import record_composition_step
from drawing_sandbox.server import render_drawing


CYAN = "#00ffff"
MAGENTA = "#ff00ff"
YELLOW = "#ffff00"
BLACK = "#000000"


def path(points, color, width, *, smooth=False):
    return {
        "type": "path", "points": points, "smooth": smooth, "closed": False,
        "fill": "none", "stroke": color, "width": width, "opacity": 0.5,
    }


def ellipse(cx, cy, rx, ry, color):
    return {
        "type": "ellipse", "cx": cx, "cy": cy, "rx": rx, "ry": ry,
        "fill": color, "stroke": "none", "width": 0, "opacity": 0.5,
    }


def polygon(points, color):
    return {
        "type": "polygon", "points": points,
        "fill": color, "stroke": "none", "width": 0, "opacity": 0.5,
    }


def build_scene():
    nodes = []
    stages = []

    # Airy cyan/magenta hatching creates the blue sky without a large fill.
    for row in range(8):
        y = 25 + row * 31
        for column in range(12):
            x = -18 + column * 82
            drift = ((row + column) % 3 - 1) * 2
            nodes.append(path([[x, y], [x + 78, y + drift]], CYAN, 10, smooth=True))
            if row < 5 and column % 3 == row % 3:
                nodes.append(path([[x + 8, y + 5], [x + 72, y + 4 - drift]], MAGENTA, 6, smooth=True))
    stages.append(("Layered sky", len(nodes)))

    # The snowy ridge is an open network of short marks; white paper remains snow.
    ridge = [
        [0, 292], [40, 273], [80, 278], [120, 259], [160, 279],
        [200, 263], [240, 284], [280, 270], [320, 291], [360, 272],
        [400, 288], [440, 266], [480, 278], [520, 260], [560, 275],
        [600, 250], [640, 270], [680, 258], [720, 279], [760, 263],
        [800, 276], [840, 258], [880, 270], [920, 252], [960, 266],
    ]
    for index, (left, right) in enumerate(zip(ridge, ridge[1:])):
        nodes.append(path([left, right], BLACK, 7))
        nodes.append(path([[left[0], left[1] + 12], [right[0], right[1] + 12]], CYAN, 10))
        if index % 2 == 0:
            nodes.append(path(
                [[left[0] + 10, left[1] + 18], [right[0] - 5, right[1] + 22]],
                MAGENTA, 5,
            ))
        if index % 3 == 1:
            midpoint = [(left[0] + right[0]) / 2, (left[1] + right[1]) / 2]
            nodes.append(path([midpoint, [midpoint[0] + 14, 305]], BLACK, 4))
    stages.append(("Snowy mountain ridge", len(nodes)))

    # Water is a field of short overprinted currents, not a single background shape.
    for row in range(8):
        y = 316 + row * 28
        for column in range(12):
            x = -20 + column * 83
            wave = ((row * 2 + column) % 5 - 2) * 2
            nodes.append(path([[x, y], [x + 38, y + wave], [x + 78, y - wave]], CYAN, 10, smooth=True))
            if row >= 3 and column % 3 == (row + 1) % 3:
                nodes.append(path([[x + 8, y + 8], [x + 68, y + 5]], YELLOW, 7, smooth=True))
            if (row + column) % 4 == 0:
                nodes.append(path([[x + 15, y + 13], [x + 66, y + 11]], BLACK, 5, smooth=True))
    stages.append(("Turquoise water currents", len(nodes)))

    # Right shoreline: cyan + yellow overprints read as the tree-covered green bank.
    for x, y, rx, ry in (
        (770, 307, 38, 13), (818, 300, 42, 16), (865, 294, 44, 18),
        (910, 300, 42, 17), (951, 308, 37, 14),
    ):
        nodes.append(ellipse(x, y, rx, ry, CYAN))
        nodes.append(ellipse(x + 3, y - 2, rx - 5, ry - 3, YELLOW))
        nodes.append(ellipse(x - 4, y + 7, rx - 8, max(5, ry - 7), BLACK))

    # Distant and central granite are built from overlapping small ellipses.
    for x, y, rx, ry in (
        (112, 318, 14, 6), (154, 316, 18, 8), (255, 317, 22, 9),
        (372, 326, 25, 13), (404, 316, 29, 18), (438, 320, 37, 24),
        (476, 312, 42, 29), (516, 327, 31, 21), (567, 338, 34, 20),
        (618, 347, 27, 15), (706, 358, 25, 14), (776, 365, 27, 15),
        (872, 359, 22, 10), (915, 350, 17, 8),
    ):
        nodes.append(ellipse(x, y, rx, ry, BLACK))
        nodes.append(ellipse(x - 4, y - 4, max(6, rx - 7), max(4, ry - 5), CYAN))
        if rx >= 25:
            nodes.append(ellipse(x - 8, y - 7, max(5, rx - 16), max(3, ry - 12), MAGENTA))

    # Upright center boulder gets extra stacked marks and a dark water reflection.
    nodes.extend([
        ellipse(552, 318, 42, 43, BLACK),
        ellipse(545, 305, 35, 35, CYAN),
        ellipse(535, 295, 22, 18, MAGENTA),
        ellipse(557, 345, 45, 10, BLACK),
        path([[514, 352], [552, 347], [593, 353]], CYAN, 8, smooth=True),
    ])
    stages.append(("Shore and granite clusters", len(nodes)))

    # Two pines use short trunks, branch strokes, and cyan/yellow needle triangles.
    for x, base, height, tiers in ((850, 306, 82, 4), (897, 309, 61, 3)):
        nodes.append(path([[x, base], [x, base - height]], BLACK, 6))
        for tier in range(tiers):
            y = base - height + 12 + tier * 16
            half = 17 + tier * 6
            triangle = [[x, y - 12], [x - half, y + 13], [x + half, y + 13]]
            nodes.append(polygon(triangle, CYAN))
            nodes.append(polygon([[px + 3, py + 2] for px, py in triangle], YELLOW))
            nodes.append(path([[x - half + 4, y + 9], [x + half - 4, y + 9]], BLACK, 3))
    stages.append(("Shoreline pines", len(nodes)))

    # Foreground left boulder: an arched cluster with cyan and yellow highlights.
    for x, y, rx, ry in (
        (226, 422, 42, 27), (264, 407, 44, 33), (304, 414, 43, 29),
        (281, 430, 45, 24),
    ):
        nodes.append(ellipse(x, y, rx, ry, BLACK))
        nodes.append(ellipse(x - 7, y - 8, rx - 10, max(7, ry - 12), CYAN))
    nodes.extend([
        ellipse(252, 395, 25, 12, MAGENTA),
        path([[215, 424], [255, 398], [301, 409]], BLACK, 7, smooth=True),
        path([[224, 413], [257, 394], [289, 401]], CYAN, 6, smooth=True),
        ellipse(483, 466, 32, 15, BLACK),
        ellipse(477, 461, 23, 9, MAGENTA),
    ])

    # The huge near boulder is a mosaic of legal marks rather than one large shape.
    foreground = [
        (596, 527, 44, 32), (632, 507, 45, 35), (671, 488, 45, 35),
        (712, 471, 45, 34), (755, 459, 45, 31), (798, 451, 45, 29),
        (842, 448, 45, 28), (886, 451, 45, 29), (928, 463, 45, 32),
        (963, 484, 43, 35), (675, 525, 45, 34), (719, 516, 45, 34),
        (765, 510, 45, 34), (811, 508, 45, 34), (857, 511, 45, 34),
        (903, 520, 45, 34),
    ]
    for x, y, rx, ry in foreground:
        nodes.append(ellipse(x, y, rx, ry, BLACK))
    for x, y in ((623, 497), (670, 476), (720, 458), (772, 447), (824, 441), (876, 444)):
        nodes.append(ellipse(x, y, 34, 10, CYAN))
    for x, y in ((644, 489), (705, 461), (770, 444), (835, 440)):
        nodes.append(ellipse(x, y, 24, 7, MAGENTA))
    for x, y in ((700, 520), (762, 493), (824, 486), (887, 501)):
        nodes.append(ellipse(x, y, 33, 18, BLACK))
    stages.append(("Foreground granite and final accents", len(nodes)))

    scene = {
        "version": 1,
        "title": "Tahoe CMYK overprint study",
        "source": "examples/26-tahoe-day/reference.jpg",
        "canvas": {"width": 960, "height": 540, "background": "#ffffff"},
        "motifs": {},
        "nodes": nodes,
    }
    return scene, stages


PROMPT_SET = [
    "Remake the supplied Lake Tahoe photograph as a layered vector print.",
    "Use only small cyan, magenta, yellow, and black marks at 50% opacity on white; create color and form through overprinting.",
    "Preserve the wide blue sky, snowy ridge, turquoise water, central granite cluster, right-shore pines, and large foreground boulder.",
]


def composition(stages):
    return {
        "title": "Tahoe CMYK overprint study",
        "source_image": "examples/26-tahoe-day/reference.jpg",
        "canvas": [960, 540],
        "strategy": "short hatching, overlapping ellipses, small polygons, and preserved white paper",
        "layers": [{"label": label, "cumulative_node_count": count} for label, count in stages],
    }


def write_render(output: Path, scene: dict, rendered: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "scene.json").write_text(json.dumps(scene, indent=2, allow_nan=False) + "\n")
    (output / "drawing.svg").write_text(rendered["svg"])
    png = base64.b64decode(rendered["png_data_url"].split(",", 1)[1])
    (output / "drawing.png").write_bytes(png)
    (output / "checks.json").write_text(json.dumps(rendered["checks"], indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("examples/26-tahoe-cmyk"))
    parser.add_argument("--record-root", type=Path, default=Path("recordings"))
    parser.add_argument("--run-id", default="tahoe-cmyk-remake")
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()

    scene, stages = build_scene()
    plan = composition(stages)
    final_render = None
    for index, (label, count) in enumerate(stages):
        candidate = {**scene, "nodes": scene["nodes"][:count]}
        rendered = render_drawing(candidate)
        if not rendered["checks"]["valid"]:
            raise ValueError("; ".join(rendered["checks"]["violations"]))
        is_final = args.final and index == len(stages) - 1
        record_composition_step(
            candidate, rendered,
            run_id=args.run_id,
            prompt_set=PROMPT_SET,
            composition=plan,
            step_label=label,
            final=is_final,
            root=args.record_root,
        )
        final_render = rendered
    write_render(args.out, scene, final_render)
    print(json.dumps({
        "output": str(args.out.resolve()),
        "run_id": args.run_id,
        "stages": len(stages),
        "nodes": len(scene["nodes"]),
        "final": args.final,
        "metrics": final_render["checks"]["metrics"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
