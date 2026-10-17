"""Authored drawings assembled from constrained CMYK marks."""

import json
from pathlib import Path

PAPER = "#FFFFFF"
INK = "#000000"
CORAL = "#FF00FF"
GOLD = "#FFFF00"
TEAL = "#00FFFF"


def _ellipse(cx, cy, rx, ry, fill, stroke="none", width=1, opacity=.5):
    return dict(type="ellipse", cx=cx, cy=cy, rx=rx, ry=ry,
                fill=fill, stroke=stroke, width=width, opacity=opacity)


def _path(points, stroke=INK, width=2, smooth=True):
    return dict(type="path", points=points, stroke=stroke, width=width,
                fill="none", smooth=smooth, closed=False, opacity=.5)


def _transform(x=0, y=0, scale=1, rotate=0):
    return dict(x=x, y=y, scale=scale, rotate=rotate)


def _group(children, x=0, y=0, scale=1, rotate=0):
    return dict(type="group", transform=_transform(x, y, scale, rotate), children=children)


def _use(name, args, x, y, scale=1, rotate=0):
    return dict(type="use", motif=name, args=dict(args),
                transform=_transform(x, y, scale, rotate))


def _scene(motifs, nodes):
    return {"version": 1, "canvas": {"width": 512, "height": 512, "background": PAPER},
            "motifs": motifs, "nodes": nodes}


def _cmyk_overprint():
    nodes = []
    colors = [TEAL, CORAL, GOLD]
    for row, y in enumerate((118, 196, 274, 352)):
        for column, x in enumerate((106, 184, 262, 340, 418)):
            nodes.append(_ellipse(x, y, 31, 31, colors[(row + column) % 3]))
            if column < 4:
                nodes.append(_ellipse(x + 31, y, 31, 31, colors[(row + column + 1) % 3]))
    for x, y in ((75, 80), (437, 80), (75, 390), (437, 390)):
        nodes.append(_path([[x - 20, y], [x + 20, y]], INK, 7, smooth=False))
        nodes.append(_path([[x, y - 20], [x, y + 20]], INK, 7, smooth=False))
    return {"id": "cmyk-overprint", "title": "CMYK overprint", "scene": _scene({}, nodes)}


def _cmyk_bloom():
    petal = {
        "params": {},
        "body": [
            _ellipse(0, -24, 13, 28, CORAL),
            _ellipse(0, -19, 9, 23, GOLD),
        ],
    }
    motifs = {"petal": petal}
    nodes = []
    for x, y, scale in ((142, 172, .85), (260, 262, 1), (376, 165, .78), (370, 370, .68), (151, 369, .72)):
        for angle in range(0, 360, 60):
            nodes.append(_use("petal", {}, x, y, scale, angle))
        nodes.append(_ellipse(x, y, 15 * scale, 15 * scale, TEAL))
        nodes.append(_ellipse(x + 5 * scale, y - 5 * scale, 8 * scale, 8 * scale, INK))
    return {"id": "cmyk-bloom", "title": "CMYK bloom", "scene": _scene(motifs, nodes)}


def _cmyk_currents():
    nodes = []
    colors = [TEAL, CORAL, GOLD, INK]
    for row, y in enumerate(range(94, 431, 48)):
        for column, x in enumerate(range(54, 455, 80)):
            color = colors[(row + column) % len(colors)]
            nodes.append(_path(
                [[x, y], [x + 18, y - 12], [x + 42, y + 12], [x + 70, y]],
                color, 7, smooth=True,
            ))
    for x, y, color in ((94, 70, GOLD), (414, 70, CORAL), (94, 454, TEAL), (414, 454, INK)):
        nodes.append(_ellipse(x, y, 20, 20, color))
    return {"id": "cmyk-currents", "title": "CMYK currents", "scene": _scene({}, nodes)}


def examples():
    """Return independent, editable CMYK small-mark compositions."""
    result = [_cmyk_overprint(), _cmyk_bloom(), _cmyk_currents()]
    tahoe_path = Path(__file__).parent.parent / "examples" / "26-tahoe-cmyk" / "scene.json"
    if tahoe_path.is_file():
        result.append({
            "id": "26-tahoe-cmyk",
            "title": "Tahoe CMYK overprint",
            "scene": json.loads(tahoe_path.read_text()),
        })
    return result
