"""A small, bounded vector drawing language and its shared geometry pipeline.

Scene version 1 uses ``canvas``, optional ``motifs``, and ``nodes``.  Root and
motif metadata is ignored so descriptions can travel with a scene.
Node attributes are deliberately strict.  Coordinates are in canvas units.

``path.points`` is a polyline.  ``smooth=True`` interpolates its points with
Catmull--Rom splines, converted to cubic Beziers; open endpoints are clamped.
``closed=True`` closes the path (and makes the spline periodic).  ``polygon``
is always closed.  A group or motif use may specify translation ``x, y``, a
uniform ``scale``, and ``rotate`` in degrees.  Composition is T * R * S.

Motif parameters are numeric.  Only an entire string such as ``"$radius"``
substitutes a parameter: expressions and arbitrary Python are never evaluated.
``expand_scene`` returns primitives with normalized defaults and a retained SVG
affine ``matrix`` [a,b,c,d,e,f].  Both renderers consume that same expansion.

Validation checks finite structure, references, and resource bounds.  It does
not certify that the drawing matches a prompt, is attractive, or is novel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any

import numpy as np
from PIL import Image, ImageColor, ImageDraw


SCENE_VERSION = 1
MIN_CANVAS_SIZE = 64
MAX_CANVAS_SIZE = 1024
MAX_EXPANDED_SHAPES = 2048
MAX_NODES = 8192
MAX_DEPTH = 16
MAX_POINTS = 256
MAX_MOTIFS = 128
MAX_PARAMETERS = 32
MAX_COORDINATE = 65_536.0
MAX_STROKE_WIDTH = 256.0
MAX_TRANSFORMED_STROKE_WIDTH = 4096.0
SUPERSAMPLE = 3

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}\Z")
_COLOR = re.compile(r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})\Z")
_IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
_STYLE_KEYS = {"fill", "stroke", "width", "opacity"}
_NODE_KEYS = {
    "path": {"type", "points", "smooth", "closed"} | _STYLE_KEYS,
    "polygon": {"type", "points"} | _STYLE_KEYS,
    "ellipse": {"type", "cx", "cy", "rx", "ry"} | _STYLE_KEYS,
    "group": {"type", "transform", "children"},
    "use": {"type", "motif", "args", "transform"},
}


@dataclass
class Evaluation:
    valid: bool
    hard_checks: dict[str, bool] = field(default_factory=dict)
    metrics: dict[str, float | int] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)


class SceneValidationError(ValueError):
    """An invalid scene, with a stable category and human-readable location."""

    def __init__(self, message: str, code: str = "scene_structure") -> None:
        super().__init__(message)
        self.code = code


def _fail(path: str, message: str, code: str = "scene_structure") -> None:
    raise SceneValidationError(f"{path}: {message}", code)


def _number(value: Any, env: dict[str, float], path: str) -> float:
    if isinstance(value, str):
        if not value.startswith("$") or not _NAME.fullmatch(value[1:]):
            _fail(path, "expected a number or an entire '$parameter' string", "parameters")
        name = value[1:]
        if name not in env:
            _fail(path, f"unknown parameter '{name}'", "parameters")
        value = env[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, "expected a finite JSON number", "finite_geometry")
    try:
        result = float(value)
    except (ValueError, OverflowError):
        _fail(path, "number cannot be represented as finite geometry", "finite_geometry")
    if not math.isfinite(result):
        _fail(path, "number must be finite", "finite_geometry")
    if abs(result) > MAX_COORDINATE:
        _fail(path, f"numeric magnitude exceeds {MAX_COORDINATE:g}", "reasonable_bounds")
    return result


def _color(value: Any, path: str, allow_none: bool = True) -> str:
    if value == "none" and allow_none:
        return "none"
    if not isinstance(value, str) or not _COLOR.fullmatch(value):
        _fail(path, "expected #RGB, #RRGGBB" + (", or 'none'" if allow_none else ""), "colors")
    # Normalize shorthand as well as case for stable expansion comparisons.
    digits = value[1:].lower()
    if len(digits) == 3:
        digits = "".join(char * 2 for char in digits)
    return "#" + digits


def _compose(
    parent: tuple[float, ...] | list[float], child: tuple[float, ...] | list[float]
) -> tuple[float, ...]:
    a, b, c, d, e, f = parent
    g, h, i, j, k, l = child
    result = (
        a * g + c * h, b * g + d * h,
        a * i + c * j, b * i + d * j,
        a * k + c * l + e, b * k + d * l + f,
    )
    if not all(math.isfinite(number) for number in result):
        _fail("transform", "composed transform is not finite", "finite_geometry")
    return result


def _transform_point(point: list[float] | tuple[float, ...], matrix: tuple[float, ...] | list[float]) -> tuple[float, float]:
    a, b, c, d, e, f = matrix
    x, y = point
    return a * x + c * y + e, b * x + d * y + f


def _transform(value: Any, env: dict[str, float], path: str) -> tuple[float, ...]:
    if not isinstance(value, dict):
        _fail(path, "transform must be an object")
    unknown = set(value) - {"x", "y", "scale", "rotate"}
    if unknown:
        _fail(path, f"unknown transform attributes: {sorted(map(str, unknown))}")
    x = _number(value.get("x", 0), env, path + ".x")
    y = _number(value.get("y", 0), env, path + ".y")
    scale = _number(value.get("scale", 1), env, path + ".scale")
    angle = _number(value.get("rotate", 0), env, path + ".rotate")
    if not 0.0001 <= abs(scale) <= 64:
        _fail(path + ".scale", "absolute scale must be between 0.0001 and 64", "reasonable_bounds")
    radians = math.radians(angle % 360)
    cosine, sine = math.cos(radians), math.sin(radians)
    return scale * cosine, scale * sine, -scale * sine, scale * cosine, x, y


def _path_segments(primitive: dict[str, Any]) -> list[tuple[Any, ...]]:
    """Canonical local line/cubic segments, shared by SVG and raster output."""
    points = primitive["points"]
    closed = primitive["type"] == "polygon" or primitive.get("closed", False)
    smooth = primitive.get("smooth", False)
    segment_count = len(points) if closed else len(points) - 1
    result: list[tuple[Any, ...]] = []
    for index in range(segment_count):
        p1 = points[index]
        p2 = points[(index + 1) % len(points)]
        if not smooth:
            result.append(("L", p2))
            continue
        p0 = points[(index - 1) % len(points)] if closed or index else p1
        p3 = points[(index + 2) % len(points)] if closed or index + 2 < len(points) else p2
        c1 = [p1[axis] + (p2[axis] - p0[axis]) / 6 for axis in (0, 1)]
        c2 = [p2[axis] - (p3[axis] - p1[axis]) / 6 for axis in (0, 1)]
        result.append(("C", c1, c2, p2))
    return result


class _Compiler:
    def __init__(self, scene: Any) -> None:
        self.scene = scene
        self.motifs: dict[str, dict[str, Any]] = {}
        self.defaults: dict[str, dict[str, float]] = {}
        self.references: dict[str, set[str]] = {}
        self.raw_nodes = 0
        self.expanded_nodes = 0
        self.shapes: list[dict[str, Any]] = []

    def compile(self) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int]]:
        if not isinstance(self.scene, dict):
            _fail("scene", "must be an object")
        if isinstance(self.scene.get("version"), bool) or self.scene.get("version") != SCENE_VERSION:
            _fail("scene.version", "must equal 1")
        source_canvas = self.scene.get("canvas")
        if not isinstance(source_canvas, dict):
            _fail("scene.canvas", "must be an object", "canvas_size")
        canvas: dict[str, Any] = {}
        for key in ("width", "height"):
            value = source_canvas.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or not MIN_CANVAS_SIZE <= value <= MAX_CANVAS_SIZE:
                _fail("scene.canvas." + key, "must be an integer from 64 to 1024", "canvas_size")
            canvas[key] = value
        canvas["background"] = _color(source_canvas.get("background", "#ffffff"), "scene.canvas.background", False)

        motif_source = self.scene.get("motifs", {})
        if not isinstance(motif_source, dict) or len(motif_source) > MAX_MOTIFS:
            _fail("scene.motifs", f"must be an object with at most {MAX_MOTIFS} motifs", "bounded_expansion")
        for name, definition in motif_source.items():
            if not isinstance(name, str) or not _NAME.fullmatch(name):
                _fail("scene.motifs", "motif names must be identifiers of at most 64 characters")
            if not isinstance(definition, dict):
                _fail(f"motifs.{name}", "definition must be an object")
            params = definition.get("params", {})
            if not isinstance(params, dict) or len(params) > MAX_PARAMETERS:
                _fail(f"motifs.{name}.params", f"must contain at most {MAX_PARAMETERS} numeric parameters", "parameters")
            defaults: dict[str, float] = {}
            for key, value in params.items():
                if not isinstance(key, str) or not _NAME.fullmatch(key):
                    _fail(f"motifs.{name}.params", "parameter names must be identifiers", "parameters")
                # Defaults cannot reference another parameter: no ordering ambiguity.
                defaults[key] = _number(value, {}, f"motifs.{name}.params.{key}")
            self.motifs[name] = definition
            self.defaults[name] = defaults
            self.references[name] = set()

        nodes = self.scene.get("nodes")
        self._scan(nodes, {}, "scene.nodes", 1, None)
        for name, definition in self.motifs.items():
            self._scan(definition.get("body"), self.defaults[name], f"motifs.{name}.body", 1, name)
        self._check_cycles()
        self._expand(nodes, {}, _IDENTITY, "scene.nodes", 1)
        metrics = {
            "raw_nodes": self.raw_nodes,
            "expanded_nodes": self.expanded_nodes,
            "expanded_shapes": len(self.shapes),
            "primitive_count": len(self.shapes),
            "motif_count": len(self.motifs),
            "path_points": sum(len(shape.get("points", [])) for shape in self.shapes),
            "canvas_width": canvas["width"],
            "canvas_height": canvas["height"],
        }
        return canvas, self.shapes, metrics

    def _read(self, node: Any, env: dict[str, float], path: str) -> dict[str, Any]:
        if not isinstance(node, dict):
            _fail(path, "node must be an object")
        kind = node.get("type")
        if not isinstance(kind, str) or kind not in _NODE_KEYS:
            _fail(path + ".type", "expected path, ellipse, polygon, group, or use")
        unknown = set(node) - _NODE_KEYS[kind]
        if unknown:
            _fail(path, f"unknown node attributes: {sorted(map(str, unknown))}")
        result: dict[str, Any] = {"type": kind}
        if kind in ("group", "use"):
            result["matrix"] = _transform(node.get("transform", {}), env, path + ".transform")
            if kind == "group":
                result["children"] = node.get("children")
            else:
                name = node.get("motif")
                if not isinstance(name, str) or name not in self.motifs:
                    _fail(path + ".motif", f"unknown motif {name!r}", "motif_references")
                args = node.get("args", {})
                if not isinstance(args, dict):
                    _fail(path + ".args", "must be an object", "parameters")
                unknown_args = set(args) - set(self.defaults[name])
                if unknown_args:
                    _fail(path + ".args", f"unknown parameters: {sorted(map(str, unknown_args))}", "parameters")
                result["motif"] = name
                result["args"] = dict(self.defaults[name])
                for key, value in args.items():
                    result["args"][key] = _number(value, env, path + ".args." + key)
            return result

        result["fill"] = _color(node.get("fill", "none"), path + ".fill")
        result["stroke"] = _color(node.get("stroke", "#1f2937"), path + ".stroke")
        result["width"] = _number(node.get("width", 3), env, path + ".width")
        result["opacity"] = _number(node.get("opacity", 1), env, path + ".opacity")
        if not 0 <= result["width"] <= MAX_STROKE_WIDTH:
            _fail(path + ".width", "must be between 0 and 256", "reasonable_bounds")
        if not 0 <= result["opacity"] <= 1:
            _fail(path + ".opacity", "must be between 0 and 1", "reasonable_bounds")

        if kind == "ellipse":
            for key in ("cx", "cy", "rx", "ry"):
                result[key] = _number(node.get(key), env, path + "." + key)
            if result["rx"] <= 0 or result["ry"] <= 0:
                _fail(path, "ellipse radii must be positive", "reasonable_bounds")
        else:
            closed = kind == "polygon" or node.get("closed", False)
            smooth = node.get("smooth", False)
            if not isinstance(closed, bool) or not isinstance(smooth, bool):
                _fail(path, "closed and smooth must be booleans")
            points = node.get("points")
            minimum = 3 if closed else 2
            if not isinstance(points, list) or not minimum <= len(points) <= MAX_POINTS:
                _fail(path + ".points", f"must contain {minimum} to {MAX_POINTS} [x,y] points", "bounded_expansion")
            normalized_points = []
            for index, point in enumerate(points):
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    _fail(f"{path}.points[{index}]", "must be an [x,y] pair")
                normalized_points.append([
                    _number(value, env, f"{path}.points[{index}][{axis}]")
                    for axis, value in enumerate(point)
                ])
            result["points"] = normalized_points
            if kind == "path":
                result["closed"] = closed
                result["smooth"] = smooth
        return result

    def _scan(self, nodes: Any, env: dict[str, float], path: str, depth: int, owner: str | None) -> None:
        if not isinstance(nodes, list):
            _fail(path, "must be a list")
        if depth > MAX_DEPTH:
            _fail(path, f"node nesting exceeds {MAX_DEPTH}", "bounded_expansion")
        if self.raw_nodes + len(nodes) > MAX_NODES:
            _fail(path, f"raw node count exceeds {MAX_NODES}", "bounded_expansion")
        for index, source in enumerate(nodes):
            self.raw_nodes += 1
            if self.raw_nodes > MAX_NODES:
                _fail(path, f"raw node count exceeds {MAX_NODES}", "bounded_expansion")
            here = f"{path}[{index}]"
            node = self._read(source, env, here)
            if node["type"] == "group":
                self._scan(node["children"], env, here + ".children", depth + 1, owner)
            elif node["type"] == "use" and owner is not None:
                self.references[owner].add(node["motif"])

    def _check_cycles(self) -> None:
        visiting: set[str] = set()
        complete: set[str] = set()

        def visit(name: str, chain: list[str]) -> None:
            if name in visiting:
                _fail("scene.motifs", "cyclic motif reference: " + " -> ".join(chain + [name]), "motif_references")
            if name in complete:
                return
            if len(chain) >= MAX_DEPTH:
                _fail("scene.motifs", f"motif dependency depth exceeds {MAX_DEPTH}", "bounded_expansion")
            visiting.add(name)
            for child in sorted(self.references[name]):
                visit(child, chain + [name])
            visiting.remove(name)
            complete.add(name)

        for name in self.motifs:
            visit(name, [])

    def _expand(self, nodes: list[Any], env: dict[str, float], matrix: tuple[float, ...], path: str, depth: int) -> None:
        if depth > MAX_DEPTH:
            _fail(path, f"expanded nesting exceeds {MAX_DEPTH}", "bounded_expansion")
        for index, source in enumerate(nodes):
            self.expanded_nodes += 1
            if self.expanded_nodes > MAX_NODES:
                _fail(path, f"expanded node count exceeds {MAX_NODES}", "bounded_expansion")
            here = f"{path}[{index}]"
            node = self._read(source, env, here)
            if node["type"] == "group":
                self._expand(node["children"], env, _compose(matrix, node["matrix"]), here + ".children", depth + 1)
            elif node["type"] == "use":
                name = node["motif"]
                self._expand(self.motifs[name]["body"], node["args"], _compose(matrix, node["matrix"]), here + f".use({name})", depth + 1)
            else:
                if len(self.shapes) >= MAX_EXPANDED_SHAPES:
                    _fail(here, f"expanded shape count exceeds {MAX_EXPANDED_SHAPES}", "bounded_expansion")
                node["matrix"] = list(matrix)
                _check_geometry(node, here)
                self.shapes.append(node)


def _check_geometry(shape: dict[str, Any], path: str) -> None:
    matrix = shape["matrix"]
    scale = math.hypot(matrix[0], matrix[1])
    if not math.isfinite(scale) or shape["width"] * scale > MAX_TRANSFORMED_STROKE_WIDTH:
        _fail(path, "transformed stroke is too wide", "reasonable_bounds")
    if shape["type"] == "ellipse":
        cx, cy = _transform_point((shape["cx"], shape["cy"]), matrix)
        ex = math.hypot(matrix[0] * shape["rx"], matrix[2] * shape["ry"])
        ey = math.hypot(matrix[1] * shape["rx"], matrix[3] * shape["ry"])
        coordinates = [cx - ex, cx + ex, cy - ey, cy + ey]
    else:
        # Cubic curves stay within the convex hull of their control points.
        points = [shape["points"][0]]
        for segment in _path_segments(shape):
            points.extend(segment[1:])
        coordinates = [coordinate for point in points for coordinate in _transform_point(point, matrix)]
    if not all(math.isfinite(number) for number in coordinates):
        _fail(path, "transformed geometry is not finite", "finite_geometry")
    if any(abs(number) > MAX_COORDINATE for number in coordinates):
        _fail(path, f"transformed geometry exceeds coordinate bounds (+/-{MAX_COORDINATE:g})", "reasonable_bounds")


def _compile(scene: Any) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int]]:
    try:
        return _Compiler(scene).compile()
    except SceneValidationError:
        raise
    except (TypeError, ValueError, KeyError, OverflowError, RecursionError) as error:
        raise SceneValidationError(f"scene: malformed data ({type(error).__name__})") from error


def validate(scene: Any) -> Evaluation:
    """Return a fail-fast structural result; malformed input does not escape."""
    try:
        _, _, metrics = _compile(scene)
    except SceneValidationError as error:
        return Evaluation(False, {"scene_structure": False, error.code: False}, {}, [str(error)])
    return Evaluation(
        True,
        {"scene_structure": True, "finite_geometry": True, "bounded_expansion": True,
         "motif_references": True, "canvas_size": True},
        metrics,
        [],
    )


def _shape_bounds(shape: dict[str, Any]) -> tuple[float, float, float, float]:
    """Return conservative transformed bounds for one expanded primitive."""
    matrix = shape["matrix"]
    if shape["type"] == "ellipse":
        cx, cy = _transform_point((shape["cx"], shape["cy"]), matrix)
        ex = math.hypot(matrix[0] * shape["rx"], matrix[2] * shape["ry"])
        ey = math.hypot(matrix[1] * shape["rx"], matrix[3] * shape["ry"])
        return cx - ex, cy - ey, cx + ex, cy + ey
    points = [shape["points"][0]]
    for segment in _path_segments(shape):
        points.extend(segment[1:])
    transformed = [_transform_point(point, matrix) for point in points]
    xs = [point[0] for point in transformed]
    ys = [point[1] for point in transformed]
    return min(xs), min(ys), max(xs), max(ys)


def check_scene(scene: Any, constraints: dict[str, Any] | None = None) -> Evaluation:
    """Check structure and optional mark-making requirements.

    ``max_shapes`` is an integer from 0 to the hard shape limit. ``palette``
    is a list of allowed hex colors; it covers the canvas background and the
    expanded shapes' active fills and strokes. Transparent shapes, ``none``
    paint, and zero-width strokes do not contribute colors. ``shape_palette``
    limits only active shape paint. ``background`` fixes the canvas color.
    ``required_opacity``, ``max_stroke_width``, and ``max_shape_span`` bound
    how each visible transformed mark can be constructed. These are finite
    checks on the scene, not judgments of its subject or visual quality.
    """
    result = validate(scene)
    if not result.valid or constraints is None:
        return result
    try:
        if not isinstance(constraints, dict):
            _fail("constraints", "must be an object", "constraints_schema")
        allowed = {
            "max_shapes", "palette", "shape_palette", "background",
            "required_opacity", "max_stroke_width", "max_shape_span",
        }
        unknown = set(constraints) - allowed
        if unknown:
            _fail("constraints", f"unknown requirements: {sorted(map(str, unknown))}", "constraints_schema")
        if "max_shapes" in constraints:
            limit = constraints["max_shapes"]
            if isinstance(limit, bool) or not isinstance(limit, int) or not 0 <= limit <= MAX_EXPANDED_SHAPES:
                _fail("constraints.max_shapes", f"must be an integer from 0 to {MAX_EXPANDED_SHAPES}", "constraints_schema")
            count = result.metrics["expanded_shapes"]
            result.hard_checks["max_shapes"] = count <= limit
            if count > limit:
                result.violations.append(f"constraints.max_shapes: {count} expanded shapes exceeds {limit}")
        if "palette" in constraints:
            requested = constraints["palette"]
            if not isinstance(requested, list) or len(requested) > 128:
                _fail("constraints.palette", "must be a list of at most 128 hex colors", "constraints_schema")
            allowed = {_color(value, f"constraints.palette[{index}]", False)
                       for index, value in enumerate(requested)}
            canvas, shapes, _ = _compile(scene)
            colors = {canvas["background"]}
            for shape in shapes:
                if shape["opacity"] == 0:
                    continue
                if shape["fill"] != "none":
                    colors.add(shape["fill"])
                if shape["stroke"] != "none" and shape["width"] > 0:
                    colors.add(shape["stroke"])
            missing = sorted(colors - allowed)
            result.hard_checks["palette"] = not missing
            result.metrics["palette_color_count"] = len(colors)
            if missing:
                result.violations.append("constraints.palette: colors outside palette: " + ", ".join(missing))
        canvas, shapes, _ = _compile(scene)
        visible = [
            shape for shape in shapes
            if shape["opacity"] > 0 and (
                shape["fill"] != "none" or
                (shape["stroke"] != "none" and shape["width"] > 0)
            )
        ]
        if "shape_palette" in constraints:
            requested = constraints["shape_palette"]
            if not isinstance(requested, list) or len(requested) > 128:
                _fail("constraints.shape_palette", "must be a list of at most 128 hex colors", "constraints_schema")
            allowed_colors = {
                _color(value, f"constraints.shape_palette[{index}]", False)
                for index, value in enumerate(requested)
            }
            colors = set()
            for shape in visible:
                if shape["fill"] != "none":
                    colors.add(shape["fill"])
                if shape["stroke"] != "none" and shape["width"] > 0:
                    colors.add(shape["stroke"])
            missing = sorted(colors - allowed_colors)
            result.hard_checks["shape_palette"] = not missing
            result.metrics["shape_palette_color_count"] = len(colors)
            if missing:
                result.violations.append(
                    "constraints.shape_palette: shape colors outside palette: " + ", ".join(missing)
                )
        if "background" in constraints:
            expected = _color(constraints["background"], "constraints.background", False)
            result.hard_checks["background"] = canvas["background"] == expected
            if canvas["background"] != expected:
                result.violations.append(
                    f"constraints.background: expected {expected}, got {canvas['background']}"
                )
        if "required_opacity" in constraints:
            opacity = constraints["required_opacity"]
            if isinstance(opacity, bool) or not isinstance(opacity, (int, float)) or not math.isfinite(opacity) or not 0 <= opacity <= 1:
                _fail("constraints.required_opacity", "must be a finite number from 0 to 1", "constraints_schema")
            failures = [index for index, shape in enumerate(visible) if not math.isclose(shape["opacity"], float(opacity), abs_tol=1e-9)]
            result.hard_checks["required_opacity"] = not failures
            if failures:
                result.violations.append(
                    f"constraints.required_opacity: visible shapes {failures[:12]} must use opacity {float(opacity):g}"
                )
        if "max_stroke_width" in constraints:
            limit = constraints["max_stroke_width"]
            if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not math.isfinite(limit) or not 0 <= limit <= MAX_TRANSFORMED_STROKE_WIDTH:
                _fail("constraints.max_stroke_width", f"must be a finite number from 0 to {MAX_TRANSFORMED_STROKE_WIDTH:g}", "constraints_schema")
            widths = [
                shape["width"] * math.hypot(shape["matrix"][0], shape["matrix"][1])
                if shape["stroke"] != "none" and shape["width"] > 0 else 0.0
                for shape in visible
            ]
            maximum = max(widths, default=0.0)
            result.metrics["max_transformed_stroke_width"] = maximum
            result.hard_checks["max_stroke_width"] = maximum <= float(limit)
            if maximum > float(limit):
                result.violations.append(
                    f"constraints.max_stroke_width: transformed width {maximum:g} exceeds {float(limit):g}"
                )
        if "max_shape_span" in constraints:
            limit = constraints["max_shape_span"]
            if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not math.isfinite(limit) or not 0 < limit <= MAX_COORDINATE * 2:
                _fail("constraints.max_shape_span", f"must be a finite number greater than 0 and at most {MAX_COORDINATE * 2:g}", "constraints_schema")
            spans = []
            for shape in visible:
                left, top, right, bottom = _shape_bounds(shape)
                stroke = (
                    shape["width"] * math.hypot(shape["matrix"][0], shape["matrix"][1])
                    if shape["stroke"] != "none" and shape["width"] > 0 else 0.0
                )
                spans.append(max(right - left, bottom - top) + stroke)
            maximum = max(spans, default=0.0)
            result.metrics["max_shape_span"] = maximum
            result.hard_checks["max_shape_span"] = maximum <= float(limit)
            if maximum > float(limit):
                result.violations.append(
                    f"constraints.max_shape_span: transformed mark span {maximum:g} exceeds {float(limit):g}"
                )
    except SceneValidationError as error:
        result.hard_checks[error.code] = False
        result.violations.append(str(error))
    result.valid = all(result.hard_checks.values())
    return result


def expand_scene(scene: Any) -> list[dict[str, Any]]:
    """Expand motifs/groups to normalized primitives; raises on invalid input."""
    return _compile(scene)[1]


def _fmt(number: float | int) -> str:
    return "0" if number == 0 else format(number, ".12g")


def to_svg(scene: Any) -> str:
    """Create self-contained SVG with no external references or executable code."""
    canvas, shapes, _ = _compile(scene)
    w, h = canvas["width"], canvas["height"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" overflow="hidden">',
        f'<rect width="{w}" height="{h}" fill="{canvas["background"]}"/>',
    ]
    for shape in shapes:
        style = (
            f'fill="{shape["fill"]}" stroke="{shape["stroke"]}" '
            f'stroke-width="{_fmt(shape["width"])}" opacity="{_fmt(shape["opacity"])}" '
            'stroke-linecap="round" stroke-linejoin="round" '
            f'transform="matrix({" ".join(_fmt(value) for value in shape["matrix"])})"'
        )
        if shape["type"] == "ellipse":
            geometry = " ".join(f'{key}="{_fmt(shape[key])}"' for key in ("cx", "cy", "rx", "ry"))
            lines.append(f"<ellipse {geometry} {style}/>")
        else:
            parts = ["M " + " ".join(_fmt(value) for value in shape["points"][0])]
            for segment in _path_segments(shape):
                parts.append(segment[0] + " " + " ".join(_fmt(value) for point in segment[1:] for value in point))
            if shape["type"] == "polygon" or shape.get("closed", False):
                parts.append("Z")
            lines.append(f'<path d="{" ".join(parts)}" {style}/>')
    lines.append("</svg>")
    return "\n".join(lines)


def _sample_shape(shape: dict[str, Any], matrix: tuple[float, ...]) -> np.ndarray:
    """Sample the same canonical geometry used by the SVG writer."""
    scale = math.hypot(matrix[0], matrix[1])
    if shape["type"] == "ellipse":
        circumference_bound = 2 * math.pi * max(shape["rx"], shape["ry"]) * scale
        count = max(48, min(720, math.ceil(circumference_bound / 3)))
        angles = np.linspace(0, 2 * math.pi, count, endpoint=False)
        points = np.column_stack((shape["cx"] + shape["rx"] * np.cos(angles), shape["cy"] + shape["ry"] * np.sin(angles)))
    else:
        previous = np.asarray(shape["points"][0], dtype=float)
        chunks = [previous[None, :]]
        for segment in _path_segments(shape):
            if segment[0] == "L":
                previous = np.asarray(segment[1], dtype=float)
                chunks.append(previous[None, :])
            else:
                c1, c2, end = (np.asarray(point, dtype=float) for point in segment[1:])
                length = sum(np.linalg.norm(right - left) for left, right in ((previous, c1), (c1, c2), (c2, end))) * scale
                count = max(8, min(96, math.ceil(length / 3)))
                t = np.linspace(0, 1, count + 1)[1:, None]
                chunks.append((1-t)**3 * previous + 3*(1-t)**2*t*c1 + 3*(1-t)*t**2*c2 + t**3*end)
                previous = end
        points = np.concatenate(chunks)
    a, b, c, d, e, f = matrix
    return points @ np.array([[a, b], [c, d]]) + np.array([e, f])


def _output_size(size: Any, canvas: dict[str, Any]) -> tuple[int, int]:
    if size is None:
        return canvas["width"], canvas["height"]
    if isinstance(size, int) and not isinstance(size, bool):
        size = (size, size)
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError("render size must be an integer or (width,height)")
    if any(isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_CANVAS_SIZE for value in size):
        raise ValueError("render dimensions must be integers from 1 to 1024")
    return int(size[0]), int(size[1])


def render(scene: Any, size: int | tuple[int, int] | None = None) -> Image.Image:
    """Render RGB with antialiasing; requested viewports preserve aspect ratio.

    An integer size produces a square image.  Non-square scenes are centered
    with their background color, clipped at their original canvas boundary.
    """
    canvas, shapes, _ = _compile(scene)
    out_w, out_h = _output_size(size, canvas)
    w, h = out_w * SUPERSAMPLE, out_h * SUPERSAMPLE
    scale = min(w / canvas["width"], h / canvas["height"])
    offset_x = (w - canvas["width"] * scale) / 2
    offset_y = (h - canvas["height"] * scale) / 2
    viewport = (scale, 0.0, 0.0, scale, offset_x, offset_y)
    clip = (math.ceil(offset_x), math.ceil(offset_y), math.floor(w-offset_x), math.floor(h-offset_y))
    image = Image.new("RGBA", (w, h), canvas["background"])
    for shape in shapes:
        if shape["opacity"] == 0 or (shape["fill"] == "none" and (shape["stroke"] == "none" or shape["width"] == 0)):
            continue
        matrix = _compose(viewport, shape["matrix"])
        points = _sample_shape(shape, matrix)
        stroke_width = shape["width"] * math.hypot(matrix[0], matrix[1])
        pad = stroke_width / 2 + 2
        left = max(clip[0], math.floor(float(points[:, 0].min()) - pad))
        top = max(clip[1], math.floor(float(points[:, 1].min()) - pad))
        right = min(clip[2], math.ceil(float(points[:, 0].max()) + pad) + 1)
        bottom = min(clip[3], math.ceil(float(points[:, 1].max()) + pad) + 1)
        if left >= right or top >= bottom:
            continue
        layer = Image.new("RGBA", (right-left, bottom-top), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        local_points = [tuple(point) for point in points - np.array([left, top])]
        closed = shape["type"] in ("polygon", "ellipse") or shape.get("closed", False)
        if shape["fill"] != "none" and len(local_points) >= 3:
            draw.polygon(local_points, fill=ImageColor.getrgb(shape["fill"]) + (255,))
        if shape["stroke"] != "none" and stroke_width > 0:
            stroke = ImageColor.getrgb(shape["stroke"]) + (255,)
            width = max(1, round(stroke_width))
            line_points = local_points + [local_points[0]] if closed else local_points
            draw.line(line_points, fill=stroke, width=width, joint="curve")
            # Match SVG's round caps; also round the closed contour's seam.
            ends = [local_points[0]] if closed else [local_points[0], local_points[-1]]
            radius = width / 2
            for x, y in ends:
                draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=stroke)
        if shape["opacity"] < 1:
            alpha = layer.getchannel("A").point([round(value * shape["opacity"]) for value in range(256)])
            layer.putalpha(alpha)
        image.alpha_composite(layer, (left, top))
    return image.convert("RGB").resize((out_w, out_h), Image.Resampling.LANCZOS)


__all__ = [
    "Evaluation", "SceneValidationError", "validate", "check_scene",
    "expand_scene", "render", "to_svg",
    "MAX_EXPANDED_SHAPES", "MAX_NODES", "MAX_DEPTH", "MAX_POINTS",
]
