"""Typed regional plans for delegating picture detail to specialist agents."""

from __future__ import annotations

from copy import deepcopy
import json
import math
import re
from typing import Any

from .iteration_context import proposal_skill, validate_context
from .renderer import MAX_EXPANDED_SHAPES


PLAN_PROTOCOL_VERSION = "drawing-sandbox.plan.v1"
PRIMITIVE_TYPES = ("path", "polygon", "ellipse")
CMYK_COLORS = ("#00ffff", "#ff00ff", "#ffff00", "#000000")
MAX_ASSIGNMENTS = 8
MAX_SHAPES_PER_ASSIGNMENT = 48
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")


class PicturePlanError(ValueError):
    """Raised when a planner result or delegated proposal violates its contract."""


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise PicturePlanError(f"{label} must be a 1-64 character identifier")
    return value


def _text(value: Any, label: str, *, limit: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise PicturePlanError(f"{label} must be a non-empty string of at most {limit} characters")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise PicturePlanError(f"{label} must be a finite number")
    return float(value)


def validate_plan(context: dict, plan: Any) -> dict:
    """Validate and normalize a planner's regional delegation plan."""
    rendered = validate_context(context)
    if not isinstance(plan, dict):
        raise PicturePlanError("plan must be an object")
    fields = {
        "protocol_version", "planner_id", "plan_id", "base_revision", "base_state_sha256",
        "critique", "strategy", "assignments",
    }
    unknown = set(plan) - fields
    missing = fields - set(plan)
    if unknown or missing:
        raise PicturePlanError(f"plan fields mismatch; missing={sorted(missing)} unknown={sorted(unknown)}")
    if plan["protocol_version"] != PLAN_PROTOCOL_VERSION:
        raise PicturePlanError("plan protocol_version does not match")
    _identifier(plan["planner_id"], "planner_id")
    _identifier(plan["plan_id"], "plan_id")
    if isinstance(plan["base_revision"], bool) or plan["base_revision"] != context["revision"]:
        raise PicturePlanError("plan base_revision is stale")
    if plan["base_state_sha256"] != context["state_sha256"]:
        raise PicturePlanError("plan base_state_sha256 does not match the context")
    _text(plan["critique"], "critique", limit=4_000)
    _text(plan["strategy"], "strategy", limit=4_000)
    assignments = plan["assignments"]
    if not isinstance(assignments, list) or not 2 <= len(assignments) <= MAX_ASSIGNMENTS:
        raise PicturePlanError(f"assignments must contain 2-{MAX_ASSIGNMENTS} regional tasks")

    canvas = context["scene_template"]["canvas"]
    canvas_width = float(canvas["width"])
    canvas_height = float(canvas["height"])
    available_layers = {
        layer["layer_id"]: layer for layer in context["layers"] if not layer["locked"]
    }
    assignment_ids: set[str] = set()
    total_budget = 0
    total_minimum = 0
    normalized = deepcopy(plan)
    for index, assignment in enumerate(assignments):
        label = f"assignments[{index}]"
        if not isinstance(assignment, dict):
            raise PicturePlanError(f"{label} must be an object")
        assignment_fields = {
            "assignment_id", "part_label", "brief", "region", "layer_id",
            "shape_types", "colors", "minimum_shapes", "shape_budget",
        }
        unknown = set(assignment) - assignment_fields
        missing = assignment_fields - set(assignment)
        if unknown or missing:
            raise PicturePlanError(
                f"{label} fields mismatch; missing={sorted(missing)} unknown={sorted(unknown)}"
            )
        assignment_id = _identifier(assignment["assignment_id"], f"{label}.assignment_id")
        if assignment_id in assignment_ids:
            raise PicturePlanError(f"duplicate assignment_id {assignment_id!r}")
        assignment_ids.add(assignment_id)
        _text(assignment["part_label"], f"{label}.part_label", limit=200)
        _text(assignment["brief"], f"{label}.brief", limit=2_000)
        layer_id = _identifier(assignment["layer_id"], f"{label}.layer_id")
        if layer_id not in available_layers:
            raise PicturePlanError(f"{label}.layer_id must name an existing unlocked layer")

        region = assignment["region"]
        if not isinstance(region, dict) or set(region) != {"x", "y", "width", "height"}:
            raise PicturePlanError(f"{label}.region must contain x, y, width, and height")
        x = _number(region["x"], f"{label}.region.x")
        y = _number(region["y"], f"{label}.region.y")
        width = _number(region["width"], f"{label}.region.width")
        height = _number(region["height"], f"{label}.region.height")
        if width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > canvas_width or y + height > canvas_height:
            raise PicturePlanError(f"{label}.region must be a positive rectangle inside the canvas")
        normalized["assignments"][index]["region"] = {
            "x": x, "y": y, "width": width, "height": height,
        }

        shape_types = assignment["shape_types"]
        if (not isinstance(shape_types, list) or not shape_types or
                len(shape_types) != len(set(shape_types)) or
                any(value not in PRIMITIVE_TYPES for value in shape_types)):
            raise PicturePlanError(f"{label}.shape_types must be distinct Drawing Sandbox primitives")
        colors = assignment["colors"]
        if (not isinstance(colors, list) or not colors or len(colors) != len(set(colors)) or
                any(value not in CMYK_COLORS for value in colors)):
            raise PicturePlanError(f"{label}.colors must be distinct CMYK paint values")
        minimum = assignment["minimum_shapes"]
        budget = assignment["shape_budget"]
        if (isinstance(minimum, bool) or not isinstance(minimum, int) or
                isinstance(budget, bool) or not isinstance(budget, int) or
                not 1 <= minimum <= budget <= MAX_SHAPES_PER_ASSIGNMENT):
            raise PicturePlanError(
                f"{label} shape counts must satisfy 1 <= minimum_shapes <= shape_budget <= "
                f"{MAX_SHAPES_PER_ASSIGNMENT}"
            )
        total_minimum += minimum
        total_budget += budget

    current_shapes = rendered["checks"]["metrics"]["expanded_shapes"]
    remaining = MAX_EXPANDED_SHAPES - current_shapes
    if total_budget > remaining:
        raise PicturePlanError(
            f"delegated shape budget {total_budget} exceeds remaining capacity {remaining}"
        )
    if total_minimum > remaining:
        raise PicturePlanError("delegated minimum shapes exceed remaining capacity")
    json.dumps(normalized, allow_nan=False)
    return normalized


def _primitive_bounds(node: dict) -> tuple[float, float, float, float]:
    node_type = node.get("type")
    stroke = node.get("stroke", "none")
    width = node.get("width", 0)
    pad = float(width) / 2 if stroke != "none" else 0.0
    if node_type == "ellipse":
        cx = float(node["cx"])
        cy = float(node["cy"])
        rx = float(node["rx"])
        ry = float(node["ry"])
        return cx - rx - pad, cy - ry - pad, cx + rx + pad, cy + ry + pad
    points = node.get("points")
    if not isinstance(points, list) or not points:
        raise PicturePlanError("delegated path/polygon must contain points")
    if node_type == "path" and node.get("smooth", False):
        raise PicturePlanError("delegated paths must use smooth=false so regional bounds are exact")
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad


def assignment_object_prefix(context: dict, assignment: dict) -> str:
    part = re.sub(r"[^a-z0-9]+", "-", assignment["assignment_id"].lower()).strip("-")
    return f"plan-{part[:32]}-r{context['revision'] + 1:03d}-"


def validate_delegated_proposal(context: dict, proposal: dict, assignment: dict) -> None:
    """Require an add_objects proposal to stay inside its planner assignment."""
    skill, operation = proposal_skill(proposal)
    if skill != "add_objects":
        raise PicturePlanError("delegated detail agents must return add_objects")
    if operation.get("layer_id") != assignment["layer_id"]:
        raise PicturePlanError("delegated proposal targeted a layer outside its assignment")
    objects = operation.get("objects")
    if not isinstance(objects, list):
        raise PicturePlanError("delegated add_objects operation must contain objects")
    if not assignment["minimum_shapes"] <= len(objects) <= assignment["shape_budget"]:
        raise PicturePlanError(
            f"delegated proposal must add {assignment['minimum_shapes']}-"
            f"{assignment['shape_budget']} shapes"
        )
    region = assignment["region"]
    left = float(region["x"])
    top = float(region["y"])
    right = left + float(region["width"])
    bottom = top + float(region["height"])
    allowed_types = set(assignment["shape_types"])
    allowed_colors = set(assignment["colors"])
    object_prefix = assignment_object_prefix(context, assignment)
    for index, item in enumerate(objects):
        if not item.get("object_id", "").startswith(object_prefix):
            raise PicturePlanError(
                f"delegated object {index} ID must start with {object_prefix!r}"
            )
        node = item.get("renderer_node", {})
        if node.get("type") not in allowed_types:
            raise PicturePlanError(f"delegated object {index} uses an unassigned shape type")
        paints = set()
        if node.get("fill", "none") != "none":
            paints.add(node["fill"])
        if node.get("stroke", "none") != "none" and node.get("width", 0) > 0:
            paints.add(node["stroke"])
        if not paints or not paints <= allowed_colors:
            raise PicturePlanError(f"delegated object {index} uses paint outside its assignment")
        obj_left, obj_top, obj_right, obj_bottom = _primitive_bounds(node)
        if obj_left < left or obj_top < top or obj_right > right or obj_bottom > bottom:
            raise PicturePlanError(f"delegated object {index} extends outside its assigned region")


__all__ = [
    "CMYK_COLORS", "MAX_ASSIGNMENTS", "MAX_SHAPES_PER_ASSIGNMENT", "PLAN_PROTOCOL_VERSION",
    "PRIMITIVE_TYPES", "PicturePlanError", "assignment_object_prefix",
    "validate_delegated_proposal", "validate_plan",
]
