"""Step differentiation: are the recorded layers coherent, distinct, and few?

Each primitive in the final scene gets a feature vector (type, ink, fill vs
outline, position, size, orientation) and the step that added it.  Two forces
pull against each other:

* coherence: marks within one step should be alike, so more steps help;
* economy: every extra step costs ``step_penalty``.

The objective for a partition into k steps is

    J(k) = unexplained_variance(k) + step_penalty * (k - 1)

where unexplained_variance is the within-step sum of squares divided by the
total.  Because paint order matters, the oracle only considers contiguous
splits of the final paint order and solves for them exactly with dynamic
programming.  Its argmin is ``steps.suggested_count``, a step count that
balances the two forces.  ``feature_weights`` set which differences count as
"different marks"; they are the obvious knob to fit per person later.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math

import numpy as np

from ..policy import CMYK_INKS
from ..renderer import _shape_bounds, _transform_point
from .core import EvalContext, metric


TYPES = ["path", "polygon", "ellipse"]
INKS = list(CMYK_INKS.values())
INK_NAMES = {value: name for name, value in CMYK_INKS.items()}
FEATURE_GROUPS = ["type", "ink", "fill", "position", "size", "orientation"]


def _ink(primitive: dict) -> str:
    return primitive["fill"] if primitive["fill"] != "none" else primitive["stroke"]


def primitive_features(primitives: list, canvas: tuple[int, int], weights: dict) -> np.ndarray:
    """Weighted feature rows; squared distances between rows are comparable."""
    width, height = canvas
    rows = []
    for primitive in primitives:
        x0, y0, x1, y1 = _shape_bounds(primitive)
        dx, dy = x1 - x0, y1 - y0
        matrix = primitive["matrix"]
        stroke = (primitive["width"] * math.hypot(matrix[0], matrix[1])
                  if primitive["stroke"] != "none" else 0.0)
        if primitive["type"] == "path" and len(primitive["points"]) >= 2:
            start = _transform_point(primitive["points"][0], matrix)
            end = _transform_point(primitive["points"][-1], matrix)
            angle = math.atan2(end[1] - start[1], end[0] - start[0])
        else:
            angle = 0.0 if dx >= dy else math.pi / 2
        elongation = 1 - min(dx, dy) / max(dx, dy) if max(dx, dy) > 0 else 0.0
        # One-hots are scaled so two different categories sit at distance 1.
        groups = {
            "type": [(primitive["type"] == kind) / math.sqrt(2) for kind in TYPES],
            "ink": [(_ink(primitive) == ink) / math.sqrt(2) for ink in INKS],
            "fill": [float(primitive["fill"] != "none")],
            "position": [(x0 + x1) / 2 / width, (y0 + y1) / 2 / height],
            "size": [max(dx, dy) / 96, stroke / 12],
            # Doubled angle: a stroke drawn left-to-right equals right-to-left.
            "orientation": [elongation * math.cos(2 * angle) / 2,
                            elongation * math.sin(2 * angle) / 2],
        }
        row = []
        for group in FEATURE_GROUPS:
            scale = math.sqrt(max(0.0, float(weights.get(group, 0.0))))
            row.extend(value * scale for value in groups[group])
        rows.append(row)
    return np.asarray(rows, dtype=np.float64).reshape(len(primitives), -1)


def _key(primitive: dict) -> str:
    return json.dumps(primitive, sort_keys=True)


def step_labels(run) -> tuple[np.ndarray, list[dict]]:
    """Which step introduced each final primitive, plus per-step add/remove counts.

    Recorded scenes are cumulative.  Append-only growth is the common case;
    otherwise identical primitives are matched as a multiset, so revisions
    are tolerated and reported.
    """
    previous_keys: list[str] = []
    previous_labels: list[int] = []
    info = []
    for position, step in enumerate(run.steps):
        keys = [_key(primitive) for primitive in step.primitives]
        if keys[:len(previous_keys)] == previous_keys:
            labels = previous_labels + [position] * (len(keys) - len(previous_keys))
            removed = 0
        else:
            pool = defaultdict(list)
            for key, label in zip(previous_keys, previous_labels):
                pool[key].append(label)
            labels = [pool[key].pop(0) if pool[key] else position for key in keys]
            removed = sum(len(remaining) for remaining in pool.values())
        info.append({"step": position, "label": step.label,
                     "added": labels.count(position), "removed": removed})
        previous_keys, previous_labels = keys, labels
    return np.asarray(previous_labels, dtype=int), info


def _sse(features: np.ndarray) -> float:
    if len(features) == 0:
        return 0.0
    return float(((features - features.mean(axis=0)) ** 2).sum())


def oracle_segmentation(features: np.ndarray, max_steps: int) -> tuple[list[float], list[list[int]]]:
    """Exact minimum within-segment SSE for 1..max_steps contiguous segments.

    Returns the SSE curve and, for each k, the segment start indices.
    """
    n = len(features)
    limit = max(1, min(max_steps, n))
    first = np.vstack([np.zeros((1, features.shape[1])), features.cumsum(axis=0)])
    second = np.concatenate([[0.0], (features ** 2).sum(axis=1).cumsum()])
    best = np.full((limit + 1, n + 1), np.inf)
    parent = np.zeros((limit + 1, n + 1), dtype=int)
    best[0, 0] = 0.0
    for k in range(1, limit + 1):
        for end in range(k, n + 1):
            starts = np.arange(k - 1, end)
            sums = first[end] - first[starts]
            cost = second[end] - second[starts] - (sums ** 2).sum(axis=1) / (end - starts)
            total = best[k - 1, starts] + np.maximum(cost, 0.0)
            choice = int(np.argmin(total))
            best[k, end], parent[k, end] = total[choice], starts[choice]
    curve, boundaries = [], []
    for k in range(1, limit + 1):
        curve.append(float(best[k, n]))
        cuts, end = [], n
        for level in range(k, 0, -1):
            end = int(parent[level, end])
            cuts.append(end)
        boundaries.append(sorted(cuts))
    return curve, boundaries


def _summarize(primitives: list) -> dict:
    """Human-readable gist of a group of marks."""
    if not primitives:
        return {"count": 0}
    inks = Counter(INK_NAMES.get(_ink(p), _ink(p)) for p in primitives)
    types = Counter(p["type"] for p in primitives)
    bounds = np.array([_shape_bounds(p) for p in primitives])
    return {
        "count": len(primitives),
        "inks": dict(inks.most_common()),
        "types": dict(types.most_common()),
        "bbox": [round(float(bounds[:, 0].min())), round(float(bounds[:, 1].min())),
                 round(float(bounds[:, 2].max())), round(float(bounds[:, 3].max()))],
    }


def analyze(context: EvalContext) -> dict:
    """Shared computation for every steps.* metric."""
    def compute():
        run = context.run
        primitives = run.final.primitives
        labels, info = step_labels(run)
        features = primitive_features(primitives, run.canvas, context.options["feature_weights"])
        total = _sse(features)
        penalty = context.options["step_penalty"]
        used = sorted(set(labels.tolist()))
        within = sum(_sse(features[labels == label]) for label in used)
        agent_unexplained = within / total if total > 0 else 0.0
        agent_objective = agent_unexplained + penalty * (len(used) - 1)

        curve, boundaries = oracle_segmentation(features, context.options["max_oracle_steps"])
        unexplained = [sse / total if total > 0 else 0.0 for sse in curve]
        objective = [u + penalty * k for k, u in enumerate(unexplained)]
        suggested = int(np.argmin(objective)) + 1
        starts = boundaries[suggested - 1] + [len(primitives)]
        oracle_steps = [dict(range=[a, b], **_summarize(primitives[a:b]))
                        for a, b in zip(starts, starts[1:])]
        agent_steps = [dict(entry, **_summarize([p for p, l in zip(primitives, labels) if l == entry["step"]]))
                       for entry in info]
        return {
            "step_count": len(run.steps),
            "nonempty_step_count": len(used),
            "agent_unexplained": agent_unexplained,
            "agent_objective": agent_objective,
            "oracle_unexplained": unexplained,
            "oracle_objective": objective,
            "suggested_count": suggested,
            "agent_steps": agent_steps,
            "oracle_steps": oracle_steps,
        }
    return context.cached("steps", compute)


@metric("steps.count", "steps")
def count(context: EvalContext):
    """Number of recorded steps, with per-step marks added/removed and their gist."""
    data = analyze(context)
    return data["step_count"], None, {"nonempty": data["nonempty_step_count"],
                                       "steps": data["agent_steps"]}


@metric("steps.coherence", "steps")
def coherence(context: EvalContext):
    """Share of mark-feature variance explained by the step grouping (R^2)."""
    value = 1 - analyze(context)["agent_unexplained"]
    return value, value, {}


@metric("steps.suggested_count", "steps")
def suggested_count(context: EvalContext):
    """Oracle step count minimizing unexplained variance + step_penalty per extra step."""
    data = analyze(context)
    return data["suggested_count"], None, {
        "step_penalty": context.options["step_penalty"],
        "unexplained_by_k": data["oracle_unexplained"],
        "objective_by_k": data["oracle_objective"],
        "suggested_steps": data["oracle_steps"],
    }


@metric("steps.count_error", "steps")
def count_error(context: EvalContext):
    """|non-empty agent steps - suggested count|."""
    data = analyze(context)
    value = abs(data["nonempty_step_count"] - data["suggested_count"])
    return value, 1 / (1 + value), {}


@metric("steps.objective_gap", "steps")
def objective_gap(context: EvalContext):
    """Agent's penalized objective minus the oracle's; 0 means as good as the best split."""
    data = analyze(context)
    oracle = min(data["oracle_objective"])
    value = data["agent_objective"] - oracle
    return value, math.exp(-max(0.0, value) / context.options["gap_scale"]), {
        "agent_objective": data["agent_objective"], "oracle_objective": oracle}


@metric("steps.visual_impact", "steps")
def visual_impact(context: EvalContext):
    """Share of steps whose pixels visibly change the drawing."""
    previous = np.ones_like(context.run.steps[0].image)
    impacts = []
    for step in context.run.steps:
        impacts.append(float(np.abs(step.image - previous).mean()))
        previous = step.image
    threshold = context.options["min_step_impact"]
    value = sum(impact >= threshold for impact in impacts) / len(impacts)
    return value, value, {"impact_by_step": impacts, "threshold": threshold}
