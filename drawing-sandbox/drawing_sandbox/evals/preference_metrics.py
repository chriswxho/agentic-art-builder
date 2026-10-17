"""Person-specific metrics: how closely the layers match what this person wants.

These run only when the case supplies preference data (from the dataset).
Today that means a preferred step count and/or cumulative reference images
for each layer the person would draw.  Learned per-person feature weights
for steps.* belong here as well once the dataset has enough of them.
"""

from __future__ import annotations

from .core import EvalContext, metric, read_image
from .image_metrics import tone_mae
from .step_metrics import analyze


@metric("preference.step_count", "preference")
def preferred_count(context: EvalContext):
    """|non-empty agent steps - the person's preferred step count|."""
    preferred = context.case.preferred_step_count
    if preferred is None:
        return None
    value = abs(analyze(context)["nonempty_step_count"] - preferred)
    return value, 1 / (1 + value), {"preferred": preferred}


@metric("preference.layer_progression", "preference")
def layer_progression(context: EvalContext):
    """Compare the person's cumulative layer images to the agent's at the same progress."""
    references = context.case.reference_steps
    if not references:
        return None
    steps = context.run.steps
    block = context.options["tone_block"]
    pairs = []
    for position, path in enumerate(references):
        fraction = (position + 1) / len(references)
        agent = max(0, round(fraction * len(steps)) - 1)
        error = tone_mae(steps[agent].image, read_image(path, context.run.canvas), block)
        pairs.append({"reference": str(path), "agent_step": agent, "tone_mae": error})
    value = sum(pair["tone_mae"] for pair in pairs) / len(pairs)
    return value, 1 - value, {"pairs": pairs}
