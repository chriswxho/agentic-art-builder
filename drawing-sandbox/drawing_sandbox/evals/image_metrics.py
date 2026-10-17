"""Image similarity between the rendered run and its target image.

``pixel_mae`` is the naive baseline.  Because these drawings build color from
overprinted 50%-opacity hatching, a per-pixel comparison mostly measures
texture, so ``tone_mae`` and ``ssim`` compare block-averaged images instead.
``skill_vs_blank`` normalizes against the blank white canvas: the "base
pixel image" every run starts from.
"""

from __future__ import annotations

import numpy as np

from ..policy import composition_constraints
from ..renderer import check_scene
from .core import EvalContext, metric


def block_mean(image: np.ndarray, block: int) -> np.ndarray:
    """Average non-overlapping block x block cells (crops the ragged edge)."""
    if block <= 1:
        return image
    height, width = image.shape[0] // block * block, image.shape[1] // block * block
    cropped = image[:height, :width]
    return cropped.reshape(height // block, block, width // block, block, -1).mean(axis=(1, 3))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).mean())


def tone_mae(a: np.ndarray, b: np.ndarray, block: int) -> float:
    return mae(block_mean(a, block), block_mean(b, block))


def _box_mean(x: np.ndarray, radius: int) -> np.ndarray:
    size = 2 * radius + 1
    total = np.pad(x, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return (total[size:, size:] - total[:-size, size:] - total[size:, :-size]
            + total[:-size, :-size]) / (size * size)


def ssim(a: np.ndarray, b: np.ndarray, radius: int = 3) -> float:
    """Mean SSIM of luminance with a box window."""
    luma = np.array([0.299, 0.587, 0.114], dtype=np.float64)
    x, y = a.astype(np.float64) @ luma, b.astype(np.float64) @ luma
    if min(x.shape) <= 2 * radius:
        return float("nan")
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mx, my = _box_mean(x, radius), _box_mean(y, radius)
    vx = _box_mean(x * x, radius) - mx * mx
    vy = _box_mean(y * y, radius) - my * my
    cov = _box_mean(x * y, radius) - mx * my
    value = ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
    return float(value.mean())


def _blank(context: EvalContext) -> np.ndarray:
    return np.ones_like(context.run.final.image)


@metric("policy.valid", "policy")
def policy_valid(context: EvalContext):
    """Final scene passes the agent-facing CMYK policy; a failure zeroes overall."""
    checks = check_scene(context.run.final.scene, composition_constraints())
    return float(checks.valid), None, {"violations": checks.violations[:10]}


@metric("image.pixel_mae", "image")
def pixel_mae(context: EvalContext):
    """Naive mean absolute RGB difference from the target."""
    if context.target is None:
        return None
    value = mae(context.run.final.image, context.target)
    return value, 1 - value, {}


@metric("image.tone_mae", "image")
def tone(context: EvalContext):
    """Mean absolute difference after block-averaging, i.e. viewed from a distance."""
    if context.target is None:
        return None
    block = context.options["tone_block"]
    value = tone_mae(context.run.final.image, context.target, block)
    return value, 1 - value, {"block": block}


@metric("image.ssim", "image")
def structural(context: EvalContext):
    """Luminance SSIM on lightly block-averaged images."""
    if context.target is None:
        return None
    # Half the tone block keeps edges while still smoothing hatching.
    block = max(1, context.options["tone_block"] // 2)
    value = ssim(block_mean(context.run.final.image, block), block_mean(context.target, block),
                 context.options["ssim_radius"])
    return value, max(0.0, value), {"block": block}


@metric("image.skill_vs_blank", "image")
def skill_vs_blank(context: EvalContext):
    """1 = perfect, 0 = no better than the untouched white canvas, < 0 = worse."""
    if context.target is None:
        return None
    block = context.options["tone_block"]
    blank = tone_mae(_blank(context), context.target, block)
    final = tone_mae(context.run.final.image, context.target, block)
    value = 1 - final / blank if blank > 0 else 0.0
    return value, min(1.0, max(0.0, value)), {"blank_tone_mae": blank, "final_tone_mae": final}


@metric("image.progress_monotonicity", "image")
def progress(context: EvalContext):
    """Share of steps that move the drawing closer to the target."""
    if context.target is None:
        return None
    block = context.options["tone_block"]
    curve = [tone_mae(_blank(context), context.target, block)]
    curve += [tone_mae(step.image, context.target, block) for step in context.run.steps]
    improved = [later < earlier for earlier, later in zip(curve, curve[1:])]
    value = sum(improved) / len(improved)
    return value, value, {"tone_mae_by_step": curve[1:], "blank_tone_mae": curve[0]}
