"""Provisional evals for recorded drawing runs.  See EVALS.md."""

from .core import (
    CATEGORY_WEIGHTS, DEFAULT_OPTIONS, METRICS, EvalCase, compare, evaluate,
    evaluate_suite, load_run, load_suite, metric,
)
# Importing these modules registers their metrics, in this order.
from . import image_metrics, step_metrics, preference_metrics  # noqa: F401,E402

__all__ = [
    "CATEGORY_WEIGHTS", "DEFAULT_OPTIONS", "METRICS", "EvalCase", "compare",
    "evaluate", "evaluate_suite", "load_run", "load_suite", "metric",
]
