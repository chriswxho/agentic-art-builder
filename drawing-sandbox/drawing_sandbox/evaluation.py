"""Typed visual-quality verdicts for planned picture-agent runs."""

from __future__ import annotations

import json
import math
import re
from typing import Any


EVALUATION_PROTOCOL_VERSION = "drawing-sandbox.evaluation.v1"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")


class PictureEvaluationError(ValueError):
    """Raised when a visual evaluator response violates its typed contract."""


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise PictureEvaluationError(f"{label} must be a 1-64 character identifier")
    return value


def _score(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise PictureEvaluationError(f"{label} must be a finite score")
    value = float(value)
    if not 0 <= value <= 100:
        raise PictureEvaluationError(f"{label} must be between 0 and 100")
    return value


def _text(value: Any, label: str, limit: int = 4_000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise PictureEvaluationError(f"{label} must be a non-empty string of at most {limit} characters")
    return value


def _text_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not 1 <= len(value) <= 12:
        raise PictureEvaluationError(f"{label} must contain 1-12 strings")
    for index, item in enumerate(value):
        _text(item, f"{label}[{index}]", 1_000)
    return list(value)


def validate_evaluation(
    evaluation: Any,
    *,
    run_id: str,
    base_revision: int,
    final_revision: int,
    evaluator_id: str,
    evaluation_id: str,
) -> dict:
    if not isinstance(evaluation, dict):
        raise PictureEvaluationError("evaluation must be an object")
    fields = {
        "protocol_version", "evaluator_id", "evaluation_id", "run_id", "base_revision",
        "final_revision", "reference_match_before", "reference_match_after",
        "composition_before", "composition_after", "improved", "summary", "strengths",
        "issues", "next_actions",
    }
    missing = fields - set(evaluation)
    unknown = set(evaluation) - fields
    if missing or unknown:
        raise PictureEvaluationError(
            f"evaluation fields mismatch; missing={sorted(missing)} unknown={sorted(unknown)}"
        )
    if evaluation["protocol_version"] != EVALUATION_PROTOCOL_VERSION:
        raise PictureEvaluationError("evaluation protocol_version does not match")
    if evaluation["run_id"] != run_id:
        raise PictureEvaluationError("evaluation run_id does not match")
    if evaluation["evaluator_id"] != evaluator_id or evaluation["evaluation_id"] != evaluation_id:
        raise PictureEvaluationError("evaluation identity does not match the assigned evaluator")
    _identifier(evaluation["evaluator_id"], "evaluator_id")
    _identifier(evaluation["evaluation_id"], "evaluation_id")
    if evaluation["base_revision"] != base_revision or evaluation["final_revision"] != final_revision:
        raise PictureEvaluationError("evaluation revisions do not match the compared renders")
    normalized = dict(evaluation)
    for field in (
        "reference_match_before", "reference_match_after", "composition_before", "composition_after",
    ):
        normalized[field] = _score(evaluation[field], field)
    if not isinstance(evaluation["improved"], bool):
        raise PictureEvaluationError("improved must be a boolean")
    _text(evaluation["summary"], "summary")
    normalized["strengths"] = _text_list(evaluation["strengths"], "strengths")
    normalized["issues"] = _text_list(evaluation["issues"], "issues")
    normalized["next_actions"] = _text_list(evaluation["next_actions"], "next_actions")
    if evaluation["improved"] != (
        normalized["reference_match_after"] > normalized["reference_match_before"]
    ):
        raise PictureEvaluationError(
            "improved must agree with the reference-match score comparison"
        )
    json.dumps(normalized, allow_nan=False)
    return normalized


__all__ = [
    "EVALUATION_PROTOCOL_VERSION", "PictureEvaluationError", "validate_evaluation",
]
