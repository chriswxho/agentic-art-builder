"""Durable progress records for constrained picture-composition runs."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from .policy import current_instruction_set
from .report import write_run_report


_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")


def _json_size(value: Any, label: str, limit: int = 256_000) -> str:
    try:
        encoded = json.dumps(value, indent=2, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite JSON data") from error
    if len(encoded.encode()) > limit:
        raise ValueError(f"{label} exceeds {limit} bytes")
    return encoded


def _validate_metadata(run_id: Any, prompt_set: Any, composition: Any, label: Any, final: Any, model_output: Any = None) -> None:
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise ValueError("run_id must use 1-64 letters, numbers, underscores, or hyphens")
    if not isinstance(prompt_set, list) or not 1 <= len(prompt_set) <= 32:
        raise ValueError("prompt_set must contain 1-32 strings")
    if any(not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 8_000 for prompt in prompt_set):
        raise ValueError("each prompt must be a non-empty string of at most 8,000 characters")
    if not isinstance(composition, dict):
        raise ValueError("composition must be a JSON object")
    if label is not None and (not isinstance(label, str) or len(label) > 200):
        raise ValueError("step_label must be a string of at most 200 characters")
    if not isinstance(final, bool):
        raise ValueError("final must be a boolean")
    if model_output is not None and (not isinstance(model_output, str) or len(model_output) > 200_000):
        raise ValueError("model_output must be a string of at most 200,000 characters")
    _json_size(prompt_set, "prompt_set")
    _json_size(composition, "composition")


def record_composition_step(
    scene: dict,
    rendered: dict,
    *,
    run_id: str,
    prompt_set: list[str],
    composition: dict,
    step_label: str | None = None,
    final: bool = False,
    model_output: str | None = None,
    root: str | Path = "recordings",
) -> dict:
    """Append one valid render and its provenance to a composition run.

    ``model_output`` is the raw text the model returned for this step (its
    reasoning, notes, or the reply that contained the scene).  It is saved
    verbatim and shown in the run's human-readable ``report.md``.
    """
    _validate_metadata(run_id, prompt_set, composition, step_label, final, model_output)
    if not rendered.get("checks", {}).get("valid"):
        raise ValueError("only valid constrained renders can be recorded")
    if "svg" not in rendered or "png_data_url" not in rendered:
        raise ValueError("recording requires rendered SVG and PNG output")

    record_root = Path(root)
    record_root.mkdir(parents=True, exist_ok=True)
    run_dir = record_root / run_id
    if run_dir.is_symlink():
        raise ValueError("run directory cannot be a symbolic link")
    steps_dir = run_dir / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("prompt_set") != prompt_set or manifest.get("composition") != composition:
            raise ValueError("prompt_set and composition must remain unchanged within a run")
    else:
        manifest = {
            "format_version": 1,
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "instruction_set": current_instruction_set(),
            "prompt_set": prompt_set,
            "composition": composition,
            "steps": [],
            "final_step": None,
        }

    index = len(manifest["steps"])
    stem = f"{index:03d}"
    scene_name = f"steps/{stem}-scene.json"
    svg_name = f"steps/{stem}-drawing.svg"
    png_name = f"steps/{stem}-drawing.png"
    checks_name = f"steps/{stem}-checks.json"
    prefix = "data:image/png;base64,"
    png_data = rendered["png_data_url"]
    if not isinstance(png_data, str) or not png_data.startswith(prefix):
        raise ValueError("rendered PNG must be a base64 data URL")
    try:
        png_bytes = base64.b64decode(png_data[len(prefix):], validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError("rendered PNG contains invalid base64 data") from error
    (run_dir / scene_name).write_text(_json_size(scene, "scene") + "\n")
    (run_dir / svg_name).write_text(rendered["svg"])
    (run_dir / png_name).write_bytes(png_bytes)
    (run_dir / checks_name).write_text(_json_size(rendered["checks"], "checks") + "\n")

    step = {
        "index": index,
        "label": step_label or f"Step {index + 1}",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "scene": scene_name,
        "svg": svg_name,
        "png": png_name,
        "checks": checks_name,
        "metrics": rendered["checks"].get("metrics", {}),
        "final": final,
    }
    if model_output is not None:
        step["model_output"] = f"steps/{stem}-model-output.txt"
        (run_dir / step["model_output"]).write_text(model_output)
    manifest["steps"].append(step)
    if final:
        (run_dir / "final-scene.json").write_text(_json_size(scene, "scene") + "\n")
        (run_dir / "final-drawing.svg").write_text(rendered["svg"])
        (run_dir / "final-drawing.png").write_bytes(png_bytes)
        manifest["final_step"] = index
    manifest["updated_at"] = step["recorded_at"]
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    report_path = write_run_report(run_dir)
    return {
        "run_id": run_id,
        "step": index,
        "final": final,
        "directory": str(run_dir.resolve()),
        "manifest": str(manifest_path.resolve()),
        "report": str(report_path.resolve()),
    }


__all__ = ["record_composition_step"]
