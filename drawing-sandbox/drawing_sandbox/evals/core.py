"""Provisional evaluation harness for recorded composition runs.

An eval reads a run written by ``record_composition_step`` (its manifest,
per-step scenes, and per-step PNGs), optionally a target image and a
person's preferred layer progression, and scores it with registered metrics.

Nothing here is final.  Metrics register themselves with ``@metric``; adding,
removing, or reweighting one does not require touching the runner.  Every
metric reports a raw ``value`` and, when it makes sense to aggregate, a
``score`` in [0, 1] where higher is better.  Informational metrics leave
``score`` as None and are excluded from the aggregate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image

from ..renderer import expand_scene


DEFAULT_OPTIONS: dict[str, Any] = {
    # Image similarity: compare tone after averaging tone_block x tone_block cells,
    # because overprinted hatching only reads as a color at a distance.
    "tone_block": 8,
    "ssim_radius": 3,
    # Step differentiation: a layer must explain at least step_penalty of the
    # mark-feature variance to be worth its own step.
    "step_penalty": 0.05,
    "max_oracle_steps": 24,
    "gap_scale": 0.1,
    "min_step_impact": 0.002,
    # Per-person notion of "similar marks".  See step_metrics.FEATURE_GROUPS.
    "feature_weights": {
        "type": 1.0, "ink": 1.0, "fill": 0.5,
        "position": 1.0, "size": 0.75, "orientation": 0.5,
    },
}

CATEGORY_WEIGHTS = {"image": 0.4, "steps": 0.4, "preference": 0.2}


@dataclass
class Step:
    index: int
    label: str
    scene: dict
    image: np.ndarray
    primitives: list


@dataclass
class Run:
    run_id: str
    directory: Path
    manifest: dict
    steps: list

    @property
    def final(self) -> Step:
        return self.steps[-1]

    @property
    def canvas(self) -> tuple[int, int]:
        canvas = self.final.scene["canvas"]
        return canvas["width"], canvas["height"]


@dataclass
class EvalCase:
    """One evaluation input.  Dataset rows should map onto this shape."""

    case_id: str
    run_dir: Path
    target_image: Optional[Path] = None
    # Cumulative images of how a person wants the layers to build up.
    reference_steps: list = field(default_factory=list)
    preferred_step_count: Optional[int] = None
    options: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict, base: str | Path = ".") -> "EvalCase":
        base = Path(base)
        target = data.get("target_image")
        return cls(
            case_id=data.get("case_id") or Path(data["run"]).name,
            run_dir=base / data["run"],
            target_image=base / target if target else None,
            reference_steps=[base / path for path in data.get("reference_steps", [])],
            preferred_step_count=data.get("preferred_step_count"),
            options=data.get("options", {}),
        )


@dataclass
class MetricResult:
    name: str
    category: str
    value: Optional[float]
    score: Optional[float] = None
    details: dict = field(default_factory=dict)


@dataclass
class EvalContext:
    case: EvalCase
    run: Run
    target: Optional[np.ndarray]
    options: dict
    cache: dict = field(default_factory=dict)

    def cached(self, key: str, compute: Callable[[], Any]) -> Any:
        if key not in self.cache:
            self.cache[key] = compute()
        return self.cache[key]


# A metric returns (value, score, details), or None when its inputs are absent.
MetricOutput = Optional[Tuple[Optional[float], Optional[float], dict]]
METRICS: dict[str, tuple[str, Callable[[EvalContext], MetricOutput]]] = {}


def metric(name: str, category: str):
    def register(function: Callable[[EvalContext], MetricOutput]):
        METRICS[name] = (category, function)
        return function
    return register


def read_image(path: str | Path, size: tuple[int, int] | None = None) -> np.ndarray:
    """RGB float array in [0, 1], resized to ``size`` (width, height) if given."""
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != tuple(size):
        image = image.resize(tuple(size), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255


def load_run(run_dir: str | Path) -> Run:
    """Load recorded steps up to and including the final step."""
    directory = Path(run_dir)
    manifest = json.loads((directory / "manifest.json").read_text())
    entries = manifest["steps"]
    if manifest.get("final_step") is not None:
        entries = entries[: manifest["final_step"] + 1]
    if not entries:
        raise ValueError(f"{directory}: run has no recorded steps")
    steps = []
    for entry in entries:
        scene = json.loads((directory / entry["scene"]).read_text())
        steps.append(Step(entry["index"], entry["label"], scene,
                          read_image(directory / entry["png"]), expand_scene(scene)))
    return Run(manifest["run_id"], directory, manifest, steps)


def _default_target(run: Run) -> Path | None:
    """Fall back to composition.source_image, as the Tahoe run records it."""
    source = run.manifest.get("composition", {}).get("source_image")
    if not isinstance(source, str):
        return None
    for base in (run.directory, run.directory.parent.parent, Path.cwd()):
        if (base / source).is_file():
            return base / source
    return None


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _plain(value: Any) -> Any:
    """Make numpy values JSON-serializable."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def evaluate(case: EvalCase, only: list[str] | None = None) -> dict:
    """Run every registered metric (or names/prefixes in ``only``) on one case."""
    run = load_run(case.run_dir)
    options = {**DEFAULT_OPTIONS, **case.options}
    target_path = case.target_image or _default_target(run)
    target = read_image(target_path, run.canvas) if target_path else None
    context = EvalContext(case, run, target, options)

    results = []
    for name, (category, function) in METRICS.items():
        if only and not any(name == prefix or name.startswith(prefix + ".") for prefix in only):
            continue
        output = function(context)
        if output is not None:
            value, score, details = output
            results.append(MetricResult(name, category, value, score, details))

    category_scores = {}
    for category in CATEGORY_WEIGHTS:
        scores = [result.score for result in results
                  if result.category == category and result.score is not None]
        if scores:
            category_scores[category] = float(np.mean(scores))
    weight = sum(CATEGORY_WEIGHTS[category] for category in category_scores)
    overall = (sum(CATEGORY_WEIGHTS[c] * s for c, s in category_scores.items()) / weight
               if weight else None)
    policy = next((r for r in results if r.name == "policy.valid"), None)
    if policy is not None and not policy.value:
        overall = 0.0

    return _plain({
        "case_id": case.case_id,
        "run_id": run.run_id,
        "run_dir": case.run_dir,
        "target_image": target_path,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "options": options,
        "overall": overall,
        "category_scores": category_scores,
        "metrics": {result.name: asdict(result) for result in results},
    })


def evaluate_suite(cases: list[EvalCase], only: list[str] | None = None) -> dict:
    reports = [evaluate(case, only) for case in cases]
    overall = [report["overall"] for report in reports if report["overall"] is not None]
    return {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "mean_overall": float(np.mean(overall)) if overall else None,
        "reports": reports,
    }


def load_suite(path: str | Path) -> list[EvalCase]:
    """A suite file is {"cases": [...]}; paths resolve relative to the file."""
    path = Path(path)
    data = json.loads(path.read_text())
    return [EvalCase.from_dict(case, path.parent) for case in data["cases"]]


def compare(before: dict, after: dict) -> list[dict]:
    """Per-case, per-metric deltas between two suite results (eval iteration)."""
    def index(suite: dict) -> dict:
        rows = {}
        for report in suite["reports"]:
            rows[(report["case_id"], "overall")] = report["overall"]
            for name, result in report["metrics"].items():
                rows[(report["case_id"], name)] = result["value"]
        return rows

    old, new = index(before), index(after)
    rows = []
    for key in sorted(set(old) | set(new)):
        a, b = old.get(key), new.get(key)
        delta = b - a if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
        rows.append({"case_id": key[0], "metric": key[1], "before": a, "after": b, "delta": delta})
    return rows
