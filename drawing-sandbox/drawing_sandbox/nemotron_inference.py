"""Crusoe Managed Inference client for constrained Nemotron picture agents."""

from __future__ import annotations

import base64
from copy import deepcopy
from datetime import datetime, timezone
from io import BytesIO
import json
import mimetypes
import os
from pathlib import Path
import re
import ssl
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import certifi
from PIL import Image, ImageDraw, ImageFont

from .evaluation import (
    EVALUATION_PROTOCOL_VERSION,
    PictureEvaluationError,
    validate_evaluation,
)
from .iteration_context import (
    EVALUATOR_MODEL,
    GENERATOR_MODEL,
    INFERENCE_ENDPOINT,
    INFERENCE_PROVIDER,
    PROTOCOL_VERSION,
    PictureContextError,
    apply_proposal,
    proposal_skill,
    validate_context,
)
from .planning import (
    MAX_ASSIGNMENTS,
    MAX_SHAPES_PER_ASSIGNMENT,
    PLAN_PROTOCOL_VERSION,
    PicturePlanError,
    assignment_object_prefix,
    validate_delegated_proposal,
    validate_plan,
)
from .renderer import MAX_EXPANDED_SHAPES


class NemotronInferenceError(RuntimeError):
    """Raised when Crusoe inference cannot produce a valid agent proposal."""

    def __init__(self, message: str, *, log: dict | None = None):
        super().__init__(message)
        self.log = log


SYSTEM_PROMPT = """You are one picture-editing agent in a consecutive-agent drawing workflow.
You receive a complete typed PictureContext plus the current render and visual reference.
Return exactly one JSON object and nothing else. Do not use Markdown fences.
Never rewrite the scene or context. Use only the requested typed skill operation.
Preserve existing stable IDs and invent unique descriptive IDs only where the skill permits.
All new visible primitives must be paths, polygons, or ellipses using only #00ffff, #ff00ff,
#ffff00, or #000000 on white. Every visible primitive must have opacity 0.5, transformed span
at most 96 canvas units, and stroke width at most 12. Prefer a few precise small marks.
The orchestrator will reject stale, malformed, or policy-violating output atomically."""


PLANNER_SYSTEM_PROMPT = """You are the planning agent for a constrained picture-editing team.
You receive the complete typed PictureContext, current render, and visual reference. Return exactly
one PicturePlan JSON object and nothing else. Do not use Markdown fences. Divide visual refinement
into ordered regional assignments. Every assignment delegates a canvas region, an existing target
layer, allowed primitive types, allowed CMYK colors, and a shape-count range. Use only paths,
polygons, and ellipses; never delegate bitmaps, groups, motifs, or background replacement. Favor
small layered marks that improve correspondence with the reference. The deterministic orchestrator
will reject stale plans, invalid layers, out-of-canvas regions, or budgets beyond the 2,048-shape
composition ceiling."""


EVALUATOR_SYSTEM_PROMPT = """You are the independent visual evaluator for a constrained picture
editing run. Compare the baseline render and final render against the photographic reference.
Judge reference correspondence and composition honestly; more marks are not automatically better.
Return exactly one PictureEvaluation JSON object and nothing else. Do not use Markdown fences.
Scores are 0-100. Set improved=true only when reference_match_after is strictly greater than
reference_match_before. Identify concrete strengths, remaining issues, and next actions."""


SKILL_GUIDANCE = {
    "split_layers": """Use split_layers only. Choose one unlocked layer with at least two objects.
Partition every source object exactly once, retain its original order, and create at least two
non-empty parts. This operation changes organization, not pixels.""",
    "combine_layers": """Use combine_layers only. Choose at least two unlocked adjacent layers in
their exact paint order. Give the merged layer a new unique ID and a clear label. This operation
changes organization, not pixels.""",
    "add_objects": """Use add_objects only. Add 1-12 small primitives that materially improve the
match to the reference. Insert them into one existing unlocked layer. Use primitive renderer nodes,
not groups, motifs, bitmaps, or large background shapes. Check coordinates, palette, opacity, span,
and stroke width carefully.""",
    "remove_objects": """Use remove_objects only. Remove 1-12 existing objects whose visible marks
weaken the match to the reference. Use exact object IDs from the context and do not remove all
objects from the composition.""",
}


def load_crusoe_api_key(env_file: str | Path | None = None) -> str:
    """Load a Crusoe key from the environment or a gitignored .env file.

    The repository's historical .env is a single bare token; normal KEY=value files are
    accepted too. The key is returned only in memory and is never placed in inference logs.
    """
    for name in ("CRUSOE_API_TOKEN", "CRUSOE_API_KEY", "API_TOKEN"):
        value = os.environ.get(name, "").strip()
        if value:
            return value

    if env_file is not None:
        candidates = [Path(env_file)]
    else:
        repo_root = Path(__file__).resolve().parents[2]
        candidates = [Path.cwd() / ".env", Path.cwd().parent / ".env", repo_root / ".env"]
    checked: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate in checked or not candidate.is_file():
            continue
        checked.add(candidate)
        bare_values: list[str] = []
        assignments: dict[str, str] = {}
        for source_line in candidate.read_text().splitlines():
            line = source_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                bare_values.append(line.strip("'\""))
                continue
            name, value = line.split("=", 1)
            assignments[name.strip()] = value.strip().strip("'\"")
        for name in ("CRUSOE_API_TOKEN", "CRUSOE_API_KEY", "API_TOKEN"):
            if assignments.get(name):
                return assignments[name]
        if len(bare_values) == 1 and bare_values[0]:
            return bare_values[0]
    raise NemotronInferenceError(
        "Crusoe API key not found; set CRUSOE_API_TOKEN or provide a .env file"
    )


def _data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _comparison_sheet_data_url(paths: list[Path], labels: list[str]) -> str:
    if len(paths) != len(labels) or not paths:
        raise NemotronInferenceError("comparison sheet paths and labels must be non-empty and aligned")
    panel_width = 640
    panel_height = 360
    label_height = 40
    sheet = Image.new("RGB", (panel_width * len(paths), panel_height + label_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=18)
    for index, (path, label) in enumerate(zip(paths, labels)):
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail((panel_width, panel_height), Image.Resampling.LANCZOS)
            x = index * panel_width + (panel_width - image.width) // 2
            y = label_height + (panel_height - image.height) // 2
            sheet.paste(image, (x, y))
        draw.rectangle(
            (index * panel_width, 0, (index + 1) * panel_width, label_height), fill="white"
        )
        draw.text((index * panel_width + 12, 10), label, fill="black", font=font)
        if index:
            draw.line((index * panel_width, 0, index * panel_width, sheet.height), fill="black", width=2)
    buffer = BytesIO()
    sheet.save(buffer, format="JPEG", quality=92, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _proposal_shape(skill: str, *, agent_id: str, mutation_id: str, revision: int) -> dict:
    operation: dict[str, Any]
    if skill == "split_layers":
        operation = {
            "source_layer_id": "existing-layer-id",
            "parts": [
                {"layer_id": "new-layer-a", "label": "...", "object_ids": ["obj-..."]},
                {"layer_id": "new-layer-b", "label": "...", "object_ids": ["obj-..."]},
            ],
        }
    elif skill == "combine_layers":
        operation = {
            "source_layer_ids": ["adjacent-layer-a", "adjacent-layer-b"],
            "output_layer_id": "new-merged-layer-id",
            "output_label": "...",
        }
    elif skill == "add_objects":
        operation = {
            "layer_id": "existing-layer-id",
            "before_object_id": "",
            "objects": [{
                "object_id": "unique-object-id",
                "renderer_node": {
                    "type": "ellipse", "cx": 10, "cy": 10, "rx": 5, "ry": 5,
                    "fill": "#00ffff", "stroke": "none", "width": 0, "opacity": 0.5,
                },
                "intent": "...",
                "tags": ["type:ellipse", "fill:#00ffff"],
            }],
        }
    else:
        operation = {"object_ids": ["existing-object-id"]}
    return {
        "protocol_version": PROTOCOL_VERSION,
        "agent_id": agent_id,
        "mutation_id": mutation_id,
        "base_revision": revision,
        "critique": "visual diagnosis of the current render",
        "rationale": "why this single operation improves the picture",
        skill: operation,
    }


def build_prompts(
    context: dict,
    skill: str,
    *,
    agent_id: str,
    mutation_id: str,
    assignment: dict | None = None,
    plan_id: str | None = None,
) -> tuple[str, str]:
    if skill not in SKILL_GUIDANCE:
        raise NemotronInferenceError(f"unknown picture-agent skill {skill!r}")
    validate_context(context)
    expected = _proposal_shape(
        skill, agent_id=agent_id, mutation_id=mutation_id, revision=context["revision"]
    )
    sections = [
        f"You are the {skill} specialist at revision {context['revision']}.",
        SKILL_GUIDANCE[skill],
        "The first image is the current constrained render. Later images are visual references.",
        "Your output must have these exact protocol_version, agent_id, mutation_id, and "
        "base_revision values, with exactly the requested skill key.",
        "Required output shape (replace placeholders with valid values):\n"
        + json.dumps(expected, separators=(",", ":"), ensure_ascii=False),
        "Complete PictureContext JSON:\n"
        + json.dumps(context, separators=(",", ":"), ensure_ascii=False, allow_nan=False),
    ]
    if assignment is not None:
        if skill != "add_objects":
            raise NemotronInferenceError("regional delegations require the add_objects skill")
        object_prefix = assignment_object_prefix(context, assignment)
        sections.insert(2, "\n".join([
            f"You are executing planner assignment {assignment['assignment_id']!r} from plan "
            f"{plan_id or 'unknown'!r}.",
            "Obey every assignment field exactly: target only its layer, keep every primitive fully "
            "inside its region, use only its shape_types and colors, and add between minimum_shapes "
            "and shape_budget objects. Paths must set smooth=false so regional bounds remain exact.",
            f"Every new object_id must be unique and start with {object_prefix!r}; use consecutive "
            "two-digit suffixes such as 01, 02, and 03.",
            "Primitive schemas are strict: a path may contain type, points, smooth=false, closed, "
            "fill, stroke, width, and opacity; a polygon may contain only type, points, fill, "
            "stroke, width, and opacity (never smooth or closed); an ellipse may contain only "
            "type, cx, cy, rx, ry, fill, stroke, width, and opacity.",
            "Delegated assignment JSON:\n"
            + json.dumps(assignment, separators=(",", ":"), ensure_ascii=False, allow_nan=False),
        ]))
    user_prompt = "\n\n".join(sections)
    return SYSTEM_PROMPT, user_prompt


def build_planner_prompts(
    context: dict,
    *,
    planner_id: str,
    plan_id: str,
    minimum_total_shapes: int = 80,
) -> tuple[str, str]:
    rendered = validate_context(context)
    current_shapes = rendered["checks"]["metrics"]["expanded_shapes"]
    remaining = MAX_EXPANDED_SHAPES - current_shapes
    required_total = min(minimum_total_shapes, remaining)
    expected = {
        "protocol_version": PLAN_PROTOCOL_VERSION,
        "planner_id": planner_id,
        "plan_id": plan_id,
        "base_revision": context["revision"],
        "base_state_sha256": context["state_sha256"],
        "critique": "visual diagnosis of the current render",
        "strategy": "ordered regional refinement strategy",
        "assignments": [
            {
                "assignment_id": "unique-part-id",
                "part_label": "descriptive picture part",
                "brief": "specific marks and visual purpose",
                "region": {"x": 0, "y": 0, "width": 96, "height": 96},
                "layer_id": "existing-unlocked-layer-id",
                "shape_types": ["ellipse", "path"],
                "colors": ["#00ffff", "#000000"],
                "minimum_shapes": 16,
                "shape_budget": 24,
            },
            {
                "assignment_id": "second-part-id",
                "part_label": "another picture part",
                "brief": "specific marks and visual purpose",
                "region": {"x": 96, "y": 96, "width": 96, "height": 96},
                "layer_id": "existing-unlocked-layer-id",
                "shape_types": ["polygon"],
                "colors": ["#ffff00", "#ff00ff"],
                "minimum_shapes": 16,
                "shape_budget": 24,
            },
        ],
    }
    user_prompt = "\n\n".join([
        f"Plan revision {context['revision']} of run {context['run_id']!r}.",
        f"The render currently has {current_shapes} expanded shapes and {remaining} slots remain "
        f"under the hard {MAX_EXPANDED_SHAPES:,}-shape ceiling.",
        f"Create 5-{min(6, MAX_ASSIGNMENTS)} ordered assignments. Give each assignment a focused "
        f"part of the image, 12-24 minimum shapes, and no more than {MAX_SHAPES_PER_ASSIGNMENT} "
        f"budgeted shapes. The sum of minimum_shapes must be at least {required_total}. Prefer "
        "non-overlapping regions and distribute work across meaningful "
        "parts such as mountains, water, granite, shoreline, and pines where appropriate.",
        "The first image is the current constrained render. Later images are visual references.",
        "Use the exact protocol_version, planner_id, plan_id, base_revision, and base_state_sha256 "
        "values shown below. Layer IDs must come from the context.",
        "Required plan shape (replace placeholders and add assignments):\n"
        + json.dumps(expected, separators=(",", ":"), ensure_ascii=False),
        "Complete PictureContext JSON:\n"
        + json.dumps(context, separators=(",", ":"), ensure_ascii=False, allow_nan=False),
    ])
    return PLANNER_SYSTEM_PROMPT, user_prompt


def _response_text(message: dict) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "") for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
    return ""


def parse_proposal(text: str) -> dict:
    """Extract the first protocol proposal object from a model response."""
    candidates = [text.strip()]
    candidates.extend(
        match.group(1).strip()
        for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    )
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
        for match in re.finditer(r"\{", candidate):
            try:
                value, _ = decoder.raw_decode(candidate[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and "protocol_version" in value:
                return value
    raise NemotronInferenceError("Nemotron response did not contain a JSON proposal")


def _call_crusoe(api_key: str, body: dict, timeout: float) -> dict:
    request = Request(
        INFERENCE_ENDPOINT,
        data=json.dumps(body, separators=(",", ":")).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        tls_context = ssl.create_default_context(cafile=certifi.where())
        with urlopen(request, timeout=timeout, context=tls_context) as response:
            return json.loads(response.read())
    except HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:2_000]
        raise NemotronInferenceError(f"Crusoe returned HTTP {error.code}: {detail}") from error
    except (URLError, TimeoutError, json.JSONDecodeError) as error:
        raise NemotronInferenceError(f"Crusoe inference failed: {error}") from error


def infer_proposal(
    context: dict,
    skill: str,
    *,
    api_key: str,
    current_image: str | Path,
    reference_images: list[str | Path] | None = None,
    agent_id: str | None = None,
    mutation_id: str | None = None,
    assignment: dict | None = None,
    plan_id: str | None = None,
    max_attempts: int = 3,
    max_tokens: int = 8_192,
    timeout: float = 180,
) -> tuple[dict, dict]:
    """Run a Nemotron agent until one proposal passes the atomic validation gate."""
    validate_context(context)
    if not 1 <= max_attempts <= 5:
        raise NemotronInferenceError("max_attempts must be between 1 and 5")
    revision = context["revision"]
    agent_id = agent_id or f"nemotron-{skill.replace('_', '-')}-r{revision + 1:03d}"
    mutation_id = mutation_id or f"{context['run_id']}-{skill}-r{revision + 1:03d}"
    system_prompt, user_prompt = build_prompts(
        context, skill, agent_id=agent_id, mutation_id=mutation_id,
        assignment=assignment, plan_id=plan_id,
    )

    current_path = Path(current_image).resolve()
    reference_paths = [Path(value).resolve() for value in (reference_images or [])]
    for path in [current_path, *reference_paths]:
        if not path.is_file():
            raise NemotronInferenceError(f"image asset not found: {path}")
    image_assets = [
        {"role": "current_render", "path": str(current_path)},
        *({"role": "visual_reference", "path": str(path)} for path in reference_paths),
    ]
    attempts = []
    correction = ""
    for attempt_number in range(1, max_attempts + 1):
        attempt_prompt = user_prompt + correction
        content: list[dict] = [{"type": "text", "text": attempt_prompt}]
        content.append({"type": "image_url", "image_url": {"url": _data_url(current_path)}})
        for path in reference_paths:
            content.append({"type": "image_url", "image_url": {"url": _data_url(path)}})
        body = {
            "model": GENERATOR_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        started = time.monotonic()
        response = _call_crusoe(api_key, body, timeout)
        elapsed = round(time.monotonic() - started, 3)
        choices = response.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            raise NemotronInferenceError("Crusoe response contains no completion choice")
        choice = choices[0]
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        content_text = _response_text(message)
        reasoning_text = message.get("reasoning_content", "")
        if not isinstance(reasoning_text, str):
            reasoning_text = ""
        attempt_log = {
            "attempt": attempt_number,
            "system_prompt": system_prompt,
            "user_prompt": attempt_prompt,
            "image_assets": deepcopy(image_assets),
            "response": {
                "id": response.get("id"),
                "created": response.get("created"),
                "model": response.get("model"),
                "finish_reason": choice.get("finish_reason"),
                "usage": deepcopy(response.get("usage", {})),
                "content": content_text,
                "reasoning_content": reasoning_text,
                "elapsed_seconds": elapsed,
            },
        }
        try:
            proposal = parse_proposal(content_text or reasoning_text)
            actual_skill, _ = proposal_skill(proposal)
            if actual_skill != skill:
                raise PictureContextError(
                    f"agent returned {actual_skill!r}; requested skill was {skill!r}"
                )
            if proposal.get("agent_id") != agent_id:
                raise PictureContextError("agent_id does not match the assigned agent")
            if proposal.get("mutation_id") != mutation_id:
                raise PictureContextError("mutation_id does not match the assigned mutation")
            apply_proposal(context, proposal)
            if assignment is not None:
                validate_delegated_proposal(context, proposal, assignment)
        except (NemotronInferenceError, PictureContextError, PicturePlanError) as error:
            attempt_log["validation_error"] = str(error)
            attempts.append(attempt_log)
            correction = (
                "\n\nYour prior response was rejected by the deterministic validator: "
                f"{error}. Return a corrected JSON object only."
            )
            continue
        attempt_log["accepted_proposal"] = deepcopy(proposal)
        attempts.append(attempt_log)
        log = {
            "provider": INFERENCE_PROVIDER,
            "endpoint": INFERENCE_ENDPOINT,
            "model": GENERATOR_MODEL,
            "protocol_version": PROTOCOL_VERSION,
            "run_id": context["run_id"],
            "base_revision": revision,
            "skill": skill,
            "agent_id": agent_id,
            "mutation_id": mutation_id,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "attempts": attempts,
            "accepted_attempt": attempt_number,
        }
        if assignment is not None:
            log["plan_id"] = plan_id
            log["assignment"] = deepcopy(assignment)
        return proposal, log
    errors = "; ".join(item.get("validation_error", "unknown error") for item in attempts)
    failed_log = {
        "provider": INFERENCE_PROVIDER,
        "endpoint": INFERENCE_ENDPOINT,
        "model": GENERATOR_MODEL,
        "protocol_version": PROTOCOL_VERSION,
        "run_id": context["run_id"],
        "base_revision": revision,
        "skill": skill,
        "agent_id": agent_id,
        "mutation_id": mutation_id,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "attempts": attempts,
        "accepted_attempt": None,
        "status": "rejected",
    }
    if assignment is not None:
        failed_log["plan_id"] = plan_id
        failed_log["assignment"] = deepcopy(assignment)
    raise NemotronInferenceError(
        f"Nemotron did not produce a valid {skill} proposal after {max_attempts} attempts: {errors}",
        log=failed_log,
    )


def infer_plan(
    context: dict,
    *,
    api_key: str,
    current_image: str | Path,
    reference_images: list[str | Path] | None = None,
    planner_id: str | None = None,
    plan_id: str | None = None,
    max_attempts: int = 3,
    max_tokens: int = 8_192,
    timeout: float = 180,
    minimum_total_shapes: int = 80,
) -> tuple[dict, dict]:
    """Run the Nemotron planner and return a validated regional delegation plan."""
    validate_context(context)
    if not 1 <= max_attempts <= 5:
        raise NemotronInferenceError("max_attempts must be between 1 and 5")
    revision = context["revision"]
    planner_id = planner_id or f"nemotron-planner-r{revision:03d}"
    plan_id = plan_id or f"{context['run_id']}-plan-r{revision:03d}"
    system_prompt, user_prompt = build_planner_prompts(
        context, planner_id=planner_id, plan_id=plan_id,
        minimum_total_shapes=minimum_total_shapes,
    )
    current_shapes = validate_context(context)["checks"]["metrics"]["expanded_shapes"]
    required_total = min(minimum_total_shapes, MAX_EXPANDED_SHAPES - current_shapes)
    current_path = Path(current_image).resolve()
    reference_paths = [Path(value).resolve() for value in (reference_images or [])]
    for path in [current_path, *reference_paths]:
        if not path.is_file():
            raise NemotronInferenceError(f"image asset not found: {path}")
    image_assets = [
        {"role": "current_render", "path": str(current_path)},
        *({"role": "visual_reference", "path": str(path)} for path in reference_paths),
    ]
    comparison_paths = [current_path, *reference_paths]
    comparison_labels = ["CURRENT RENDER", *(
        f"REFERENCE {index + 1}" for index in range(len(reference_paths))
    )]
    comparison_data_url = _comparison_sheet_data_url(comparison_paths, comparison_labels)

    attempts = []
    correction = ""
    for attempt_number in range(1, max_attempts + 1):
        attempt_prompt = user_prompt + correction
        content: list[dict] = [{"type": "text", "text": attempt_prompt}]
        content.append({"type": "image_url", "image_url": {"url": comparison_data_url}})
        body = {
            "model": EVALUATOR_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        started = time.monotonic()
        response = _call_crusoe(api_key, body, timeout)
        elapsed = round(time.monotonic() - started, 3)
        choices = response.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            raise NemotronInferenceError("Crusoe response contains no completion choice")
        choice = choices[0]
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        content_text = _response_text(message)
        reasoning_text = message.get("reasoning_content", "")
        if not isinstance(reasoning_text, str):
            reasoning_text = ""
        attempt_log = {
            "attempt": attempt_number,
            "system_prompt": system_prompt,
            "user_prompt": attempt_prompt,
            "image_assets": deepcopy(image_assets),
            "comparison_sheet": {"labels": comparison_labels, "layout": "horizontal"},
            "response": {
                "id": response.get("id"),
                "created": response.get("created"),
                "model": response.get("model"),
                "finish_reason": choice.get("finish_reason"),
                "usage": deepcopy(response.get("usage", {})),
                "content": content_text,
                "reasoning_content": reasoning_text,
                "elapsed_seconds": elapsed,
            },
        }
        try:
            plan = parse_proposal(content_text or reasoning_text)
            if plan.get("planner_id") != planner_id:
                raise PicturePlanError("planner_id does not match the assigned planner")
            if plan.get("plan_id") != plan_id:
                raise PicturePlanError("plan_id does not match the assigned plan")
            plan = validate_plan(context, plan)
            planned_minimum = sum(item["minimum_shapes"] for item in plan["assignments"])
            if planned_minimum < required_total:
                raise PicturePlanError(
                    f"plan minimum {planned_minimum} is below required total {required_total}"
                )
        except (NemotronInferenceError, PictureContextError, PicturePlanError) as error:
            attempt_log["validation_error"] = str(error)
            attempts.append(attempt_log)
            correction = (
                "\n\nYour prior plan was rejected by the deterministic validator: "
                f"{error}. Return a corrected PicturePlan JSON object only."
            )
            continue
        attempt_log["accepted_plan"] = deepcopy(plan)
        attempts.append(attempt_log)
        log = {
            "provider": INFERENCE_PROVIDER,
            "endpoint": INFERENCE_ENDPOINT,
            "model": EVALUATOR_MODEL,
            "protocol_version": PLAN_PROTOCOL_VERSION,
            "run_id": context["run_id"],
            "base_revision": revision,
            "planner_id": planner_id,
            "plan_id": plan_id,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "attempts": attempts,
            "accepted_attempt": attempt_number,
        }
        return plan, log
    errors = "; ".join(item.get("validation_error", "unknown error") for item in attempts)
    failed_log = {
        "provider": INFERENCE_PROVIDER,
        "endpoint": INFERENCE_ENDPOINT,
        "model": EVALUATOR_MODEL,
        "protocol_version": PLAN_PROTOCOL_VERSION,
        "run_id": context["run_id"],
        "base_revision": revision,
        "planner_id": planner_id,
        "plan_id": plan_id,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "attempts": attempts,
        "accepted_attempt": None,
        "status": "rejected",
    }
    raise NemotronInferenceError(
        f"Nemotron did not produce a valid plan after {max_attempts} attempts: {errors}",
        log=failed_log,
    )


def infer_evaluation(
    base_context: dict,
    final_context: dict,
    plan: dict,
    *,
    api_key: str,
    baseline_image: str | Path,
    final_image: str | Path,
    reference_images: list[str | Path] | None = None,
    evaluator_id: str | None = None,
    evaluation_id: str | None = None,
    max_attempts: int = 3,
    max_tokens: int = 4_096,
    timeout: float = 180,
) -> tuple[dict, dict]:
    """Use Nemotron Super VL to compare the planned final render with its baseline."""
    validate_context(base_context)
    validate_context(final_context)
    if base_context["run_id"] != final_context["run_id"]:
        raise NemotronInferenceError("evaluation contexts must belong to the same run")
    if not 1 <= max_attempts <= 5:
        raise NemotronInferenceError("max_attempts must be between 1 and 5")
    run_id = final_context["run_id"]
    base_revision = base_context["revision"]
    final_revision = final_context["revision"]
    evaluator_id = evaluator_id or f"nemotron-super-vl-evaluator-r{final_revision:03d}"
    evaluation_id = evaluation_id or f"{run_id[:40]}-evaluation-r{final_revision:03d}"
    expected = {
        "protocol_version": EVALUATION_PROTOCOL_VERSION,
        "evaluator_id": evaluator_id,
        "evaluation_id": evaluation_id,
        "run_id": run_id,
        "base_revision": base_revision,
        "final_revision": final_revision,
        "reference_match_before": 0,
        "reference_match_after": 0,
        "composition_before": 0,
        "composition_after": 0,
        "improved": False,
        "summary": "concise comparison",
        "strengths": ["specific strength"],
        "issues": ["specific remaining issue"],
        "next_actions": ["specific next refinement"],
    }
    user_prompt = "\n\n".join([
        "Image order: (1) baseline constrained render, (2) final planned render, then "
        "(3+) photographic visual references.",
        "Evaluate the final result against both the baseline and reference. Do not reward raw shape "
        "count; reward better placement, silhouette, depth, color mixing, and subject readability.",
        "Use these exact identity and revision values in the response.",
        "Required PictureEvaluation JSON:\n"
        + json.dumps(expected, separators=(",", ":"), ensure_ascii=False),
        "Original prompt set:\n"
        + json.dumps(base_context["prompt_set"], separators=(",", ":"), ensure_ascii=False),
        "Composition plan:\n"
        + json.dumps(base_context["composition"], separators=(",", ":"), ensure_ascii=False),
        "Delegation plan executed:\n"
        + json.dumps(plan, separators=(",", ":"), ensure_ascii=False),
    ])
    baseline_path = Path(baseline_image).resolve()
    final_path = Path(final_image).resolve()
    reference_paths = [Path(value).resolve() for value in (reference_images or [])]
    for path in [baseline_path, final_path, *reference_paths]:
        if not path.is_file():
            raise NemotronInferenceError(f"image asset not found: {path}")
    image_assets = [
        {"role": "baseline_render", "path": str(baseline_path)},
        {"role": "final_render", "path": str(final_path)},
        *({"role": "visual_reference", "path": str(path)} for path in reference_paths),
    ]
    comparison_paths = [baseline_path, final_path, *reference_paths]
    comparison_labels = ["BASELINE", "FINAL", *(
        f"REFERENCE {index + 1}" for index in range(len(reference_paths))
    )]
    comparison_data_url = _comparison_sheet_data_url(comparison_paths, comparison_labels)

    attempts = []
    correction = ""
    for attempt_number in range(1, max_attempts + 1):
        attempt_prompt = user_prompt + correction
        content: list[dict] = [{"type": "text", "text": attempt_prompt}]
        content.append({"type": "image_url", "image_url": {"url": comparison_data_url}})
        body = {
            "model": EVALUATOR_MODEL,
            "messages": [
                {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            "temperature": 0.1,
            "max_tokens": max_tokens,
        }
        started = time.monotonic()
        response = _call_crusoe(api_key, body, timeout)
        elapsed = round(time.monotonic() - started, 3)
        choices = response.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            raise NemotronInferenceError("Crusoe response contains no completion choice")
        choice = choices[0]
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        content_text = _response_text(message)
        reasoning_text = message.get("reasoning_content", "")
        if not isinstance(reasoning_text, str):
            reasoning_text = ""
        attempt_log = {
            "attempt": attempt_number,
            "system_prompt": EVALUATOR_SYSTEM_PROMPT,
            "user_prompt": attempt_prompt,
            "image_assets": deepcopy(image_assets),
            "comparison_sheet": {"labels": comparison_labels, "layout": "horizontal"},
            "response": {
                "id": response.get("id"),
                "created": response.get("created"),
                "model": response.get("model"),
                "finish_reason": choice.get("finish_reason"),
                "usage": deepcopy(response.get("usage", {})),
                "content": content_text,
                "reasoning_content": reasoning_text,
                "elapsed_seconds": elapsed,
            },
        }
        try:
            evaluation = parse_proposal(content_text or reasoning_text)
            evaluation = validate_evaluation(
                evaluation,
                run_id=run_id,
                base_revision=base_revision,
                final_revision=final_revision,
                evaluator_id=evaluator_id,
                evaluation_id=evaluation_id,
            )
        except (NemotronInferenceError, PictureEvaluationError) as error:
            attempt_log["validation_error"] = str(error)
            attempts.append(attempt_log)
            correction = (
                "\n\nYour prior evaluation was rejected by the deterministic validator: "
                f"{error}. Return a corrected PictureEvaluation JSON object only."
            )
            continue
        attempt_log["accepted_evaluation"] = deepcopy(evaluation)
        attempts.append(attempt_log)
        log = {
            "provider": INFERENCE_PROVIDER,
            "endpoint": INFERENCE_ENDPOINT,
            "model": EVALUATOR_MODEL,
            "protocol_version": EVALUATION_PROTOCOL_VERSION,
            "run_id": run_id,
            "base_revision": base_revision,
            "final_revision": final_revision,
            "evaluator_id": evaluator_id,
            "evaluation_id": evaluation_id,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "attempts": attempts,
            "accepted_attempt": attempt_number,
        }
        return evaluation, log
    errors = "; ".join(item.get("validation_error", "unknown error") for item in attempts)
    failed_log = {
        "provider": INFERENCE_PROVIDER,
        "endpoint": INFERENCE_ENDPOINT,
        "model": EVALUATOR_MODEL,
        "protocol_version": EVALUATION_PROTOCOL_VERSION,
        "run_id": run_id,
        "base_revision": base_revision,
        "final_revision": final_revision,
        "evaluator_id": evaluator_id,
        "evaluation_id": evaluation_id,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "attempts": attempts,
        "accepted_attempt": None,
        "status": "rejected",
    }
    raise NemotronInferenceError(
        f"Nemotron evaluator did not produce a valid verdict after {max_attempts} attempts: {errors}",
        log=failed_log,
    )


def write_planner_artifacts(
    plan: dict,
    log: dict,
    *,
    root: str | Path = "recordings",
) -> tuple[Path, Path]:
    folder = Path(root) / log["run_id"] / "handoffs"
    folder.mkdir(parents=True, exist_ok=True)
    plan_path = folder / "plan.json"
    log_path = folder / "planner-inference.json"
    plan_path.write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")
    log_path.write_text(json.dumps(log, indent=2, allow_nan=False) + "\n")
    return plan_path, log_path


def write_evaluation_artifacts(
    evaluation: dict,
    log: dict,
    *,
    root: str | Path = "recordings",
) -> tuple[Path, Path]:
    folder = Path(root) / log["run_id"] / "handoffs"
    folder.mkdir(parents=True, exist_ok=True)
    evaluation_path = folder / "evaluation.json"
    log_path = folder / "evaluator-inference.json"
    evaluation_path.write_text(json.dumps(evaluation, indent=2, allow_nan=False) + "\n")
    log_path.write_text(json.dumps(log, indent=2, allow_nan=False) + "\n")
    return evaluation_path, log_path


def planner_provenance(log: dict) -> dict:
    accepted = log["attempts"][log["accepted_attempt"] - 1]["response"]
    return {
        "plan": "handoffs/plan.json",
        "planner_inference": "handoffs/planner-inference.json",
        "planner_id": log["planner_id"],
        "plan_id": log["plan_id"],
        "model": EVALUATOR_MODEL,
        "response_id": accepted.get("id"),
        "usage": deepcopy(accepted.get("usage", {})),
        "accepted_attempt": log["accepted_attempt"],
    }


def evaluator_provenance(log: dict) -> dict:
    accepted = log["attempts"][log["accepted_attempt"] - 1]["response"]
    return {
        "evaluation": "handoffs/evaluation.json",
        "evaluator_inference": "handoffs/evaluator-inference.json",
        "evaluator_id": log["evaluator_id"],
        "evaluation_id": log["evaluation_id"],
        "model": EVALUATOR_MODEL,
        "response_id": accepted.get("id"),
        "usage": deepcopy(accepted.get("usage", {})),
        "accepted_attempt": log["accepted_attempt"],
    }


def write_inference_log(log: dict, *, root: str | Path = "recordings") -> Path:
    run_id = log["run_id"]
    revision = int(log["base_revision"]) + 1
    folder = Path(root) / run_id / "handoffs"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"inference-{revision:03d}.json"
    path.write_text(json.dumps(log, indent=2, allow_nan=False) + "\n")
    return path


def write_rejected_inference_log(log: dict, *, root: str | Path = "recordings") -> Path:
    folder = Path(root) / log["run_id"] / "handoffs"
    folder.mkdir(parents=True, exist_ok=True)
    if "skill" in log:
        revision = int(log["base_revision"]) + 1
        prefix = f"inference-{revision:03d}-rejected-"
    elif "evaluation_id" in log:
        prefix = "evaluator-inference-rejected-"
    else:
        prefix = "planner-inference-rejected-"
    index = len(list(folder.glob(prefix + "*.json"))) + 1
    path = folder / f"{prefix}{index:03d}.json"
    path.write_text(json.dumps(log, indent=2, allow_nan=False) + "\n")
    return path


def inference_provenance(log: dict) -> dict:
    accepted = log["attempts"][log["accepted_attempt"] - 1]["response"]
    provenance = {
        "inference": f"handoffs/inference-{int(log['base_revision']) + 1:03d}.json",
        "response_id": accepted.get("id"),
        "usage": deepcopy(accepted.get("usage", {})),
        "accepted_attempt": log["accepted_attempt"],
    }
    if "assignment" in log:
        provenance["plan_id"] = log.get("plan_id")
        provenance["assignment_id"] = log["assignment"]["assignment_id"]
        if log["assignment"].get("parent_assignment_id"):
            provenance["parent_assignment_id"] = log["assignment"]["parent_assignment_id"]
        provenance["part_label"] = log["assignment"]["part_label"]
        provenance["region"] = deepcopy(log["assignment"]["region"])
        provenance["shape_budget"] = log["assignment"]["shape_budget"]
    return provenance


__all__ = [
    "EVALUATOR_SYSTEM_PROMPT", "NemotronInferenceError", "PLANNER_SYSTEM_PROMPT",
    "SYSTEM_PROMPT", "SKILL_GUIDANCE", "build_planner_prompts", "build_prompts",
    "evaluator_provenance", "infer_evaluation", "infer_plan", "infer_proposal",
    "inference_provenance", "load_crusoe_api_key", "parse_proposal", "planner_provenance",
    "write_evaluation_artifacts", "write_inference_log", "write_planner_artifacts",
    "write_rejected_inference_log",
]
