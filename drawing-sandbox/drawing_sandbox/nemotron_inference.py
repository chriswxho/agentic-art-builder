"""Crusoe Managed Inference client for constrained Nemotron picture agents."""

from __future__ import annotations

import base64
from copy import deepcopy
from datetime import datetime, timezone
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

from .iteration_context import (
    INFERENCE_ENDPOINT,
    INFERENCE_MODEL,
    INFERENCE_PROVIDER,
    PROTOCOL_VERSION,
    PictureContextError,
    apply_proposal,
    proposal_skill,
    validate_context,
)


class NemotronInferenceError(RuntimeError):
    """Raised when Crusoe inference cannot produce a valid agent proposal."""


SYSTEM_PROMPT = """You are one picture-editing agent in a consecutive-agent drawing workflow.
You receive a complete typed PictureContext plus the current render and visual reference.
Return exactly one JSON object and nothing else. Do not use Markdown fences.
Never rewrite the scene or context. Use only the requested typed skill operation.
Preserve existing stable IDs and invent unique descriptive IDs only where the skill permits.
All new visible primitives must be paths, polygons, or ellipses using only #00ffff, #ff00ff,
#ffff00, or #000000 on white. Every visible primitive must have opacity 0.5, transformed span
at most 96 canvas units, and stroke width at most 12. Prefer a few precise small marks.
The orchestrator will reject stale, malformed, or policy-violating output atomically."""


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
) -> tuple[str, str]:
    if skill not in SKILL_GUIDANCE:
        raise NemotronInferenceError(f"unknown picture-agent skill {skill!r}")
    validate_context(context)
    expected = _proposal_shape(
        skill, agent_id=agent_id, mutation_id=mutation_id, revision=context["revision"]
    )
    user_prompt = "\n\n".join([
        f"You are the {skill} specialist at revision {context['revision']}.",
        SKILL_GUIDANCE[skill],
        "The first image is the current constrained render. Later images are visual references.",
        "Your output must have these exact protocol_version, agent_id, mutation_id, and "
        "base_revision values, with exactly the requested skill key.",
        "Required output shape (replace placeholders with valid values):\n"
        + json.dumps(expected, separators=(",", ":"), ensure_ascii=False),
        "Complete PictureContext JSON:\n"
        + json.dumps(context, separators=(",", ":"), ensure_ascii=False, allow_nan=False),
    ])
    return SYSTEM_PROMPT, user_prompt


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
        context, skill, agent_id=agent_id, mutation_id=mutation_id
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
            "model": INFERENCE_MODEL,
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
        except (NemotronInferenceError, PictureContextError) as error:
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
            "model": INFERENCE_MODEL,
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
        return proposal, log
    errors = "; ".join(item.get("validation_error", "unknown error") for item in attempts)
    raise NemotronInferenceError(
        f"Nemotron did not produce a valid {skill} proposal after {max_attempts} attempts: {errors}"
    )


def write_inference_log(log: dict, *, root: str | Path = "recordings") -> Path:
    run_id = log["run_id"]
    revision = int(log["base_revision"]) + 1
    folder = Path(root) / run_id / "handoffs"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"inference-{revision:03d}.json"
    path.write_text(json.dumps(log, indent=2, allow_nan=False) + "\n")
    return path


def inference_provenance(log: dict) -> dict:
    accepted = log["attempts"][log["accepted_attempt"] - 1]["response"]
    return {
        "inference": f"handoffs/inference-{int(log['base_revision']) + 1:03d}.json",
        "response_id": accepted.get("id"),
        "usage": deepcopy(accepted.get("usage", {})),
        "accepted_attempt": log["accepted_attempt"],
    }


__all__ = [
    "NemotronInferenceError", "SYSTEM_PROMPT", "SKILL_GUIDANCE", "build_prompts",
    "infer_proposal", "inference_provenance", "load_crusoe_api_key", "parse_proposal",
    "write_inference_log",
]
