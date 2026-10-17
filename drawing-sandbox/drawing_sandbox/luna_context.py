"""Typed JSON handoffs and atomic picture mutations for consecutive Luna agents."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .policy import current_instruction_set
from .recording import record_composition_step
from .server import render_drawing


PROTOCOL_VERSION = "drawing-sandbox.iteration.v1"
ALLOWED_SKILLS = ("split_layers", "combine_layers", "add_objects", "remove_objects")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")


class LunaContextError(ValueError):
    """Raised when a handoff or proposed mutation violates the protocol."""


def _json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise LunaContextError("context must contain finite JSON data") from error


def _text(value: Any, label: str, *, required: bool = True, limit: int = 4_000) -> str:
    if not isinstance(value, str) or (required and not value.strip()) or len(value) > limit:
        qualifier = "non-empty " if required else ""
        raise LunaContextError(f"{label} must be a {qualifier}string of at most {limit} characters")
    return value


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise LunaContextError(f"{label} must be a 1-64 character identifier")
    return value


def _state_payload(context: dict) -> dict:
    return {
        key: context[key]
        for key in (
            "protocol_version", "run_id", "revision", "prompt_set", "composition",
            "instruction_set", "scene_template", "layers", "allowed_skills",
            "references", "last_handoff", "history",
        )
    }


def state_sha256(context: dict) -> str:
    return hashlib.sha256(_json(_state_payload(context)).encode()).hexdigest()


def materialize_scene(context: dict) -> dict:
    scene = deepcopy(context["scene_template"])
    scene["nodes"] = [
        deepcopy(item["renderer_node"])
        for layer in context["layers"]
        for item in layer["objects"]
    ]
    return scene


def _object_tags(node: dict, layer_id: str) -> list[str]:
    tags = [f"layer:{layer_id}", f"type:{node.get('type', 'unknown')}"]
    for key in ("fill", "stroke"):
        value = node.get(key)
        if isinstance(value, str) and value != "none":
            tags.append(f"{key}:{value.lower()}")
    return tags


def create_context(
    scene: dict,
    *,
    run_id: str,
    prompt_set: list[str],
    composition: dict,
    layer_specs: list[dict],
    references: list[dict] | None = None,
) -> dict:
    """Create revision zero from a valid renderer scene and cumulative layer ends."""
    _identifier(run_id, "run_id")
    if not isinstance(prompt_set, list) or not 1 <= len(prompt_set) <= 32:
        raise LunaContextError("prompt_set must contain 1-32 strings")
    for index, prompt in enumerate(prompt_set):
        _text(prompt, f"prompt_set[{index}]", limit=8_000)
    if not isinstance(composition, dict):
        raise LunaContextError("composition must be an object")
    if not isinstance(scene, dict) or not isinstance(scene.get("nodes"), list):
        raise LunaContextError("scene must be a renderer scene with a nodes list")
    if not isinstance(layer_specs, list) or not layer_specs:
        raise LunaContextError("layer_specs must be a non-empty list")

    layers = []
    start = 0
    seen_layers: set[str] = set()
    for spec_index, spec in enumerate(layer_specs):
        if not isinstance(spec, dict):
            raise LunaContextError(f"layer_specs[{spec_index}] must be an object")
        layer_id = _identifier(spec.get("layer_id"), f"layer_specs[{spec_index}].layer_id")
        if layer_id in seen_layers:
            raise LunaContextError(f"duplicate layer_id {layer_id!r}")
        seen_layers.add(layer_id)
        label = _text(spec.get("label"), f"layer_specs[{spec_index}].label", limit=200)
        end = spec.get("end_index")
        if isinstance(end, bool) or not isinstance(end, int) or not start < end <= len(scene["nodes"]):
            raise LunaContextError("layer end indexes must be strictly increasing within scene.nodes")
        objects = []
        for source_index in range(start, end):
            node = deepcopy(scene["nodes"][source_index])
            objects.append({
                "object_id": f"obj-{source_index:04d}",
                "renderer_node": node,
                "intent": label,
                "tags": _object_tags(node, layer_id),
            })
        layers.append({"layer_id": layer_id, "label": label, "locked": False, "objects": objects})
        start = end
    if start != len(scene["nodes"]):
        raise LunaContextError("the last layer end_index must equal the scene node count")

    template = deepcopy(scene)
    template["nodes"] = []
    context = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": run_id,
        "revision": 0,
        "state_sha256": "",
        "prompt_set": deepcopy(prompt_set),
        "composition": deepcopy(composition),
        "instruction_set": current_instruction_set(),
        "scene_template": template,
        "layers": layers,
        "allowed_skills": list(ALLOWED_SKILLS),
        "artifacts": {},
        "references": deepcopy(references or []),
        "last_handoff": {},
        "history": [],
    }
    context["state_sha256"] = state_sha256(context)
    validate_context(context)
    return context


def validate_context(context: Any, *, render: bool = True) -> dict | None:
    """Validate a JSON handoff; optionally return its constrained render."""
    if not isinstance(context, dict):
        raise LunaContextError("context must be an object")
    required = {
        "protocol_version", "run_id", "revision", "state_sha256", "prompt_set",
        "composition", "instruction_set", "scene_template", "layers",
        "allowed_skills", "artifacts", "references", "last_handoff", "history",
    }
    missing = required - set(context)
    if missing:
        raise LunaContextError(f"context is missing fields: {sorted(missing)}")
    if context["protocol_version"] != PROTOCOL_VERSION:
        raise LunaContextError("unsupported protocol_version")
    _identifier(context["run_id"], "run_id")
    if isinstance(context["revision"], bool) or not isinstance(context["revision"], int) or context["revision"] < 0:
        raise LunaContextError("revision must be a non-negative integer")
    if context["allowed_skills"] != list(ALLOWED_SKILLS):
        raise LunaContextError("allowed_skills must match the protocol skill set")
    if context["instruction_set"] != current_instruction_set():
        raise LunaContextError("instruction_set does not match the current enforced policy")
    if not isinstance(context["scene_template"], dict) or context["scene_template"].get("nodes") != []:
        raise LunaContextError("scene_template must be a renderer scene with an empty nodes list")
    if not isinstance(context["layers"], list) or not context["layers"]:
        raise LunaContextError("layers must be a non-empty list")
    layer_ids: set[str] = set()
    object_ids: set[str] = set()
    for layer_index, layer in enumerate(context["layers"]):
        if not isinstance(layer, dict):
            raise LunaContextError(f"layers[{layer_index}] must be an object")
        layer_id = _identifier(layer.get("layer_id"), f"layers[{layer_index}].layer_id")
        if layer_id in layer_ids:
            raise LunaContextError(f"duplicate layer_id {layer_id!r}")
        layer_ids.add(layer_id)
        _text(layer.get("label"), f"layers[{layer_index}].label", limit=200)
        if not isinstance(layer.get("locked"), bool) or not isinstance(layer.get("objects"), list):
            raise LunaContextError(f"layers[{layer_index}] has invalid locked/objects fields")
        for object_index, item in enumerate(layer["objects"]):
            if not isinstance(item, dict):
                raise LunaContextError(f"layers[{layer_index}].objects[{object_index}] must be an object")
            object_id = _identifier(item.get("object_id"), "object_id")
            if object_id in object_ids:
                raise LunaContextError(f"duplicate object_id {object_id!r}")
            object_ids.add(object_id)
            if not isinstance(item.get("renderer_node"), dict):
                raise LunaContextError(f"object {object_id!r} must contain renderer_node")
            _text(item.get("intent", ""), f"object {object_id}.intent", required=False, limit=500)
            tags = item.get("tags", [])
            if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
                raise LunaContextError(f"object {object_id!r} tags must be strings")
    if not isinstance(context["history"], list) or not isinstance(context["last_handoff"], dict):
        raise LunaContextError("history and last_handoff have invalid types")
    if not isinstance(context["artifacts"], dict) or not isinstance(context["references"], list):
        raise LunaContextError("artifacts and references have invalid types")
    expected_hash = state_sha256(context)
    if context["state_sha256"] != expected_hash:
        raise LunaContextError("state_sha256 does not match the handoff contents")
    _json(context)
    if not render:
        return None
    rendered = render_drawing(materialize_scene(context))
    if not rendered["checks"]["valid"]:
        raise LunaContextError("context scene violates the picture policy: " + "; ".join(rendered["checks"]["violations"]))
    return rendered


def _find_layer(context: dict, layer_id: str) -> tuple[int, dict]:
    for index, layer in enumerate(context["layers"]):
        if layer["layer_id"] == layer_id:
            return index, layer
    raise LunaContextError(f"unknown layer_id {layer_id!r}")


def _all_object_ids(context: dict) -> set[str]:
    return {item["object_id"] for layer in context["layers"] for item in layer["objects"]}


def proposal_skill(proposal: dict) -> tuple[str, dict]:
    present = [skill for skill in ALLOWED_SKILLS if skill in proposal and proposal[skill] is not None]
    if len(present) != 1:
        raise LunaContextError("proposal must contain exactly one allowed skill operation")
    payload = proposal[present[0]]
    if not isinstance(payload, dict):
        raise LunaContextError(f"{present[0]} must be an object")
    return present[0], payload


def _normalize_added(item: Any, index: int) -> dict:
    if not isinstance(item, dict):
        raise LunaContextError(f"add_objects.objects[{index}] must be an object")
    allowed = {"object_id", "renderer_node", "intent", "tags"}
    unknown = set(item) - allowed
    if unknown:
        raise LunaContextError(f"add_objects object has unknown fields: {sorted(unknown)}")
    object_id = _identifier(item.get("object_id"), f"add_objects.objects[{index}].object_id")
    node = item.get("renderer_node")
    if not isinstance(node, dict):
        raise LunaContextError(f"add_objects object {object_id!r} must contain renderer_node")
    intent = _text(item.get("intent", ""), f"add_objects object {object_id}.intent", required=False, limit=500)
    tags = item.get("tags", [])
    if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
        raise LunaContextError(f"add_objects object {object_id!r} tags must be strings")
    return {"object_id": object_id, "renderer_node": deepcopy(node), "intent": intent, "tags": list(tags)}


def apply_proposal(context: dict, proposal: Any) -> tuple[dict, dict]:
    """Apply one skill atomically and return the next context plus its render."""
    validate_context(context)
    if not isinstance(proposal, dict):
        raise LunaContextError("proposal must be an object")
    if proposal.get("protocol_version") != PROTOCOL_VERSION:
        raise LunaContextError("proposal protocol_version does not match")
    agent_id = _identifier(proposal.get("agent_id"), "agent_id")
    mutation_id = _identifier(proposal.get("mutation_id"), "mutation_id")
    base_revision = proposal.get("base_revision")
    if isinstance(base_revision, bool) or base_revision != context["revision"]:
        raise LunaContextError("proposal base_revision is stale")
    critique = _text(proposal.get("critique"), "critique")
    rationale = _text(proposal.get("rationale"), "rationale")
    if any(event.get("mutation_id") == mutation_id for event in context["history"]):
        raise LunaContextError(f"duplicate mutation_id {mutation_id!r}")
    skill, operation = proposal_skill(proposal)
    candidate = deepcopy(context)

    if skill == "split_layers":
        source_id = _identifier(operation.get("source_layer_id"), "split_layers.source_layer_id")
        layer_index, source = _find_layer(candidate, source_id)
        if source["locked"]:
            raise LunaContextError(f"layer {source_id!r} is locked")
        parts = operation.get("parts")
        if not isinstance(parts, list) or len(parts) < 2:
            raise LunaContextError("split_layers.parts must contain at least two parts")
        existing_layers = {layer["layer_id"] for layer in candidate["layers"]}
        object_map = {item["object_id"]: item for item in source["objects"]}
        expected_order = list(object_map)
        flattened = []
        replacements = []
        new_layer_ids: set[str] = set()
        for part_index, part in enumerate(parts):
            if not isinstance(part, dict):
                raise LunaContextError(f"split_layers.parts[{part_index}] must be an object")
            new_id = _identifier(part.get("layer_id"), f"split_layers.parts[{part_index}].layer_id")
            if new_id in existing_layers or new_id in new_layer_ids:
                raise LunaContextError(f"split output layer_id {new_id!r} is not unique")
            new_layer_ids.add(new_id)
            label = _text(part.get("label"), f"split_layers.parts[{part_index}].label", limit=200)
            ids = part.get("object_ids")
            if not isinstance(ids, list) or not ids or any(not isinstance(value, str) for value in ids):
                raise LunaContextError("every split part must contain object_ids")
            flattened.extend(ids)
            replacements.append({
                "layer_id": new_id, "label": label, "locked": False,
                "objects": [deepcopy(object_map[value]) for value in ids if value in object_map],
            })
        if flattened != expected_order:
            raise LunaContextError("split parts must exhaustively partition objects in original order")
        candidate["layers"][layer_index:layer_index + 1] = replacements

    elif skill == "combine_layers":
        source_ids = operation.get("source_layer_ids")
        if not isinstance(source_ids, list) or len(source_ids) < 2 or len(source_ids) != len(set(source_ids)):
            raise LunaContextError("combine_layers.source_layer_ids must contain at least two distinct IDs")
        indices = [_find_layer(candidate, value)[0] for value in source_ids]
        if indices != list(range(indices[0], indices[0] + len(indices))):
            raise LunaContextError("combine source layers must be adjacent and listed in paint order")
        sources = [candidate["layers"][index] for index in indices]
        if any(layer["locked"] for layer in sources):
            raise LunaContextError("locked layers cannot be combined")
        output_id = _identifier(operation.get("output_layer_id"), "combine_layers.output_layer_id")
        if output_id in {layer["layer_id"] for layer in candidate["layers"]}:
            raise LunaContextError(f"combine output layer_id {output_id!r} is not unique")
        output_label = _text(operation.get("output_label"), "combine_layers.output_label", limit=200)
        objects = [deepcopy(item) for layer in sources for item in layer["objects"]]
        replacement = {"layer_id": output_id, "label": output_label, "locked": False, "objects": objects}
        candidate["layers"][indices[0]:indices[-1] + 1] = [replacement]

    elif skill == "add_objects":
        layer_id = _identifier(operation.get("layer_id"), "add_objects.layer_id")
        _, layer = _find_layer(candidate, layer_id)
        if layer["locked"]:
            raise LunaContextError(f"layer {layer_id!r} is locked")
        objects = operation.get("objects")
        if not isinstance(objects, list) or not 1 <= len(objects) <= 64:
            raise LunaContextError("add_objects.objects must contain 1-64 objects")
        additions = [_normalize_added(item, index) for index, item in enumerate(objects)]
        ids = [item["object_id"] for item in additions]
        if len(ids) != len(set(ids)) or set(ids) & _all_object_ids(candidate):
            raise LunaContextError("added object IDs must be globally unique")
        before = operation.get("before_object_id", "")
        if not isinstance(before, str):
            raise LunaContextError("before_object_id must be a string")
        if before:
            positions = [index for index, item in enumerate(layer["objects"]) if item["object_id"] == before]
            if not positions:
                raise LunaContextError(f"before_object_id {before!r} is not in target layer")
            insert_at = positions[0]
        else:
            insert_at = len(layer["objects"])
        layer["objects"][insert_at:insert_at] = additions

    else:  # remove_objects
        ids = operation.get("object_ids")
        if not isinstance(ids, list) or not 1 <= len(ids) <= 64 or len(ids) != len(set(ids)):
            raise LunaContextError("remove_objects.object_ids must contain 1-64 distinct IDs")
        existing = _all_object_ids(candidate)
        missing = [value for value in ids if value not in existing]
        if missing:
            raise LunaContextError(f"cannot remove unknown object IDs: {missing}")
        targets = set(ids)
        for layer in candidate["layers"]:
            if layer["locked"] and any(item["object_id"] in targets for item in layer["objects"]):
                raise LunaContextError(f"cannot remove objects from locked layer {layer['layer_id']!r}")
            layer["objects"] = [item for item in layer["objects"] if item["object_id"] not in targets]

    candidate["revision"] += 1
    candidate["last_handoff"] = {
        "agent_id": agent_id,
        "mutation_id": mutation_id,
        "skill": skill,
        "critique": critique,
        "rationale": rationale,
    }
    event = {
        "revision": candidate["revision"],
        **candidate["last_handoff"],
        "operation": deepcopy(operation),
    }
    candidate["history"].append(event)
    candidate["state_sha256"] = state_sha256(candidate)
    rendered = validate_context(candidate)
    event["metrics"] = deepcopy(rendered["checks"]["metrics"])
    candidate["state_sha256"] = state_sha256(candidate)
    validate_context(candidate, render=False)
    return candidate, rendered


def _artifact_refs(run_id: str, step: int) -> dict:
    stem = f"steps/{step:03d}"
    return {
        "scene_json": f"recordings/{run_id}/{stem}-scene.json",
        "svg": f"recordings/{run_id}/{stem}-drawing.svg",
        "png": f"recordings/{run_id}/{stem}-drawing.png",
        "checks": f"recordings/{run_id}/{stem}-checks.json",
        "manifest": f"recordings/{run_id}/manifest.json",
    }


def _write_handoff(root: Path, context: dict, proposal: dict | None = None) -> Path:
    handoffs = root / context["run_id"] / "handoffs"
    handoffs.mkdir(parents=True, exist_ok=True)
    revision = context["revision"]
    if proposal is not None:
        (handoffs / f"proposal-{revision:03d}.json").write_text(
            json.dumps(proposal, indent=2, allow_nan=False) + "\n"
        )
    context_path = handoffs / f"context-{revision:03d}.json"
    context_path.write_text(json.dumps(context, indent=2, allow_nan=False) + "\n")
    return context_path


def initialize_recorded_context(context: dict, *, root: str | Path = "recordings") -> Path:
    rendered = validate_context(context)
    record = record_composition_step(
        materialize_scene(context), rendered,
        run_id=context["run_id"], prompt_set=context["prompt_set"],
        composition=context["composition"], step_label="Initial Luna handoff",
        provenance={"protocol_version": PROTOCOL_VERSION, "context_revision": 0,
                    "context": "handoffs/context-000.json"},
        root=root,
    )
    context["artifacts"] = _artifact_refs(context["run_id"], record["step"])
    return _write_handoff(Path(root), context)


def apply_and_record(
    context: dict,
    proposal: dict,
    *,
    root: str | Path = "recordings",
    final: bool = False,
) -> tuple[dict, Path]:
    next_context, rendered = apply_proposal(context, proposal)
    skill, _ = proposal_skill(proposal)
    revision = next_context["revision"]
    context_rel = f"handoffs/context-{revision:03d}.json"
    proposal_rel = f"handoffs/proposal-{revision:03d}.json"
    record = record_composition_step(
        materialize_scene(next_context), rendered,
        run_id=next_context["run_id"], prompt_set=next_context["prompt_set"],
        composition=next_context["composition"],
        step_label=f"Luna revision {revision}: {skill}", final=final,
        provenance={
            "protocol_version": PROTOCOL_VERSION,
            "context_revision": revision,
            "agent_id": proposal["agent_id"],
            "mutation_id": proposal["mutation_id"],
            "skill": skill,
            "context": context_rel,
            "proposal": proposal_rel,
        },
        root=root,
    )
    next_context["artifacts"] = _artifact_refs(next_context["run_id"], record["step"])
    path = _write_handoff(Path(root), next_context, proposal)
    return next_context, path


def read_context(path: str | Path) -> dict:
    context = json.loads(Path(path).read_text())
    validate_context(context)
    return context


__all__ = [
    "ALLOWED_SKILLS", "LunaContextError", "PROTOCOL_VERSION", "apply_and_record",
    "apply_proposal", "create_context", "initialize_recorded_context",
    "materialize_scene", "proposal_skill", "read_context", "state_sha256",
    "validate_context",
]
