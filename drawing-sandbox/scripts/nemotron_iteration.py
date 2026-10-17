#!/usr/bin/env python3
"""Run consecutive Nemotron picture agents through Crusoe Managed Inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from drawing_sandbox.iteration_context import (
    ALLOWED_SKILLS,
    apply_and_record,
    create_context,
    initialize_recorded_context,
    read_context,
)
from drawing_sandbox.nemotron_inference import (
    infer_proposal,
    inference_provenance,
    load_crusoe_api_key,
    write_inference_log,
)


DEFAULT_SKILLS = ("split_layers", "add_objects", "remove_objects", "combine_layers")


def slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return result[:48] or "layer"


def reference_record(path: Path) -> dict:
    return {
        "asset_id": slug(path.stem),
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "role": "visual_reference",
    }


def initialize(
    scene_path: Path,
    manifest_path: Path,
    run_id: str,
    reference_paths: list[Path],
    record_root: Path,
) -> Path:
    destination = record_root / run_id
    if destination.exists():
        raise ValueError(f"recording already exists: {destination}")
    scene = json.loads(scene_path.read_text())
    source_manifest = json.loads(manifest_path.read_text())
    layer_plan = source_manifest["composition"].get("layers")
    if not isinstance(layer_plan, list) or not layer_plan:
        raise ValueError("source manifest composition.layers is required")
    layer_specs = []
    used = set()
    for index, layer in enumerate(layer_plan):
        label = layer["label"]
        base = slug(label)
        layer_id = base if base not in used else f"{base}-{index}"
        used.add(layer_id)
        layer_specs.append({
            "layer_id": layer_id,
            "label": label,
            "end_index": layer["cumulative_node_count"],
        })
    context = create_context(
        scene,
        run_id=run_id,
        prompt_set=source_manifest["prompt_set"],
        composition=source_manifest["composition"],
        layer_specs=layer_specs,
        references=[reference_record(path) for path in reference_paths],
    )
    return initialize_recorded_context(context, root=record_root)


def current_png(context: dict, record_root: Path) -> Path:
    return record_root / context["run_id"] / "steps" / f"{context['revision']:03d}-drawing.png"


def advance(
    context_path: Path,
    skill: str,
    *,
    reference_paths: list[Path],
    record_root: Path,
    api_key: str,
    final: bool,
    max_attempts: int,
    max_tokens: int,
    timeout: float,
) -> Path:
    context = read_context(context_path)
    proposal, inference_log = infer_proposal(
        context,
        skill,
        api_key=api_key,
        current_image=current_png(context, record_root),
        reference_images=reference_paths,
        max_attempts=max_attempts,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    log_path = write_inference_log(inference_log, root=record_root)
    next_context, next_path = apply_and_record(
        context,
        proposal,
        root=record_root,
        final=final,
        inference_provenance=inference_provenance(inference_log),
    )
    accepted = inference_log["attempts"][inference_log["accepted_attempt"] - 1]["response"]
    print(json.dumps({
        "revision": next_context["revision"],
        "skill": skill,
        "agent_id": proposal["agent_id"],
        "response_id": accepted.get("id"),
        "attempt": inference_log["accepted_attempt"],
        "usage": accepted.get("usage", {}),
        "inference_log": str(log_path.resolve()),
        "context": str(next_path.resolve()),
        "final": final,
    }, separators=(",", ":")), flush=True)
    return next_path


def init_command(args) -> int:
    path = initialize(
        args.scene, args.source_manifest, args.run_id, args.reference, args.record_root
    )
    print(path.resolve())
    return 0


def infer_command(args) -> int:
    api_key = load_crusoe_api_key(args.env_file)
    advance(
        args.context,
        args.skill,
        reference_paths=args.reference,
        record_root=args.record_root,
        api_key=api_key,
        final=args.final,
        max_attempts=args.max_attempts,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    return 0


def run_command(args) -> int:
    api_key = load_crusoe_api_key(args.env_file)
    path = initialize(
        args.scene, args.source_manifest, args.run_id, args.reference, args.record_root
    )
    print(json.dumps({"initialized": str(path.resolve()), "run_id": args.run_id}), flush=True)
    for index, skill in enumerate(args.skill):
        path = advance(
            path,
            skill,
            reference_paths=args.reference,
            record_root=args.record_root,
            api_key=api_key,
            final=index == len(args.skill) - 1,
            max_attempts=args.max_attempts,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
    return 0


def add_inference_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--reference", type=Path, action="append", default=[])
    parser.add_argument("--record-root", type=Path, default=Path("recordings"))
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=180)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    initializer = subparsers.add_parser("init", help="Create and record revision zero")
    initializer.add_argument("--scene", type=Path, required=True)
    initializer.add_argument("--source-manifest", type=Path, required=True)
    initializer.add_argument("--run-id", required=True)
    initializer.add_argument("--reference", type=Path, action="append", default=[])
    initializer.add_argument("--record-root", type=Path, default=Path("recordings"))
    initializer.set_defaults(function=init_command)

    inference = subparsers.add_parser("infer", help="Run one Nemotron skill agent")
    inference.add_argument("--context", type=Path, required=True)
    inference.add_argument("--skill", choices=ALLOWED_SKILLS, required=True)
    inference.add_argument("--final", action="store_true")
    add_inference_options(inference)
    inference.set_defaults(function=infer_command)

    run = subparsers.add_parser("run", help="Initialize and run a consecutive agent chain")
    run.add_argument("--scene", type=Path, required=True)
    run.add_argument("--source-manifest", type=Path, required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--skill", choices=ALLOWED_SKILLS, action="append", default=[])
    add_inference_options(run)
    run.set_defaults(function=run_command)

    args = parser.parse_args()
    if args.command == "run" and not args.skill:
        args.skill = list(DEFAULT_SKILLS)
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
