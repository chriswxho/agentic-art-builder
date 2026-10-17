#!/usr/bin/env python3
"""Run consecutive Nemotron picture agents through Crusoe Managed Inference."""

from __future__ import annotations

import argparse
from copy import deepcopy
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
    NemotronInferenceError,
    evaluator_provenance,
    infer_evaluation,
    infer_plan,
    infer_proposal,
    inference_provenance,
    load_crusoe_api_key,
    planner_provenance,
    write_inference_log,
    write_evaluation_artifacts,
    write_planner_artifacts,
    write_rejected_inference_log,
)
from drawing_sandbox.planning import validate_plan


DEFAULT_SKILLS = ("split_layers", "add_objects", "remove_objects", "combine_layers")
MAX_GENERATOR_BATCH_SHAPES = 8


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


def execution_assignments(plan: dict) -> list[dict]:
    """Split each planned regional minimum into compact, deterministic generator batches."""
    tasks = []
    for assignment in plan["assignments"]:
        remaining = assignment["minimum_shapes"]
        batch_count = (remaining + MAX_GENERATOR_BATCH_SHAPES - 1) // MAX_GENERATOR_BATCH_SHAPES
        for batch_index in range(batch_count):
            batches_left = batch_count - batch_index
            count = min(MAX_GENERATOR_BATCH_SHAPES, (remaining + batches_left - 1) // batches_left)
            task = deepcopy(assignment)
            parent_id = assignment["assignment_id"]
            task["assignment_id"] = f"{slug(parent_id)[:54]}-b{batch_index + 1:02d}"
            task["parent_assignment_id"] = parent_id
            task["part_label"] = f"{assignment['part_label']} · batch {batch_index + 1}/{batch_count}"
            task["minimum_shapes"] = count
            task["shape_budget"] = count
            tasks.append(task)
            remaining -= count
    return tasks


def write_execution_plan(plan: dict, *, run_id: str, record_root: Path) -> Path:
    path = record_root / run_id / "handoffs" / "execution-plan.json"
    payload = {
        "plan_id": plan["plan_id"],
        "max_generator_batch_shapes": MAX_GENERATOR_BATCH_SHAPES,
        "assignments": execution_assignments(plan),
    }
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return path


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
    assignment: dict | None = None,
    plan_id: str | None = None,
) -> Path:
    context = read_context(context_path)
    agent_id = None
    mutation_id = None
    if assignment is not None:
        part = slug(assignment["assignment_id"])[:36]
        next_revision = context["revision"] + 1
        agent_id = f"nemotron-{part}-r{next_revision:03d}"
        mutation_id = f"{slug(context['run_id'])[:28]}-{part[:24]}-r{next_revision:03d}"
    try:
        proposal, inference_log = infer_proposal(
            context,
            skill,
            api_key=api_key,
            current_image=current_png(context, record_root),
            reference_images=reference_paths,
            agent_id=agent_id,
            mutation_id=mutation_id,
            assignment=assignment,
            plan_id=plan_id,
            max_attempts=max_attempts,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    except NemotronInferenceError as error:
        if error.log is not None:
            rejected_path = write_rejected_inference_log(error.log, root=record_root)
            print(json.dumps({"rejected_inference": str(rejected_path.resolve())}), flush=True)
        raise
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


def plan_assignments(
    context_path: Path,
    *,
    reference_paths: list[Path],
    record_root: Path,
    api_key: str,
    max_attempts: int,
    max_tokens: int,
    timeout: float,
) -> tuple[dict, dict]:
    context = read_context(context_path)
    try:
        plan, log = infer_plan(
            context,
            api_key=api_key,
            current_image=current_png(context, record_root),
            reference_images=reference_paths,
            max_attempts=max_attempts,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    except NemotronInferenceError as error:
        if error.log is not None:
            rejected_path = write_rejected_inference_log(error.log, root=record_root)
            print(json.dumps({"rejected_inference": str(rejected_path.resolve())}), flush=True)
        raise
    plan_path, log_path = write_planner_artifacts(plan, log, root=record_root)
    execution_path = write_execution_plan(
        plan, run_id=context["run_id"], record_root=record_root
    )
    provenance = planner_provenance(log)
    manifest_path = record_root / context["run_id"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["planner"] = provenance
    manifest["planner"]["execution_plan"] = "handoffs/execution-plan.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "planner_id": plan["planner_id"],
        "plan_id": plan["plan_id"],
        "assignments": len(plan["assignments"]),
        "shape_budget": sum(item["shape_budget"] for item in plan["assignments"]),
        "plan": str(plan_path.resolve()),
        "execution_plan": str(execution_path.resolve()),
        "inference_log": str(log_path.resolve()),
        "response_id": provenance["response_id"],
    }, separators=(",", ":")), flush=True)
    return plan, provenance


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


def planned_run_command(args) -> int:
    api_key = load_crusoe_api_key(args.env_file)
    path = initialize(
        args.scene, args.source_manifest, args.run_id, args.reference, args.record_root
    )
    print(json.dumps({"initialized": str(path.resolve()), "run_id": args.run_id}), flush=True)
    plan, _ = plan_assignments(
        path,
        reference_paths=args.reference,
        record_root=args.record_root,
        api_key=api_key,
        max_attempts=args.max_attempts,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    final_path = execute_assignments(
        path, plan, set(), reference_paths=args.reference, record_root=args.record_root,
        api_key=api_key, max_attempts=args.max_attempts, max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    evaluate_run(
        path, final_path, plan, reference_paths=args.reference, record_root=args.record_root,
        api_key=api_key, max_attempts=args.max_attempts, timeout=args.timeout,
    )
    return 0


def execute_assignments(
    context_path: Path,
    plan: dict,
    completed: set[str],
    *,
    reference_paths: list[Path],
    record_root: Path,
    api_key: str,
    max_attempts: int,
    max_tokens: int,
    timeout: float,
) -> Path:
    tasks = execution_assignments(plan)
    remaining = [
        assignment for assignment in tasks
        if assignment["assignment_id"] not in completed
        and assignment["parent_assignment_id"] not in completed
    ]
    path = context_path
    for index, assignment in enumerate(remaining):
        path = advance(
            path,
            "add_objects",
            reference_paths=reference_paths,
            record_root=record_root,
            api_key=api_key,
            final=index == len(remaining) - 1,
            max_attempts=max_attempts,
            max_tokens=max_tokens,
            timeout=timeout,
            assignment=assignment,
            plan_id=plan["plan_id"],
        )
    return path


def evaluate_run(
    base_context_path: Path,
    final_context_path: Path,
    plan: dict,
    *,
    reference_paths: list[Path],
    record_root: Path,
    api_key: str,
    max_attempts: int,
    timeout: float,
) -> dict:
    base_context = read_context(base_context_path)
    final_context = read_context(final_context_path)
    try:
        evaluation, log = infer_evaluation(
            base_context,
            final_context,
            plan,
            api_key=api_key,
            baseline_image=current_png(base_context, record_root),
            final_image=current_png(final_context, record_root),
            reference_images=reference_paths,
            max_attempts=max_attempts,
            timeout=timeout,
        )
    except NemotronInferenceError as error:
        if error.log is not None:
            rejected_path = write_rejected_inference_log(error.log, root=record_root)
            print(json.dumps({"rejected_inference": str(rejected_path.resolve())}), flush=True)
        raise
    evaluation_path, log_path = write_evaluation_artifacts(
        evaluation, log, root=record_root
    )
    provenance = evaluator_provenance(log)
    manifest_path = record_root / final_context["run_id"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["evaluator"] = provenance
    manifest["evaluation_summary"] = {
        "reference_match_before": evaluation["reference_match_before"],
        "reference_match_after": evaluation["reference_match_after"],
        "composition_before": evaluation["composition_before"],
        "composition_after": evaluation["composition_after"],
        "improved": evaluation["improved"],
        "summary": evaluation["summary"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "evaluator_id": evaluation["evaluator_id"],
        "improved": evaluation["improved"],
        "reference_match_before": evaluation["reference_match_before"],
        "reference_match_after": evaluation["reference_match_after"],
        "evaluation": str(evaluation_path.resolve()),
        "inference_log": str(log_path.resolve()),
        "response_id": provenance["response_id"],
    }, separators=(",", ":")), flush=True)
    return evaluation


def resume_plan_command(args) -> int:
    api_key = load_crusoe_api_key(args.env_file)
    run_dir = args.record_root / args.run_id
    plan = json.loads((run_dir / "handoffs" / "plan.json").read_text())
    base_context = read_context(run_dir / "handoffs" / "context-000.json")
    plan = validate_plan(base_context, plan)
    if not (run_dir / "handoffs" / "execution-plan.json").exists():
        write_execution_plan(plan, run_id=args.run_id, record_root=args.record_root)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    completed = {
        step.get("provenance", {}).get("assignment_id")
        for step in manifest["steps"]
        if step.get("provenance", {}).get("assignment_id")
    }
    context_paths = sorted((run_dir / "handoffs").glob("context-*.json"))
    if not context_paths:
        raise ValueError("run has no context handoffs")
    path = execute_assignments(
        context_paths[-1], plan, completed, reference_paths=args.reference,
        record_root=args.record_root, api_key=api_key, max_attempts=args.max_attempts,
        max_tokens=args.max_tokens, timeout=args.timeout,
    )
    if not (run_dir / "handoffs" / "evaluation.json").exists():
        evaluate_run(
            run_dir / "handoffs" / "context-000.json", path, plan,
            reference_paths=args.reference, record_root=args.record_root, api_key=api_key,
            max_attempts=args.max_attempts, timeout=args.timeout,
        )
    print(json.dumps({
        "run_id": args.run_id,
        "completed_assignments": len(plan["assignments"]),
        "context": str(path.resolve()),
    }, separators=(",", ":")), flush=True)
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

    planned = subparsers.add_parser(
        "planned-run", help="Plan regional shape work, then run one specialist per assignment"
    )
    planned.add_argument("--scene", type=Path, required=True)
    planned.add_argument("--source-manifest", type=Path, required=True)
    planned.add_argument("--run-id", required=True)
    add_inference_options(planned)
    planned.set_defaults(function=planned_run_command)

    resume = subparsers.add_parser(
        "resume-plan", help="Resume unaccepted assignments from an existing planned run"
    )
    resume.add_argument("--run-id", required=True)
    add_inference_options(resume)
    resume.set_defaults(function=resume_plan_command)

    args = parser.parse_args()
    if args.command == "run" and not args.skill:
        args.skill = list(DEFAULT_SKILLS)
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
