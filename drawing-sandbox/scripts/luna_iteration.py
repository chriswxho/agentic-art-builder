#!/usr/bin/env python3
"""Initialize and apply typed Luna picture-agent handoffs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from drawing_sandbox.luna_context import (
    apply_and_record,
    create_context,
    initialize_recorded_context,
    read_context,
)


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


def init(args) -> int:
    destination = args.record_root / args.run_id
    if destination.exists():
        raise ValueError(f"recording already exists: {destination}")
    scene = json.loads(args.scene.read_text())
    source_manifest = json.loads(args.source_manifest.read_text())
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
        run_id=args.run_id,
        prompt_set=source_manifest["prompt_set"],
        composition=source_manifest["composition"],
        layer_specs=layer_specs,
        references=[reference_record(path) for path in args.reference],
    )
    path = initialize_recorded_context(context, root=args.record_root)
    print(path.resolve())
    return 0


def apply(args) -> int:
    context = read_context(args.context)
    proposal = json.loads(args.proposal.read_text())
    next_context, path = apply_and_record(
        context, proposal, root=args.record_root, final=args.final,
    )
    print(json.dumps({
        "context": str(path.resolve()),
        "revision": next_context["revision"],
        "skill": next_context["last_handoff"]["skill"],
        "agent_id": next_context["last_handoff"]["agent_id"],
        "final": args.final,
        "artifacts": next_context["artifacts"],
    }, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    initializer = subparsers.add_parser("init", help="Create revision zero and record it")
    initializer.add_argument("--scene", type=Path, required=True)
    initializer.add_argument("--source-manifest", type=Path, required=True)
    initializer.add_argument("--run-id", required=True)
    initializer.add_argument("--record-root", type=Path, default=Path("recordings"))
    initializer.add_argument("--reference", type=Path, action="append", default=[])
    initializer.set_defaults(function=init)

    applier = subparsers.add_parser("apply", help="Validate, apply, render, and record one proposal")
    applier.add_argument("--context", type=Path, required=True)
    applier.add_argument("--proposal", type=Path, required=True)
    applier.add_argument("--record-root", type=Path, default=Path("recordings"))
    applier.add_argument("--final", action="store_true")
    applier.set_defaults(function=apply)
    args = parser.parse_args()
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
