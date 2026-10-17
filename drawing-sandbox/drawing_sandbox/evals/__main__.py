"""Command line: python -m drawing_sandbox.evals {run,suite,compare,metrics}."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from . import METRICS, EvalCase, compare, evaluate_suite, load_suite


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _print_suite(suite: dict) -> None:
    for report in suite["reports"]:
        print(f"\n{report['case_id']}  (run {report['run_id']}, target {report['target_image'] or 'none'})")
        print(f"  overall {_fmt(report['overall'])}  " + "  ".join(
            f"{category} {_fmt(score)}" for category, score in report["category_scores"].items()))
        for name, result in report["metrics"].items():
            print(f"  {name:32} value {_fmt(result['value']):>10}   score {_fmt(result['score']):>8}")
        suggested = report["metrics"].get("steps.suggested_count")
        if suggested:
            print("  suggested layers:")
            for step in suggested["details"]["suggested_steps"]:
                print(f"    marks {step['range'][0]:>3}-{step['range'][1]:<3} inks {step.get('inks')} types {step.get('types')}")
    if len(suite["reports"]) > 1:
        print(f"\nmean overall {_fmt(suite['mean_overall'])}")


def _write(suite: dict, out: str | None) -> None:
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(suite, indent=2, allow_nan=False) + "\n")
        print(f"\nwrote {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m drawing_sandbox.evals")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="evaluate one recorded run")
    run.add_argument("run_dir")
    run.add_argument("--target", help="target image (defaults to composition.source_image)")
    run.add_argument("--reference-step", action="append", default=[],
                     help="cumulative image of a preferred layer, in order (repeatable)")
    run.add_argument("--preferred-steps", type=int)
    run.add_argument("--option", action="append", default=[], metavar="KEY=JSON",
                     help="override an option, e.g. step_penalty=0.08")

    suite = commands.add_parser("suite", help="evaluate every case in a suite file")
    suite.add_argument("suite_file")

    for command in (run, suite):
        command.add_argument("--only", action="append", help="metric name or prefix, e.g. steps")
        command.add_argument("--out", help="write the JSON report here")
        command.add_argument("--json", action="store_true", help="print JSON instead of a table")

    diff = commands.add_parser("compare", help="metric deltas between two saved reports")
    diff.add_argument("before")
    diff.add_argument("after")

    commands.add_parser("metrics", help="list registered metrics")
    args = parser.parse_args(argv)

    if args.command == "metrics":
        for name, (category, function) in METRICS.items():
            summary = (function.__doc__ or "").strip().split("\n")[0]
            print(f"{name:32} {category:11} {summary}")
        return 0
    if args.command == "compare":
        rows = compare(json.loads(Path(args.before).read_text()), json.loads(Path(args.after).read_text()))
        for row in rows:
            print(f"{row['case_id']:24} {row['metric']:32} {_fmt(row['before']):>10} -> "
                  f"{_fmt(row['after']):>10}   {_fmt(row['delta']):>10}")
        return 0

    if args.command == "run":
        options = {}
        for item in args.option:
            key, _, value = item.partition("=")
            options[key] = json.loads(value)
        cases = [EvalCase(
            case_id=Path(args.run_dir).name, run_dir=Path(args.run_dir),
            target_image=Path(args.target) if args.target else None,
            reference_steps=[Path(path) for path in args.reference_step],
            preferred_step_count=args.preferred_steps, options=options)]
    else:
        cases = load_suite(args.suite_file)

    result = evaluate_suite(cases, args.only)
    if args.json:
        print(json.dumps(result, indent=2, allow_nan=False))
    else:
        _print_suite(result)
    _write(result, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
