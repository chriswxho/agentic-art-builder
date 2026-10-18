#!/usr/bin/env python3
"""Run test_4 on every reference image through a queue, a few at a time.

Runs whose folder already exists in test_4/runs/ are skipped. Each run is its own process
(run_test_4.py --only <id>); progress goes to test_4/run_all.log, and when everything is done
results.md summarizes every run.

usage (from the repo root):
    drawing-sandbox/.venv/bin/python test_4/run_all.py [--parallel 6] [--budget 400]
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "test_4"
sys.path[:0] = [str(HERE)]


def log(message: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    with (HERE / "run_all.log").open("a") as handle:
        handle.write(line + "\n")


def summarize() -> Path:
    from run_test_4 import TARGETS
    rows = []
    for run_id in TARGETS:
        result_path = HERE / f"result-{run_id}.json"
        scores_path = HERE / "runs" / run_id / "scores.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        scores = json.loads(scores_path.read_text()) if scores_path.exists() else []
        final = scores[-1] if scores else {}
        minutes = (scores[-1]["time"] - scores[0]["time"]) / 60 if len(scores) > 1 else 0
        verdict = str(result.get("super_vl", ""))
        match = verdict.split(",")[0].replace("reference_match ", "") if verdict.startswith("reference_match") else "—"
        rows.append((run_id.split("-")[0], run_id, final.get("lab"), final.get("lenient"), final.get("commit"),
                     result.get("calls"), minutes, match, result.get("stop_reason") or result.get("error") or "not run"))
    out = ["# test_4 results: all images", "",
           "| Level | Run | Lab | Lenient | Commits | Calls | Minutes | Super-VL match | Stop reason |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for level, run_id, lab, lenient, commits, calls, minutes, match, stop in rows:
        link = f"[{run_id}](runs/{run_id}/AGENT_REPORT.md)" if (HERE / "runs" / run_id / "AGENT_REPORT.md").exists() else run_id
        out.append(f"| {level} | {link} | {lab if lab is not None else '—'} | {lenient if lenient is not None else '—'} | "
                   f"{commits if commits is not None else '—'} | {calls or '—'} | {minutes:.1f} | {match} | {stop} |")
    out += ["", "## Averages by level", "", "| Level | Runs finished | Mean Lab | Best Lab | Runs at or above 0.9 |",
            "| --- | --- | --- | --- | --- |"]
    for level in ("easy", "medium", "hard"):
        labs = [row[2] for row in rows if row[0] == level and row[2] is not None]
        if labs:
            out.append(f"| {level} | {len(labs)} | {sum(labs) / len(labs):.3f} | {max(labs):.3f} | {sum(lab >= 0.9 for lab in labs)} |")
    path = HERE / "results.md"
    path.write_text("\n".join(out) + "\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parallel", type=int, default=6)
    parser.add_argument("--budget", type=int, default=400)
    args = parser.parse_args()
    from run_test_4 import TARGETS
    queue = [run_id for run_id in TARGETS if not (HERE / "runs" / run_id).exists()]
    log(f"queue: {len(queue)} runs ({len(TARGETS) - len(queue)} already done), {args.parallel} at a time")
    running: dict = {}
    while queue or running:
        while queue and len(running) < args.parallel:
            run_id = queue.pop(0)
            stdout = open(HERE / f"stdout-{run_id}.txt", "w")
            running[run_id] = (subprocess.Popen(
                [sys.executable, str(HERE / "run_test_4.py"), "--only", run_id, "--budget", str(args.budget)],
                cwd=ROOT, stdout=stdout, stderr=subprocess.STDOUT), stdout)
            log(f"started {run_id} ({len(queue)} waiting)")
        time.sleep(10)
        for run_id, (process, stdout) in list(running.items()):
            if process.poll() is None:
                continue
            stdout.close()
            del running[run_id]
            result_path = HERE / f"result-{run_id}.json"
            result = json.loads(result_path.read_text()) if result_path.exists() else {"error": f"exit {process.returncode}"}
            if (HERE / "runs" / run_id / "tree.json").exists():
                subprocess.run([sys.executable, str(HERE / "agent_report.py"), str(HERE / "runs" / run_id)],
                               cwd=ROOT, capture_output=True)
            log(f"finished {run_id}: lab {result.get('lab')}, commits {result.get('commits')}, calls {result.get('calls')}, "
                f"{result.get('stop_reason') or result.get('error')}")
            summarize()
    log(f"all done; summary in {summarize()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
