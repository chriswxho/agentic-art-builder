#!/usr/bin/env python3
"""Write AGENT_REPORT.md for a test_4 run: what every agent in the tree did, from the run's own records.

usage: drawing-sandbox/.venv/bin/python test_4/agent_report.py test_4/runs/easy-sea-sunset
"""

import json
from pathlib import Path
import statistics
import sys


def load(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def short(text, limit=160) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def main(run_dir: Path) -> Path:
    tree = load(run_dir / "tree.json")
    scores = load(run_dir / "scores.json", [])
    manifest = load(run_dir / "manifest.json")
    evaluation = load(run_dir / "handoffs" / "evaluation.json")
    nodes = {n["id"]: n for n in tree["nodes"]}
    children: dict = {}
    for n in tree["nodes"]:
        children.setdefault(n["parent"], []).append(n["id"])
    logs = {p.stem: load(p) for p in sorted((run_dir / "tree" / "nodes").glob("*.json"))}
    turns = [t for log in logs.values() for t in log["turns"]]
    t0 = scores[0]["time"] if scores else 0
    final = scores[-1] if scores else {"lab": 0, "lenient": 0, "commit": 0}

    out = [f"# Agent report: {tree['run_id']}", ""]
    out.append(f"Reference: `{Path(manifest['composition']['source_image']).name}` · model: "
               f"`nvidia/NVIDIA-Nemotron-3.5-Super-VL` for every node · stop reason: **{tree['stop_reason']}**")
    out += ["", "## Result", ""]
    out.append("| Measure | Value |")
    out.append("| --- | --- |")
    out.append(f"| Lab score (strict, target 0.9) | **{final['lab']:.3f}** |")
    out.append(f"| Lenient score (`skill_vs_blank`) | {final['lenient']:.3f} |")
    out.append(f"| Commits (patches kept) | {final['commit']} |")
    out.append(f"| Model calls | {tree['calls']} of {tree['budget']} budget |")
    out.append(f"| Agents in the tree | {len(tree['nodes'])} |")
    if scores:
        out.append(f"| Wall time to last commit | {(scores[-1]['time'] - t0) / 60:.1f} min |")
    if evaluation:
        out.append(f"| Super-VL verdict | reference match {evaluation['reference_match_after']}/100, "
                   f"composition {evaluation['composition_after']}/100 |")
    out += ["", "![reference vs final](comparison.png)", "", "![progress by commit](milestones.png)", ""]
    if evaluation:
        out += ["**Super-VL's summary:** " + evaluation["summary"], ""]
        if evaluation.get("issues"):
            out += ["Issues it named:"] + [f"- {item}" for item in evaluation["issues"]] + [""]

    # Who did what
    kinds = {}
    for n in tree["nodes"]:
        k = kinds.setdefault(n["kind"], {"count": 0, "calls": 0, "status": {}})
        k["count"] += 1
        k["calls"] += n.get("calls", 0)
        k["status"][n["status"]] = k["status"].get(n["status"], 0) + 1
    roles = {"director": "plans areas and back-to-front order", "background": "free base fill (no model call)",
             "area": "area lead: draws patch outlines", "patch": "one patch (holds its candidates)",
             "candidate": "painter: proposes 1-3 shapes", "refiner": "splits a patch that is still off"}
    out += ["## The tree at a glance", "", "| Agent type | What it does | Count | Model calls | Outcomes |",
            "| --- | --- | --- | --- | --- |"]
    for kind in ("director", "background", "area", "patch", "candidate", "refiner"):
        if kind in kinds:
            k = kinds[kind]
            outcomes = ", ".join(f"{v} {s}" for s, v in sorted(k["status"].items()))
            out.append(f"| {kind} | {roles[kind]} | {k['count']} | {k['calls']} | {outcomes} |")
    out.append("")

    # Director plan
    director = next(n for n in tree["nodes"] if n["kind"] == "director")
    out += ["## 1. Director's plan", "", "| Area | What's in it | Box (x0, y0, x1, y1) | Order | Base color |",
            "| --- | --- | --- | --- | --- |"]
    for area in director.get("result", []):
        out.append(f"| {area['area_id']} | {area['label']} | {area['box']} | {area['order']} | `{area.get('base_color', '')}` |")
    out += ["", "Order 0 is painted first (background); higher orders go on top.", ""]

    # Free base fills
    fills = [n for n in tree["nodes"] if n["kind"] == "background"]
    if fills:
        out += ["## 2. Free base fills (0 model calls)", "", "| Fill | Target | Ink layers | Result | Color difference left |",
                "| --- | --- | --- | --- | --- |"]
        for n in fills:
            layers = " then ".join(f"{ink} @ {a:.2f}" for ink, a in n.get("layers", []))
            out.append(f"| {n['label']} | `{n.get('target', '')}` | {layers} | {n['status']} | {n.get('remaining_delta_e')} |")
        out.append("")

    # Commit timeline
    commit_rows = []
    for step in manifest["steps"][1:]:
        prov = step.get("provenance", {})
        node = nodes.get(prov.get("tree_node"), {})
        commit_rows.append((prov.get("commit"), node, prov))
    out += ["## 3. Every commit, in order", "",
            "Each commit is a patch (or base fill) that passed the keep-if-better gate: its color difference "
            "inside the patch dropped and the whole-picture score didn't.", "",
            "| # | Time | Agent | Patch | Depth | Patch color diff (before → after) | Lab score after |",
            "| --- | --- | --- | --- | --- | --- | --- |"]
    times = {s["commit"]: s["time"] for s in scores}
    for number, node, prov in commit_rows:
        d = prov.get("patch_delta_e", ["", ""])
        lab = prov.get("lab_skill", ["", ""])
        when = f"{(times.get(number, t0) - t0) / 60:.1f} min" if number in times else ""
        out.append(f"| {number} | {when} | {node.get('kind', '')} `{node.get('id', '')}` | {short(node.get('label'), 60)} | "
                   f"{node.get('depth', '')} | {d[0]} → {d[1]} | {lab[1] if len(lab) > 1 else ''} |")
    out.append("")

    # Patches that did not commit
    misses = [n for n in tree["nodes"] if n["kind"] == "patch" and n["status"] != "committed"]
    if misses:
        out += [f"## 4. Patches that were tried but not kept ({len(misses)})", "",
                "None of their candidates improved the canvas enough to pass the gate.", "",
                "| Patch | Depth | Parent | Candidate gains (color diff points) | Color diff left |",
                "| --- | --- | --- | --- | --- |"]
        for n in misses:
            gains = [nodes[c].get("gain") for c in children.get(n["id"], []) if nodes[c]["kind"] == "candidate"]
            parent = nodes.get(n["parent"], {})
            out.append(f"| {short(n['label'], 60)} | {n['depth']} | {parent.get('kind', '')}: {short(parent.get('label'), 40)} | "
                       f"{', '.join(str(g) for g in gains) or 'no valid candidates'} | {n.get('remaining_delta_e')} |")
        out.append("")

    # Full tree outline
    def outline(node_id: str, depth: int = 0):
        n = nodes[node_id]
        extras = []
        if n.get("calls"):
            extras.append(f"{n['calls']} call{'s' if n['calls'] != 1 else ''}")
        if "gain" in n:
            extras.append(f"gain {n['gain']}")
        if "remaining_delta_e" in n:
            extras.append(f"left {n['remaining_delta_e']}")
        if n.get("tool_calls"):
            extras.append("tools: " + ", ".join(t["name"] for t in n["tool_calls"]))
        line = f"{'  ' * depth}- **{n['kind']}** `{n['id']}` {short(n['label'], 70)} — {n['status']}"
        lines = [line + (f" ({'; '.join(extras)})" if extras else "")]
        for child in children.get(node_id, []):
            lines += outline(child, depth + 1)
        return lines

    out += ["## 5. The whole tree", "", "Indentation shows who spawned whom. `gain` is a candidate's measured "
            "improvement inside its patch; `left` is the patch's remaining color difference (Lab ΔE).", ""]
    for root in children.get(None, []):
        out += outline(root)
    out.append("")

    # Call statistics
    secs = [t["seconds"] for t in turns if t.get("seconds")]
    toks = [(t.get("usage") or {}).get("completion_tokens") or 0 for t in turns if t.get("usage")]
    errors = [t for t in turns if t.get("error")]
    invalid = [t for t in turns if t.get("validation_error")]
    tools = [t["tool"]["name"] for t in turns if t.get("tool")]
    out += ["## 6. Model calls", "", "| Measure | Value |", "| --- | --- |",
            f"| Calls logged | {len(turns)} |"]
    if secs:
        out.append(f"| Seconds per call (median / 90th pct / max) | {statistics.median(secs):.0f} / "
                   f"{sorted(secs)[int(len(secs) * 0.9)]:.0f} / {max(secs):.0f} |")
    if toks:
        out.append(f"| Output tokens per call (median / max) | {statistics.median(toks):.0f} / {max(toks)} |")
    out.append(f"| Tool calls | {len(tools)} ({', '.join(f'{name} {tools.count(name)}' for name in sorted(set(tools)))}) |")
    out.append(f"| Network errors / timeouts | {len(errors)} |")
    out.append(f"| Answers rejected by validation (then retried) | {len(invalid)} |")
    out.append("")
    if invalid:
        reasons: dict = {}
        for t in invalid:
            key = short(t["validation_error"], 90)
            reasons[key] = reasons.get(key, 0) + 1
        out += ["Validation rejections:"] + [f"- {count}× {reason}" for reason, count in sorted(reasons.items(), key=lambda x: -x[1])] + [""]

    # Examples: exact prompts and answers
    def example(node_id: str, title: str):
        log = logs.get(node_id)
        if not log:
            return []
        accepted = next((t for t in log["turns"] if t.get("accepted")), log["turns"][-1] if log["turns"] else {})
        return [f"### {title} (`{node_id}`)", "", "Prompt sent (text part; the image was the sheet in "
                f"`{log['image']}`):", "", "```text", log["user_prompt"][:3000], "```", "", "Answer:", "",
                "```json", (accepted.get("content") or "")[:2500], "```", ""]

    first_lead = next((n["id"] for n in tree["nodes"] if n["kind"] == "area"), None)
    first_commit = next((nid for nid, n in nodes.items() if n["kind"] == "candidate" and n["status"] == "committed"), None)
    out += ["## 7. What the agents actually said", ""]
    out += example(director["id"], "Director")
    if first_lead:
        out += example(first_lead, "An area lead")
    if first_commit:
        out += example(first_commit, "A painter whose shapes were kept")
    out += ["## Files", "",
            "- `tree.json` / `tree.html`: every agent, its parent, status and results",
            "- `tree/nodes/<id>.json`: each agent's full conversation (prompts, raw answers, tool calls, usage)",
            "- `tree/sheets/<id>.png`: the image each agent was shown",
            "- `steps/`, `manifest.json`: each commit as a recorded revision; `scores.json`: score after each commit",
            "- `skills-log.jsonl`: one line per kept patch (target color, shapes, improvement), raw material for a skills file", ""]
    path = run_dir / "AGENT_REPORT.md"
    path.write_text("\n".join(out))
    return path


if __name__ == "__main__":
    print(main(Path(sys.argv[1])))
