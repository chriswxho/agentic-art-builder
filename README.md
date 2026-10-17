# agentic-art-builder

A workspace for AI agents that draw. An agent writes a drawing as JSON, a renderer turns it into an image and checks it against a fixed set of rules, and every step of the drawing is saved so you can see how it was built.

> This README describes how the app works **as of the current commit**. It is updated in every commit that changes behavior.

## How it works right now

Everything lives in [drawing-sandbox/](drawing-sandbox/), a small Python package (Python 3.10+, NumPy and Pillow only, no model calls).

1. **An agent writes a scene.** A scene is JSON listing shapes (paths, polygons, ellipses), optionally grouped or reused as "motifs".
2. **The renderer checks it.** Every drawing must follow the CMYK rules in [policy.py](drawing-sandbox/drawing_sandbox/policy.py):
   - white canvas; paint only in cyan, magenta, yellow or black
   - every mark at 50% opacity, so overlaps mix into new colors
   - each mark at most 96 units across, strokes at most 12 wide, at most 512 marks in total
3. **If it passes, it's rendered** to SVG and PNG. If it fails, the agent gets a list of what broke.
4. **The agent looks at the PNG and revises.** The agent (not this package) runs that loop.
5. **Each step is recorded** under `drawing-sandbox/recordings/<run_id>/`: the scene, SVG, PNG, checks, the prompts used, the plan, and (optionally) the model's raw text output for that step.
6. **There's a graded task dataset** for testing agents: 10 drawing tasks from "one cyan dot" (level 1) to "harbor at sunset" (level 10), each with a prompt, skills, extra limits, success criteria, and a passing reference drawing. See [drawing-sandbox/dataset/](drawing-sandbox/dataset/README.md).
7. **Everything has a readable version.** Every render result carries a plain-text `summary` (pass/fail, broken rules, and a line-by-line outline of what the model drew), and every recorded run gets a `report.md` showing prompts, plan, and each step's image, checks, model output and outline.

### Ways to use it

| Interface | What it does |
| --- | --- |
| `python -m drawing_sandbox` | Opens a local web editor at http://127.0.0.1:8766 |
| `POST /api/render` | Renders a scene and returns checks, SVG and PNG |
| `POST /api/record` | Renders and saves one step of a run (accepts `model_output` text); rewrites the run's `report.md` |
| `GET /api/examples`, `GET /api/instruction-set` | Starter scenes and the rules |
| `GET /api/dataset` | The graded task cards (no reference answers) |
| `python -m drawing_sandbox render / check / examples` | Same things from the terminal; `check` prints a readable summary (`--json` for raw) |
| `python -m drawing_sandbox report <run_dir>` | Regenerates a run's `report.md` |
| `python -m drawing_sandbox dataset --out dataset` | Regenerates the task dataset |
| `render_drawing(scene)` in Python | Same as `/api/render` |

### What's in the repo

- [drawing-sandbox/drawing_sandbox/](drawing-sandbox/drawing_sandbox/): renderer, rules, recording, local server and web editor
- [drawing-sandbox/examples/](drawing-sandbox/examples/): finished example drawings
- [drawing-sandbox/dataset/](drawing-sandbox/dataset/README.md): graded drawing tasks with reference solutions (built by [dataset.py](drawing-sandbox/drawing_sandbox/dataset.py))
- [drawing-sandbox/recordings/](drawing-sandbox/recordings/): a sample recorded run, with its [report.md](drawing-sandbox/recordings/cmyk-overprint-demo/report.md)
- [drawing-sandbox/tests/](drawing-sandbox/tests/): unit tests
- [AGENTS.md](AGENTS.md): rules for any agent that draws or edits code here

## Quick start

```bash
cd drawing-sandbox
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
python -m drawing_sandbox
```

Run the tests with `python -m unittest discover -s tests -v`. More detail is in [drawing-sandbox/README.md](drawing-sandbox/README.md).

## Changelog

- **Graded drawing dataset**: 10 tasks from beginner to expert with prompts, success criteria and reference solutions; `dataset` CLI command and `GET /api/dataset`.
- **Human-readable output**: render results gain a text `summary`; recordings save the model's raw output and a `report.md`; CLI `check` is readable by default; new `report` command.
- **Build constrained CMYK drawing sandbox**: renderer, rules, recording, editor, examples.
- **Initial commit**: hello world.
