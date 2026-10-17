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
4. **An agent looks at the PNG and revises.** The built-in agent loop chains Nemotron calls (`nvidia/Nemotron-3-Nano-Omni-Reasoning-30B-A3B` on Crusoe Managed Inference, token from `CRUSOE_API_TOKEN` or the gitignored `.env`). Each call returns one typed edit (`split_layers`, `combine_layers`, `add_objects` or `remove_objects`) against a shared `PictureContext`, which the orchestrator validates, renders, records and hands to the next agent. See [scripts/nemotron_iteration.py](drawing-sandbox/scripts/nemotron_iteration.py).
5. **Each step is recorded** under `drawing-sandbox/recordings/<run_id>/`: the scene, SVG, PNG, checks, the prompts used, the plan, and (optionally) the model's raw text output for that step.
6. **There's a graded task dataset** for testing agents: 10 drawing tasks from "one cyan dot" (level 1) to "harbor at sunset" (level 10), each with a prompt, skills, extra limits, success criteria, and a passing reference drawing. See [drawing-sandbox/dataset/](drawing-sandbox/dataset/README.md).
7. **Runs can be watched and scored.** `/watch?run=<run_id>` replays a run frame by frame in the browser, and `python -m drawing_sandbox.evals` scores recorded runs (step count, image similarity to a target, preferences). The eval metrics are provisional; see [EVALS.md](drawing-sandbox/EVALS.md).
8. **Everything has a readable version.** Every render result carries a plain-text `summary` (pass/fail, broken rules, and a line-by-line outline of what the model drew), and every recorded run gets a `report.md` showing prompts, plan, and each step's image, checks, model output and outline.

### Ways to use it

| Interface | What it does |
| --- | --- |
| `python -m drawing_sandbox` | Opens a local web editor at http://127.0.0.1:8766 |
| `POST /api/render` | Renders a scene and returns checks, SVG and PNG |
| `POST /api/record` | Renders and saves one step of a run (accepts `model_output` text); rewrites the run's `report.md` |
| `GET /api/examples`, `GET /api/instruction-set` | Starter scenes and the rules |
| `GET /api/dataset` | The graded task cards (no reference answers) |
| `GET /watch?run=<run_id>` | Live, read-only playback of a recorded run |
| `python scripts/nemotron_iteration.py run ...` | Runs a chain of Nemotron picture agents |
| `python -m drawing_sandbox.evals run / suite / compare` | Scores recorded runs |
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
- Reference landscape images in three difficulty levels, all 1536×1024 (center-cropped, not stretched):
  - [landscape-paintings/](landscape-paintings/) **easy**: 10 flat, simple landscape pictures as PNG and SVG. The SVGs are still at the original 1740×1080.
  - [medphotos/](medphotos/) **medium**: 9 famous landscape paintings (Starry Night, The Great Wave, The Hay Wain, …)
  - [hardphotos/](hardphotos/) **hard**: 10 photographs of famous real places (Grand Canyon, Mount Fuji, Yosemite, Matterhorn, …), with author, license and source in [sources.json](hardphotos/sources.json)
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

- **Reference landscapes in three levels**: easy `landscape-paintings/`, medium `medphotos/`, hard `hardphotos/` (photos with credits), all resized to 1536×1024. Root README now also covers the Nemotron agent loop, watch view and evals merged from `main`.
- **Nemotron agent loop, watch view, evals** (merged from `main`): chained Crusoe Nemotron picture agents with typed edits, `/watch` playback, provisional eval framework.
- **Graded drawing dataset**: 10 tasks from beginner to expert with prompts, success criteria and reference solutions; `dataset` CLI command and `GET /api/dataset`.
- **Human-readable output**: render results gain a text `summary`; recordings save the model's raw output and a `report.md`; CLI `check` is readable by default; new `report` command.
- **Build constrained CMYK drawing sandbox**: renderer, rules, recording, editor, examples.
- **Initial commit**: hello world.
