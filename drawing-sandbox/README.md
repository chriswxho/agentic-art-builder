# Drawing Sandbox

A small tool for **assembling vector drawings, rendering them, and recording how they were built**. The visual style uses a deliberately narrow vocabulary: small cyan, magenta, yellow, and black marks overprinted on white.

![Starter drawings](preview.png)

## Run it

Use Python 3.10 or newer. In this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m drawing_sandbox
```

Open **http://127.0.0.1:8766** if the browser does not open automatically. The only runtime dependencies are **NumPy and Pillow**. Rendering runs on the CPU; model weights and GPU setup are unnecessary.

Choose a starter drawing, add shapes, change object transforms, or edit its JSON. The editor offers undo and SVG, PNG, and JSON exports. Work stays in the current browser page; use the JSON download to keep a drawing before closing or reloading it.

## The environment contract

**Prompt → compose → render → record → revise.**

An agent such as Nemotron can produce the same scene JSON that the editor uses. One tool call returns an SVG, a PNG, and structured checks:

```python
from drawing_sandbox.server import render_drawing

result = render_drawing(scene)
# result["checks"]: valid, hard_checks, metrics, violations
# result["svg"] and result["png_data_url"] exist when the drawing is valid
# result["summary"]: the same checks plus an outline of the scene, as plain text
print(result["summary"])
```

The summary looks like:

```text
PASS: 10 of 10 rules passed
Stats: marks 9, inks used 3, widest mark 62, thickest stroke 0

Canvas: 512×512, background white
Marks: 9 total (9 ellipse)
Ink use: magenta 4, cyan 3, yellow 2
Top-level nodes (9), in paint order:
    1. ellipse at (106, 118), radius 31, cyan fill
    2. ellipse at (137, 118), radius 31, magenta fill
    ...
```

Pass the returned PNG to the VLM, ask it to revise the scene, and render again. Your agent controls that loop. This package makes no model/API calls and does not run an optimizer.

The same operation is available over HTTP:

```text
POST http://127.0.0.1:8766/api/render
Content-Type: application/json

{"scene": {...}}
```

The server binds only to localhost. `GET /api/examples` returns three compliant starter scenes. Optional request constraints can make a task stricter, but cannot relax the base instruction set.
`GET /api/instruction-set` returns the exact machine-readable policy agents should follow.

## Constrained instruction set

Agent-facing renders enforce these rules:

- The canvas background is white (`#ffffff`).
- Shape paint is limited to cyan (`#00ffff`), magenta (`#ff00ff`), yellow (`#ffff00`), and black (`#000000`).
- Every visible primitive has `opacity: 0.5`.
- A transformed primitive may span at most 96 canvas units.
- A transformed stroke may be at most 12 canvas units wide.
- Expanded scenes remain capped at 512 primitive shapes.

This makes color mixing, density, and form emerge through repeated small marks and source-over layering. Low-level structural utilities remain useful for inspecting older artifacts, while the HTTP API, `render_drawing`, and command-line render/check commands enforce the current policy.

## Drawing vocabulary

- **Path:** control points, stroke color, width; optional smoothing and fill.
- **Ellipse and polygon:** simple filled or outlined shapes.
- **Group:** move, rotate, and scale several shapes together.
- **Motif call:** use a manually defined reusable group with numeric arguments.

For example:

```json
{
  "version": 1,
  "canvas": {"width": 512, "height": 512, "background": "#ffffff"},
  "motifs": {},
  "nodes": [
    {"type": "ellipse", "cx": 240, "cy": 220, "rx": 30, "ry": 30,
     "fill": "#00ffff", "stroke": "none", "width": 0, "opacity": 0.5},
    {"type": "ellipse", "cx": 270, "cy": 220, "rx": 30, "ry": 30,
     "fill": "#ff00ff", "stroke": "none", "width": 0, "opacity": 0.5},
    {"type": "path", "points": [[220, 290], [250, 275], [290, 290]],
     "smooth": true, "stroke": "#000000", "width": 6,
     "fill": "none", "opacity": 0.5}
  ]
}
```

The checks cover finite coordinates, legal shapes and colors, motif arguments/references, cycles, geometry/expansion limits, the CMYK palette, opacity, and transformed mark size. They establish that a drawing follows the construction contract; visual meaning and attractiveness remain questions for the viewer or VLM. The renderer permits geometry outside the canvas and clips it.

## Recording progress and provenance

Use `POST /api/record` for every meaningful composition state. It accepts the same `scene` as `/api/render`, plus a stable run ID, the complete prompt set, a composition plan, and a step label:

```json
{
  "run_id": "lake-study-01",
  "prompt_set": ["Build a lake scene using layered CMYK marks."],
  "composition": {
    "layers": ["water marks", "mountain marks", "foreground accents"]
  },
  "step_label": "Water and distant ridge",
  "final": false,
  "model_output": "Optional: the model's raw reply for this step.",
  "scene": {}
}
```

Accepted states are appended under `recordings/<run_id>/steps/` as scene JSON, SVG, PNG, and checks. Send the accepted scene with `"final": true` to also save `final-scene.json`, `final-drawing.svg`, and `final-drawing.png`. `manifest.json` preserves the instruction set, prompts, composition plan, step labels, metrics, timestamps, and final-step pointer. If a step includes `model_output`, it is saved verbatim as `steps/NNN-model-output.txt`.

After every step the run also gets a readable `report.md`: prompts, plan as bullets, and for each step its image, pass/fail summary, model output, and an outline of what was drawn. See [the sample report](recordings/cmyk-overprint-demo/report.md). Regenerate it with `python -m drawing_sandbox report recordings/<run_id>`.

## Terminal use

```bash
python -m drawing_sandbox examples --out examples
python -m drawing_sandbox render examples/cmyk-overprint/scene.json --out output
python -m drawing_sandbox check examples/cmyk-overprint/scene.json          # readable summary
python -m drawing_sandbox check examples/cmyk-overprint/scene.json --json   # raw checks
python -m drawing_sandbox report recordings/cmyk-overprint-demo
python -m unittest discover -s tests -v
```

SVG, PNG, and JSON starter files are included under `examples/`. They are authored illustrations, ready to edit. The reusable motifs in them are predefined drawing parts.

The earlier Tahoe study is preserved under `examples/26-tahoe-day/` with its source `reference.jpg`, vector scene, rendered PNG/SVG, and construction timelapse. It predates the CMYK policy, so it is reference material rather than an accepted input to the current agent-facing renderer.

## Later domains

Keep the same outer interaction—submit a construction, render it, return checks. A graph domain can add node/edge construction and graph predicates. A small 3D domain can add primitive objects, a fixed orthographic camera, and geometric checks. Those implementations are future work; the first environment stays small.

The 3D concern is manageable by constraining the scene and render budget: a few primitives and one low-resolution view is a sensible first target. Benchmark an actual renderer on your Mac before adding multi-view search or more elaborate lighting.
