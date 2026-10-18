# test_4: a tree of Super-VL agents

test_4 recreates a reference picture with a **tree of model calls** instead of a fixed number of rounds:
- **A director** plans the picture.
- **Area leads** outline the flat-color patches in their part.
- **Painters** propose shapes for each patch, several at once.
- **Refiners** split any patch that's still wrong into smaller patches, recursively.

Every shape is measured before it's kept, so the canvas only ever gets closer to the reference.

The goal is a strict Lab score of at least **0.9** on the easy images. With perfect shapes and the best ink recipes, the ceiling is about 0.97, so 0.9 is reachable but needs precise outlines.

- Design rationale and the original defaults: [DESIGN.md](DESIGN.md)
- Harness: [run_test_4.py](run_test_4.py)
- Per-run agent report generator: [agent_report.py](agent_report.py)

## Results so far (3 easy images)

| Image | Lab score | Commits | Calls | Time | Super-VL |
| --- | --- | --- | --- | --- | --- |
| Sea sunset | **0.807** | 21 | 170 of 400 | 15.5 min | 70/100 reference match, 75/100 composition |
| Snowy mountains | 0.595 | 15 | 244 of 400 | 24.4 min | 50/100 reference match, 55/100 composition |
| Green hills | 0.715 | 14 | 209 of 400 | 18.1 min | 65/100 reference match, 70/100 composition |

For comparison, test_3 (fixed rounds, one shape per painter) sat at about 0.50–0.69 on the same images after 7–9 rounds, and test_1 at about 0.01. None of the test_4 runs reached 0.9; the best is the sea sunset at 0.81. All three stopped with budget left because the tree ran out of patches to refine.

Agent reports: [sea sunset](runs/easy-sea-sunset/AGENT_REPORT.md) · [green hills](runs/easy-green-hills/AGENT_REPORT.md) · [snowy mountains](runs/easy-snowy-mountains/AGENT_REPORT.md)

![sea sunset: reference vs final](runs/easy-sea-sunset/comparison.png)

## The levels of the tree

```text
depth 0   DIRECTOR ─────────────── 1 call: areas, back-to-front order, base colors
             │
             ├── free BASE FILLS ── 0 calls: background boxes filled with an exact ink recipe
             │
depth 1   AREA LEADS ──────────── 1 call each, all areas at once: draw patch outlines
             │
depth 2   PATCHES ─┬─ 2 PAINTER candidates in parallel → measured → best kept if it helps
             │      └─ still far off? ─┐
depth 3      │                     REFINER: draws sub-patches inside the patch
             │                         │
depth 4      │                     SUB-PATCHES ─ 2 painter candidates each → keep if better
             ▼
          canvas (every commit = one recorded revision)
```

| Level | Agent | Input it sees | What it returns | Calls |
| --- | --- | --- | --- | --- |
| 0 | **Director** | the whole reference | 3–8 content areas (boxes), each with a paint order (0 = background) and a base color | 1 |
| — | **Base fill** (harness, no model) | the director's order-0 boxes | fills each box with the best 1–2 ink layers for its color | 0 |
| 1 | **Area lead** | a 4-panel sheet: reference with the area outlined, reference zoom, flat-color zoom, canvas zoom | 2–12 **patches**, each a polygon outline it draws, a target color and an order | 1–3 |
| 2 and 4 | **Patch painter** (2 candidates per patch) | a 3-panel sheet zoomed to the patch, its outline, the measured colors | 1–3 shapes (polygons, ellipses, paths) with inks and opacities | 1–3 each |
| 3 | **Refiner** | the same sheet as an area lead, zoomed to one patch | sub-patches inside that patch | 1–3 |

### Rules every level follows

- **The model draws every outline.** The harness never invents geometry. It only offers measuring tools: `measure` (reference colors and error inside a shape), `recipe` (best ink layers for a target color), and `preview` (score change if shapes were added, without committing).
- **Keep if better:** a candidate commits only if, against the canvas *as it is at that moment*, its patch's color difference drops and the whole-picture score doesn't. Otherwise it's dropped.
- **Back to front:** all areas work at once, but an area only commits after every lower-order area has finished its first pass. Within an area, patches commit in their order.
- **Refine where it's still wrong:** after its first pass, a patch whose remaining color difference (Lab ΔE) is above 8 gets a refiner, up to depth 4.
- **Stop** when the Lab score reaches 0.9, the call budget runs out, or the tree has nothing left to refine.

## How a run unfolds over time (sea sunset, full run)

The scores come from `runs/easy-sea-sunset/scores.json`; times are minutes from the start.

| Time | What happened | Lab |
| --- | --- | --- |
| 0.0 | **Director** splits the picture into sky (order 0), water (0), horizon line (1) and sun (2), with base colors. | 0.00 |
| 0.4 | **Free base fills:** the sky box gets yellow 0.85 then magenta 0.30, almost exact (ΔE 55 → 7). The water box gets cyan then magenta, too light, because the director's guessed color was off (ΔE 64 → 45). | 0.60 |
| 0.4–3 | **All 4 area leads work in parallel** and outline 13 patches. Each patch's 2 painters run at the same time. The horizon and sun areas finish early but wait to commit until the sky and water (order 0) finish their first pass. | |
| 3.2 | **Water commits:** the main lake and the dark bottom edge. | 0.66 |
| 4.2 | First **refiners** (depth 3) on the still-wrong water draw sub-patches; depth-4 painters fix the main ocean blue. | 0.66 |
| 7.3 | The sky area's first pass finishes, so the **sky bands and horizon commit**. The horizon line and the strip just below it give the biggest single gain (ΔE 89 → 25). | 0.69 |
| 9.5 | **Biggest jump:** a depth-4 refiner repaints the main lake body (ΔE 36 → 17) and its dark bottom strip (25 → 5). | 0.80 |
| 10.7–15.5 | Depth-4 refiners make small fixes: horizon strip pieces, the band under the sun, the **sun semicircle**, sky bands. | 0.81 |
| 15.5 | Nothing left above the refine threshold within depth 4: **tree finished** with 230 of 400 calls unused. | **0.807** |

**What the agents did, by the numbers:**

| Agent type | Count | Model calls | Outcome |
| --- | --- | --- | --- |
| Director | 1 | 1 | 1 plan |
| Base fills | 2 | 0 | 2 committed |
| Area leads | 4 | 9 | 13 patches outlined |
| Refiners | 9 | 20 | 25 sub-patches outlined |
| Patches | 38 (13 at depth 2, 25 at depth 4) | — | **19 committed, 19 not kept** |
| Painter candidates | 76 | 140 | 19 committed, 56 proposed but not kept, 1 failed |

**Takeaways from this run:**
- **Most of the score came from free fills and refiners.** Free fills gave 0.60 for no calls; depth-4 refiners gave the largest late gains.
- **Painters often make things worse.** Half the patches never committed because both candidates scored *negative* gains (for example on the thin sky bands and around the sun). The gate is what keeps those out.
- **The order barrier costs time.** The sky and horizon commits waited until 7.3 minutes for the order-0 areas to finish their first pass.
- **What's still wrong:**
  - The sun is a stepped polygon with an olive tint.
  - The three yellow reflection bars on the water were never committed.
  - The horizon band is too thin.

## How this version came about

The first build (kept in `cancelled/`) painted areas one order group at a time and patches one order at a time, with 3 candidates and 3 tool turns per patch. It managed only 3 commits in 12 minutes. The current version changed four things:
1. **Free base fills:** the director's background boxes are filled with an exact ink recipe, using no model calls.
2. **Areas in parallel:** all areas start at once.
3. **Patches in parallel:** all of an area's patch candidates are generated at the same time, while commits still land back to front.
4. **Cheaper patches:** 2 candidates and 2 tool turns per patch.

## Known problems and next fixes

1. **Base colors come from the director's guess.** The water fill was far off. Fix: fill the director's box with the reference's *measured* main color. The geometry stays the model's.
2. **The director is a single point of failure.** If its call fails, the run falls back to top and bottom halves. Fix: run 2 directors at once and take the first valid plan.
3. **Thin and small shapes rarely commit** (reflection bars, horizon band, sun). Painters' outlines for small patches are often off by more than the patch's width. The candidates for these patches mostly scored negative gains.
4. **Calls slow down under shared load.** Several runs at once share one endpoint; calls took 30–80 seconds instead of about 20.
5. **Unused budget.** The sea sunset stopped with 230 of 400 calls unused, because no patch was above the refine threshold within depth 4. Allowing more depth, or another pass on the worst areas, would put that budget to work.

## Run it

```bash
drawing-sandbox/.venv/bin/python test_4/run_test_4.py --only easy-sea-sunset
drawing-sandbox/.venv/bin/python test_4/agent_report.py test_4/runs/easy-sea-sunset
```

Options: `--budget` (default 400 calls), `--target` (default 0.9 Lab), `--name` (output subfolder, default `runs`).

An existing run folder is never overwritten. Delete it to redo the run.

## Files per run (`runs/<run_id>/`)

| File | What it holds |
| --- | --- |
| `AGENT_REPORT.md` | readable account of what every agent did (generated by `agent_report.py`) |
| `tree.json`, `tree.html` | every agent: parent, depth, status, calls, measured gains; `tree.html` is a collapsible view |
| `tree/nodes/<id>.json` | each agent's full conversation: prompts, raw answers, tool calls and results, token usage, validation errors |
| `tree/sheets/<id>.png` | the exact image each agent was shown |
| `scores.json` | Lab and lenient score after every commit, with timestamps |
| `steps/`, `manifest.json`, `handoffs/` | each commit as a standard recorded revision (scene, SVG, PNG, checks, context, proposal) |
| `comparison.png`, `milestones.png`, `timelapse.gif` | reference vs final; progress across commits; animated replay of every commit (made by [timelapse.py](timelapse.py)) |
| `evals.txt`, `handoffs/evaluation.json` | offline evals; Super-VL's before/after verdict |
| `skills-log.jsonl` | one line per kept patch: target color, shapes and inks, improvement. Raw material for a skills file |

## Settings

| Setting | Value |
| --- | --- |
| Model | Super-VL for every node |
| Candidates per patch | 2 |
| Tool turns per node | 2 (`measure`, `recipe`, `preview`) |
| Output limit per call | 16,384 tokens |
| Call timeout | 240 s |
| Calls in flight | 24 |
| Call budget | 400 per image |
| Max depth / refine threshold | 4 / patch ΔE > 8 |
| Mark rules (local override) | CMYK on white, opacity 0.1–1.0, marks up to the full canvas, strokes ≤ 48 |

These mark rules and the use of Super-VL as the painter depart from AGENTS.md (which specifies 0.5 opacity, 96-unit marks and Nano as the generator). They apply only inside this test's process; the shared policy is unchanged.
