# test_4 design: a tree of Super-VL calls

Goal: easy images recreated at `skill_vs_blank` ≥ 0.9. The best achievable with perfect shapes and the best ink recipes is 0.91–0.99, so this needs near-exact outlines and colors. Medium and hard images use the same machinery with more depth.

Decided so far:
- **Models:** Super-VL only (`nvidia/NVIDIA-Nemotron-3.5-Super-VL`).
- **Outlines:** the model draws every outline itself. The harness provides only measuring and color tools.
- **Stopping:** a branch stops when it reaches its target score or stops improving. That logic is already in test_3.

## Why a tree instead of rounds

test_3 runs fixed rounds: every painter adds one shape, everything waits for the slowest painter, and work is spread evenly whether or not a region needs it. Its scores also fall after good rounds (green hills: 0.60 → 0.50), because nothing checks whether a shape helped.

A tree changes three things:

1. **Work goes where the error is.** A part that already matches stops early; a part that's still wrong gets split into smaller parts and gets more calls. Depth grows only where it's needed.
2. **Every leaf picks the best of several tries.** The leaf branches into several candidate proposals, the harness measures each one, and only a candidate that improves the canvas is kept. The canvas can only get better.
3. **Independent branches run at the same time.** Many calls are in flight at once, so wall-clock time follows the tree's depth, not its total size.

## The tree

```text
                         ┌───────────────────────────┐
 depth 0                 │ DIRECTOR (1 call)         │  palette, back-to-front order,
                         │ whole reference           │  4–8 content areas + budgets
                         └─────────────┬─────────────┘
              ┌────────────────────────┼────────────────────────┐
 depth 1  ┌───▼────┐              ┌────▼───┐               ┌────▼───┐
          │ AREA   │  sky         │ AREA   │ mountains     │ AREA   │ grass …   (parallel)
          │ LEAD   │              │ LEAD   │               │ LEAD   │
          └───┬────┘              └───┬────┘               └───┬────┘
              │ draws PATCHES: polygon outline + target color + order
     ┌────────┼────────┐        ┌─────┼──────┐
 d2  ▼        ▼        ▼        ▼     ▼      ▼
   PATCH    PATCH    PATCH    PATCH PATCH  PATCH          (parallel where they don't overlap)
     │ fork N candidates (N=3)
     ├─ cand A ─┐
     ├─ cand B ─┼─► harness: render each, measure Δ score on the patch → keep best if it improves
     └─ cand C ─┘
     │
     ▼ still too far off?  (patch error > threshold, budget left, depth < max)
   REFINE: the patch becomes a mini area lead → draws sub-patches inside it → recurse
```

### Node types

| Node | Sees | Must return | Tools |
| --- | --- | --- | --- |
| **Director** (depth 0) | the whole reference | painting brief: subject, 4–8 content areas (boxes), back-to-front order, per-area call budget, palette notes | `measure`, `recipe` |
| **Area lead** | sheet: area of reference, canvas, flat-color simplification | 3–15 **patches**: each a polygon outline (the model draws it), a target color, a paint order, and whether it's background or detail | `measure`, `recipe` |
| **Patch painter** (leaf) | sheet zoomed to the patch | 1–3 shapes (layers) filling the patch, using the outline from its parent or a refinement of it | `measure`, `recipe`, `preview` |
| **Refiner** (a patch that recurses) | same as area lead, zoomed to the patch | sub-patches inside the patch | `measure`, `recipe` |
| **Critic** (optional, one per area at the end) | area before/after | "does it read as a tree / lake?", plus the 1–2 patches to redo | none |

### Tools (measuring and color only; the model draws everything)

The model calls tools by returning a JSON action instead of a final answer; the harness answers and the call continues, up to 3 tool turns per node. (We can switch to the API's native tool calling if Crusoe's Super-VL supports it.)

| Tool | Input | Returns |
| --- | --- | --- |
| `measure` | a box or polygon | reference's dominant colors there (with area share), canvas's current average, color difference (Lab ΔE), `skill_vs_blank` inside it |
| `recipe` | target hex, and optionally a polygon (to use the canvas color under it) | best 1–3-layer ink/opacity recipe over what's there now, with the expected ΔE |
| `preview` | a list of shapes | renders them on a copy of the canvas and returns the score change inside the patch and over the whole picture. It doesn't commit anything. |

`preview` is a measuring tool: the model can test an outline before committing to it. The harness never edits or suggests geometry.

### Commit rules (deterministic, in code)

- **Keep if better:** a candidate is committed only if it lowers the error inside its patch and doesn't raise the whole-picture error. Otherwise the candidate is dropped, and its node gets the numbers back to retry or give up.
- **Paint order:** commits are applied back-to-front in the director's order. Patches that overlap within an area go one after another; patches that don't overlap commit in parallel.
- **Undo:** a refiner may remove shapes inside its patch (`remove_objects`) if measurement shows they're hurting.
- **Stopping:** a node stops when its patch reaches the target or after 2 non-improving tries. The whole run stops at the global target, or when the budget runs out.

### Budgets and limits (per image, defaults to tune)

| Limit | Default |
| --- | --- |
| Max depth | 4 (director → area → patch → sub-patch → sub-sub-patch) |
| Candidates per leaf | 3 |
| Tool turns per node | 3 |
| Total calls | 400 for easy images, 800 for hard |
| Calls in flight | 24, shared across the whole tree |
| Refine threshold | patch ΔE > 8, or patch score < 0.9 |

Rough cost for an easy image: 1 director + 6 area leads + ~30 patches × 3 candidates + ~20 refinements ≈ 120–200 calls. That's about 4 levels deep at ~40 s per call with tool turns, so roughly **5–10 minutes per image**. test_3 takes about an hour for 40 rounds.

### What gets recorded

- **`tree.json`:** every node, with its parent, depth, type, prompt, tool calls and results, candidates, measured gains, and whether it was committed.
- **`tree.html`:** a collapsible view of the tree with thumbnails of each node's before and after.
- **Standard recording:** every commit is a revision through the orchestrator (validated, rendered, recorded), plus a per-depth milestone strip and the evals.
- **Skill notes:** each committed leaf appends one line (patch type, color, recipe, shape kind, gain). That's the raw material for the skills file you want to grow.

## Open decisions

1. **Target metric.** `skill_vs_blank` compares 8×8-pixel block averages, so it's lenient on dark, mid-tone images: the Grand Canyon scored 0.78 as gray blocks. Should the 0.9 target use it, or a stricter perceptual (Lab) color difference over finer blocks? I'd report both and target the stricter one.
2. **Budget.** Is 400 calls per easy image acceptable, or should it be capped lower?
3. **Tool calling.** Should tools be JSON actions in the reply (works today), or should I first check whether Crusoe's Super-VL supports native function calling?
4. **Scope of the first test.** Should the first run be the 3 easy images only, to tune the thresholds, before running medium and hard?
