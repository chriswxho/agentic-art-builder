# Evals (provisional)

A scaffold for scoring recorded composition runs. **The metrics, weights, and
defaults are placeholders that will change.** The goal is a harness where
changing them is cheap.

```text
dataset case ──► recorded run (recordings/<run_id>/) ──► evals ──► report JSON ──► compare across iterations
                 + optional target image
                 + optional person preferences
```

## Run it

```bash
python -m drawing_sandbox.evals metrics                       # list registered metrics
python -m drawing_sandbox.evals run recordings/tahoe-cmyk-remake
python -m drawing_sandbox.evals run recordings/tahoe-cmyk-remake --option step_penalty=0.02 --only steps
python -m drawing_sandbox.evals suite evals/smoke.json --out eval-results/a.json
python -m drawing_sandbox.evals compare eval-results/a.json eval-results/b.json
```

`run` falls back to the manifest's `composition.source_image` when no `--target` is given.

## Case format (what the dataset should produce)

A suite file is `{"cases": [...]}`. Paths resolve relative to the suite file:

```json
{
  "case_id": "tahoe-remake",
  "run": "../recordings/tahoe-cmyk-remake",
  "target_image": "../examples/26-tahoe-day/reference.jpg",
  "reference_steps": ["person-a/layer-1.png", "person-a/layer-2.png"],
  "preferred_step_count": 3,
  "options": {"step_penalty": 0.05, "feature_weights": {"position": 2.0}}
}
```

Only `run` is required. Metrics whose inputs are missing are skipped rather than scored as zero.

## Metrics

Each metric reports a raw `value` and a `score` in [0, 1], where higher is better. Informational metrics have no `score`.
Each category's score is the mean of its metric scores. `overall` is the weighted mean of the category scores (`CATEGORY_WEIGHTS`: image 0.4, steps 0.4, preference 0.2).
If `policy.valid` fails, `overall` is forced to 0.

**Image similarity** (`image_metrics.py`), comparing the final render with the target:

| metric | meaning |
| --- | --- |
| `image.pixel_mae` | Naive per-pixel RGB difference. It mostly measures hatching texture. |
| `image.tone_mae` | The same difference after averaging 8×8 blocks: what the overprint reads as from a distance. |
| `image.ssim` | Structural similarity of luminance on lightly blurred images. |
| `image.skill_vs_blank` | `1 - tone_mae(final) / tone_mae(white canvas)`. The blank canvas is the baseline image: 0 means no better than doing nothing. |
| `image.progress_monotonicity` | Share of steps that move the drawing closer to the target. |

**Step differentiation** (`step_metrics.py`). Each mark in the final scene gets a feature vector: type, ink, fill or outline, position, size, and orientation. It is labelled with the step that added it. Two forces pull against each other:

- **coherence:** marks within a step should be alike, so more steps help;
- **economy:** every extra step costs `step_penalty`.

`J(k) = unexplained_variance(k) + step_penalty · (k − 1)`. Paint order matters, so an oracle uses dynamic programming to find the best *contiguous* split of the final paint order for every k. The argmin of J is the suggested number of layers.

| metric | meaning |
| --- | --- |
| `steps.count` | Recorded steps, with per-step marks added/removed and their inks/types/bbox. |
| `steps.coherence` | R² of the agent's step grouping. |
| `steps.suggested_count` | The oracle's k, the full J(k) curve, and what each suggested layer contains. |
| `steps.count_error` | \|agent's non-empty steps − suggested\|. |
| `steps.objective_gap` | J(agent) − J(oracle). A gap of 0 means the agent's layering is as good as the best split. |
| `steps.visual_impact` | Share of steps that visibly change the pixels. |

**Preference** (`preference_metrics.py`) runs only with per-person data:
`preference.step_count` and `preference.layer_progression`. The second compares the person's cumulative layer images with the agent's step images at the same fraction of progress.

## Knobs to tune

- `step_penalty` controls how much variance a layer must explain to earn its own step. On Tahoe, 0.05 suggests 3 layers and 0.02 suggests 6, which is what the agent recorded.
- `feature_weights` decide what counts as "similar marks". Differences in a feature with a weight of 0 are ignored. **These weights are where person-specific layering should be learned.** Fit them so that the oracle's split reproduces a person's labelled layers.
- `tone_block`, `gap_scale`, `min_step_impact`, and `CATEGORY_WEIGHTS` in `core.py`.

## Adding a metric

```python
from drawing_sandbox.evals import metric

@metric("image.my_metric", "image")
def my_metric(context):
    """One-line description shown by `metrics`."""
    if context.target is None:
        return None                     # skipped
    value = ...
    return value, score_or_None, {"any": "details"}
```

`context` holds `run` (steps with scenes, images, and expanded primitives), `case`, `target` (an RGB float array at canvas size), `options`, and `cached()` for sharing expensive work between metrics.
Import the module in `evals/__init__.py` to register the metric.

## Known gaps / next ideas

- Perceptual similarity (LPIPS/CLIP/VLM judge) once a model dependency is acceptable.
- A palette-reachable baseline: the target quantized to colors that 50% CMYK overprints can actually produce. This gives the best image the policy can reach to compare against.
- Agreement between the agent's layers and a person's labelled layers (for example, adjusted Rand index on the marks), and learning `feature_weights` from that agreement.
- Semantic step-label checks against the composition plan.
