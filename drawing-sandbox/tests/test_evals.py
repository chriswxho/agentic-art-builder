import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from drawing_sandbox.evals import EvalCase, compare, evaluate, evaluate_suite
from drawing_sandbox.evals.step_metrics import oracle_segmentation
from drawing_sandbox.recording import record_composition_step
from drawing_sandbox.server import render_drawing


def dot(x, y, color):
    return {"type": "ellipse", "cx": x, "cy": y, "rx": 6, "ry": 6,
            "fill": color, "stroke": "none", "width": 0, "opacity": 0.5}


def dash(x, y):
    return {"type": "path", "points": [[x, y], [x + 30, y]], "stroke": "#000000",
            "width": 4, "fill": "none", "opacity": 0.5}


CYAN_DOTS = [dot(20 + 12 * i, 30 + 8 * (i % 3), "#00ffff") for i in range(10)]
BLACK_DASHES = [dash(120 + 4 * i, 150 + 6 * i) for i in range(10)]


def scene(nodes):
    return {"version": 1, "canvas": {"width": 256, "height": 256, "background": "#ffffff"},
            "motifs": {}, "nodes": nodes}


def record_run(root, run_id, step_nodes):
    for index, nodes in enumerate(step_nodes):
        drawing = scene(nodes)
        record_composition_step(
            drawing, render_drawing(drawing), run_id=run_id, prompt_set=["test"],
            composition={"layers": len(step_nodes)}, step_label=f"step {index}",
            final=index == len(step_nodes) - 1, root=root)
    return Path(root) / run_id


def save_final_png(run_dir, path):
    Path(path).write_bytes((run_dir / "final-drawing.png").read_bytes())
    return Path(path)


class EvalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_distinct_layers_match_suggested_count(self):
        run = record_run(self.root, "two", [CYAN_DOTS, CYAN_DOTS + BLACK_DASHES])
        metrics = evaluate(EvalCase("two", run))["metrics"]
        self.assertEqual(metrics["steps.suggested_count"]["value"], 2)
        self.assertEqual(metrics["steps.count_error"]["value"], 0)
        self.assertGreater(metrics["steps.coherence"]["value"], 0.8)
        self.assertAlmostEqual(metrics["steps.objective_gap"]["value"], 0, places=6)

    def test_redundant_steps_are_penalized(self):
        steps = [CYAN_DOTS[:5], CYAN_DOTS, CYAN_DOTS + BLACK_DASHES[:5], CYAN_DOTS + BLACK_DASHES]
        run = record_run(self.root, "four", steps)
        metrics = evaluate(EvalCase("four", run))["metrics"]
        self.assertEqual(metrics["steps.count"]["value"], 4)
        self.assertEqual(metrics["steps.suggested_count"]["value"], 2)
        self.assertGreater(metrics["steps.objective_gap"]["value"], 0)
        self.assertLess(metrics["steps.objective_gap"]["score"], 1)

    def test_mixed_steps_are_less_coherent(self):
        mixed = [CYAN_DOTS[:5] + BLACK_DASHES[:5], CYAN_DOTS[:5] + BLACK_DASHES[:5] + CYAN_DOTS[5:] + BLACK_DASHES[5:]]
        clean = record_run(self.root, "clean", [CYAN_DOTS, CYAN_DOTS + BLACK_DASHES])
        messy = record_run(self.root, "messy", mixed)
        clean_score = evaluate(EvalCase("clean", clean))["metrics"]["steps.coherence"]["value"]
        messy_score = evaluate(EvalCase("messy", messy))["metrics"]["steps.coherence"]["value"]
        self.assertGreater(clean_score, messy_score + 0.5)

    def test_revisions_are_tolerated(self):
        run = record_run(self.root, "revise", [CYAN_DOTS, CYAN_DOTS[2:] + BLACK_DASHES])
        details = evaluate(EvalCase("revise", run))["metrics"]["steps.count"]["details"]
        self.assertEqual(details["steps"][1]["removed"], 2)
        self.assertEqual(details["steps"][1]["added"], 10)

    def test_identical_target_scores_perfectly(self):
        run = record_run(self.root, "same", [CYAN_DOTS, CYAN_DOTS + BLACK_DASHES])
        target = save_final_png(run, self.root / "target.png")
        report = evaluate(EvalCase("same", run, target_image=target))
        metrics = report["metrics"]
        self.assertAlmostEqual(metrics["image.pixel_mae"]["value"], 0, places=6)
        self.assertAlmostEqual(metrics["image.skill_vs_blank"]["value"], 1, places=6)
        self.assertAlmostEqual(metrics["image.ssim"]["value"], 1, places=4)
        self.assertEqual(metrics["image.progress_monotonicity"]["value"], 1)
        self.assertIn("image", report["category_scores"])

    def test_skill_is_relative_to_blank_canvas(self):
        run = record_run(self.root, "dark", [CYAN_DOTS])
        target = self.root / "black.png"
        Image.new("RGB", (256, 256), "#000000").save(target)
        skill = evaluate(EvalCase("dark", run, target_image=target))["metrics"]["image.skill_vs_blank"]
        self.assertGreater(skill["value"], 0)
        self.assertLess(skill["value"], 0.1)

    def test_preferences_only_run_when_supplied(self):
        run = record_run(self.root, "pref", [CYAN_DOTS, CYAN_DOTS + BLACK_DASHES])
        self.assertNotIn("preference.step_count", evaluate(EvalCase("pref", run))["metrics"])
        layer = save_final_png(run, self.root / "layer.png")
        metrics = evaluate(EvalCase("pref", run, reference_steps=[layer],
                                    preferred_step_count=3))["metrics"]
        self.assertEqual(metrics["preference.step_count"]["value"], 1)
        self.assertAlmostEqual(metrics["preference.layer_progression"]["value"], 0, places=6)

    def test_oracle_segmentation_is_exact(self):
        features = np.array([[0.0], [0.1], [0.0], [5.0], [5.1], [9.0]])
        curve, boundaries = oracle_segmentation(features, 4)
        self.assertEqual(boundaries[2], [0, 3, 5])
        self.assertTrue(all(a >= b for a, b in zip(curve, curve[1:])))

    def test_compare_reports_deltas(self):
        run = record_run(self.root, "cmp", [CYAN_DOTS, CYAN_DOTS + BLACK_DASHES])
        before = evaluate_suite([EvalCase("cmp", run)])
        after = evaluate_suite([EvalCase("cmp", run, options={"step_penalty": 1.0})])
        rows = {row["metric"]: row for row in compare(before, after)}
        self.assertEqual(rows["steps.suggested_count"]["before"], 2)
        self.assertEqual(rows["steps.suggested_count"]["after"], 1)
        json.dumps(after, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
