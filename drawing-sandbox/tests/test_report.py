import tempfile
import unittest
from pathlib import Path

from drawing_sandbox.examples import examples
from drawing_sandbox.recording import record_composition_step
from drawing_sandbox.report import color_name, describe_checks, describe_scene
from drawing_sandbox.server import render_drawing


class ReportTests(unittest.TestCase):
    def test_color_names(self):
        self.assertEqual(color_name("#00FFFF"), "cyan")
        self.assertEqual(color_name("#f0f"), "magenta")
        self.assertEqual(color_name("#123456"), "#123456")

    def test_render_result_includes_readable_summary(self):
        result = render_drawing(examples()[0]["scene"])
        self.assertTrue(result["summary"].startswith("PASS"))
        self.assertIn("ellipse at (106, 118), radius 31, cyan fill", result["summary"])

    def test_failed_checks_list_broken_rules(self):
        scene = {"version": 1, "canvas": {"width": 128, "height": 128, "background": "#ffffff"},
                 "motifs": {}, "nodes": [{"type": "ellipse", "cx": 64, "cy": 64, "rx": 20, "ry": 20,
                                          "fill": "#ff0000", "stroke": "none", "width": 0, "opacity": .5}]}
        text = describe_checks(render_drawing(scene)["checks"])
        self.assertTrue(text.startswith("FAIL"))
        self.assertIn("shape_palette", text)
        self.assertIn("1 total (1 ellipse)", describe_scene(scene))

    def test_recording_saves_model_output_and_report(self):
        scene = examples()[0]["scene"]
        with tempfile.TemporaryDirectory() as folder:
            saved = record_composition_step(
                scene, render_drawing(scene), run_id="readable", prompt_set=["Draw dots."],
                composition={"layers": ["dots"]}, step_label="Dots", final=True,
                model_output="I placed a grid of overlapping circles.", root=folder,
            )
            run = Path(saved["directory"])
            self.assertEqual((run / "steps/000-model-output.txt").read_text(),
                             "I placed a grid of overlapping circles.")
            report = Path(saved["report"]).read_text()
            self.assertIn("## Step 1: Dots (final)", report)
            self.assertIn("I placed a grid of overlapping circles.", report)
            self.assertIn("PASS", report)

    def test_model_output_must_be_text(self):
        scene = examples()[0]["scene"]
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
            record_composition_step(scene, render_drawing(scene), run_id="bad", prompt_set=["x"],
                                    composition={}, model_output=42, root=folder)


if __name__ == "__main__":
    unittest.main()
