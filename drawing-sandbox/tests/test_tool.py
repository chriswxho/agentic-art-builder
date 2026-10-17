import base64
import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from drawing_sandbox.examples import examples
from drawing_sandbox.recording import record_composition_step
from drawing_sandbox.server import render_drawing


class DrawingToolTests(unittest.TestCase):
    def test_examples_return_images_and_structured_checks(self):
        for example in examples():
            with self.subTest(example=example["id"]):
                result=render_drawing(example["scene"])
                self.assertTrue(result["checks"]["valid"])
                self.assertGreater(result["checks"]["metrics"]["expanded_shapes"],0)
                self.assertIn("<svg",result["svg"])
                png=base64.b64decode(result["png_data_url"].split(",",1)[1])
                canvas=example["scene"]["canvas"]
                self.assertEqual(Image.open(io.BytesIO(png)).size,
                                 (canvas["width"],canvas["height"]))

    def test_invalid_drawing_returns_errors_without_images(self):
        result=render_drawing({"canvas":{},"nodes":[{"type":"unsupported"}]})
        self.assertFalse(result["checks"]["valid"])
        self.assertTrue(result["checks"]["violations"])
        self.assertNotIn("svg",result)
        self.assertNotIn("png_data_url",result)

    def test_task_constraints_are_applied_before_render(self):
        result=render_drawing(examples()[0]["scene"],{"max_shapes":2})
        self.assertFalse(result["checks"]["valid"])
        self.assertNotIn("png_data_url",result)

    def test_agent_policy_cannot_be_relaxed(self):
        source={
            "version":1,
            "canvas":{"width":128,"height":128,"background":"#ffffff"},
            "motifs":{},
            "nodes":[{"type":"ellipse","cx":64,"cy":64,"rx":20,"ry":20,
                      "fill":"#ff0000","stroke":"none","width":0,"opacity":.5}],
        }
        result=render_drawing(source,{"palette":["#ffffff","#ff0000"]})
        self.assertFalse(result["checks"]["valid"])
        self.assertFalse(result["checks"]["hard_checks"]["shape_palette"])

    def test_recording_preserves_prompts_progress_and_final(self):
        source=examples()[0]["scene"]
        rendered=render_drawing(source)
        prompt_set=["Build an image from overlapping CMYK marks."]
        composition={"layers":["overprint grid","registration marks"]}
        with tempfile.TemporaryDirectory() as folder:
            first=record_composition_step(
                source,rendered,run_id="test-run",prompt_set=prompt_set,
                composition=composition,step_label="First pass",root=folder,
            )
            final=record_composition_step(
                source,rendered,run_id="test-run",prompt_set=prompt_set,
                composition=composition,step_label="Accepted",final=True,root=folder,
            )
            manifest=json.loads(Path(final["manifest"]).read_text())
            self.assertEqual(manifest["prompt_set"],prompt_set)
            self.assertEqual(manifest["composition"],composition)
            self.assertEqual(len(manifest["steps"]),2)
            self.assertEqual(manifest["final_step"],1)
            run=Path(first["directory"])
            self.assertTrue((run/"steps/000-drawing.png").is_file())
            self.assertTrue((run/"final-drawing.png").is_file())
            self.assertTrue((run/"final-drawing.svg").is_file())
            self.assertTrue((run/"final-scene.json").is_file())


if __name__=="__main__":
    unittest.main()
