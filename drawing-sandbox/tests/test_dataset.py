import json
import tempfile
import unittest
from pathlib import Path

from drawing_sandbox.dataset import export_dataset, task_card, tasks
from drawing_sandbox.server import render_drawing


class DatasetTests(unittest.TestCase):
    def test_levels_run_easiest_first(self):
        levels = [task["level"] for task in tasks()]
        self.assertEqual(levels, sorted(levels))
        self.assertEqual(tasks()[0]["tier"], "beginner")
        self.assertEqual(tasks()[-1]["tier"], "expert")

    def test_every_reference_passes_policy_and_task_limits(self):
        for task in tasks():
            with self.subTest(task=task["id"]):
                result = render_drawing(task["reference_scene"], task["constraints"] or None)
                self.assertTrue(result["checks"]["valid"], result["checks"]["violations"])

    def test_references_get_harder(self):
        marks = [render_drawing(t["reference_scene"])["checks"]["metrics"]["expanded_shapes"] for t in tasks()]
        self.assertLess(marks[0], marks[4])
        self.assertLess(marks[4], marks[-1])

    def test_task_card_hides_reference(self):
        card = task_card(tasks()[0])
        self.assertNotIn("reference_scene", card)
        self.assertIn("prompt", card)

    def test_export_writes_index_and_references(self):
        with tempfile.TemporaryDirectory() as folder:
            index = export_dataset(folder)
            root = Path(folder)
            self.assertEqual(len(index), len(tasks()))
            self.assertEqual(len((root / "tasks.jsonl").read_text().splitlines()), len(tasks()))
            first = root / index[0]["path"]
            self.assertTrue((first / "reference/drawing.png").is_file())
            self.assertTrue(json.loads((first / "reference/checks.json").read_text())["hard_checks"]["max_shapes"])
            self.assertIn("Single dot", (root / "README.md").read_text())


if __name__ == "__main__":
    unittest.main()
