from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from drawing_sandbox.iteration_context import (
    PictureContextError,
    PROTOCOL_VERSION,
    apply_and_record,
    apply_proposal,
    create_context,
    initialize_recorded_context,
    materialize_scene,
    read_context,
    state_sha256,
)


def mark(x, color="#00ffff"):
    return {
        "type": "ellipse", "cx": x, "cy": 64, "rx": 10, "ry": 10,
        "fill": color, "stroke": "none", "width": 0, "opacity": .5,
    }


def base_context(run_id="agent-test"):
    scene = {
        "version": 1,
        "canvas": {"width": 128, "height": 128, "background": "#ffffff"},
        "motifs": {},
        "nodes": [mark(20), mark(45, "#ff00ff"), mark(75, "#ffff00"), mark(105, "#000000")],
    }
    return create_context(
        scene,
        run_id=run_id,
        prompt_set=["Build with small layered CMYK marks."],
        composition={"layers": ["left", "right"]},
        layer_specs=[
            {"layer_id": "left", "label": "Left marks", "end_index": 2},
            {"layer_id": "right", "label": "Right marks", "end_index": 4},
        ],
    )


def proposal(revision, skill, payload, suffix):
    return {
        "protocol_version": PROTOCOL_VERSION,
        "agent_id": f"agent-{suffix}",
        "mutation_id": f"mutation-{suffix}",
        "base_revision": revision,
        "critique": "The composition can be made easier to refine.",
        "rationale": "Use one typed operation while preserving renderer constraints.",
        skill: payload,
    }


class PictureContextTests(unittest.TestCase):
    def test_four_skills_form_a_revision_checked_chain(self):
        context = base_context()
        self.assertEqual(context["state_sha256"], state_sha256(context))

        context, _ = apply_proposal(context, proposal(0, "split_layers", {
            "source_layer_id": "left",
            "parts": [
                {"layer_id": "left-cyan", "label": "Cyan", "object_ids": ["obj-0000"]},
                {"layer_id": "left-magenta", "label": "Magenta", "object_ids": ["obj-0001"]},
            ],
        }, "split"))
        self.assertEqual([layer["layer_id"] for layer in context["layers"]],
                         ["left-cyan", "left-magenta", "right"])

        added = {
            "object_id": "agent-dot",
            "renderer_node": mark(58, "#00ffff"),
            "intent": "Bridge the two halves",
            "tags": ["type:ellipse", "fill:#00ffff"],
        }
        context, _ = apply_proposal(context, proposal(1, "add_objects", {
            "layer_id": "left-magenta", "before_object_id": "", "objects": [added],
        }, "add"))
        self.assertIn("agent-dot", [item["object_id"] for layer in context["layers"] for item in layer["objects"]])

        context, _ = apply_proposal(context, proposal(2, "remove_objects", {
            "object_ids": ["obj-0000"],
        }, "remove"))
        self.assertNotIn("obj-0000", [item["object_id"] for layer in context["layers"] for item in layer["objects"]])

        context, rendered = apply_proposal(context, proposal(3, "combine_layers", {
            "source_layer_ids": ["left-cyan", "left-magenta"],
            "output_layer_id": "left-refined", "output_label": "Refined left marks",
        }, "combine"))
        self.assertEqual(context["revision"], 4)
        self.assertEqual([layer["layer_id"] for layer in context["layers"]],
                         ["left-refined", "right"])
        self.assertTrue(rendered["checks"]["valid"])
        self.assertEqual(len(context["history"]), 4)
        self.assertEqual(len(materialize_scene(context)["nodes"]), 4)

    def test_invalid_proposals_are_atomic(self):
        context = base_context()
        before = deepcopy(context)
        with self.assertRaises(PictureContextError):
            apply_proposal(context, proposal(1, "remove_objects", {"object_ids": ["obj-0000"]}, "stale"))
        self.assertEqual(context, before)

        with self.assertRaises(PictureContextError):
            apply_proposal(context, proposal(0, "split_layers", {
                "source_layer_id": "left",
                "parts": [
                    {"layer_id": "only-one", "label": "One", "object_ids": ["obj-0000"]},
                    {"layer_id": "empty", "label": "Empty", "object_ids": []},
                ],
            }, "bad-split"))
        self.assertEqual(context, before)

        bad = {
            "object_id": "bad-red",
            "renderer_node": mark(58, "#ff0000"),
            "intent": "Invalid color",
            "tags": [],
        }
        with self.assertRaises(PictureContextError):
            apply_proposal(context, proposal(0, "add_objects", {
                "layer_id": "left", "before_object_id": "", "objects": [bad],
            }, "bad-color"))
        self.assertEqual(context, before)

    def test_recorded_handoffs_include_context_and_proposal_provenance(self):
        context = base_context("recorded-agent-test")
        with tempfile.TemporaryDirectory() as folder:
            initial_path = initialize_recorded_context(context, root=folder)
            loaded = read_context(initial_path)
            change = proposal(0, "remove_objects", {"object_ids": ["obj-0000"]}, "recorded")
            next_context, next_path = apply_and_record(loaded, change, root=folder, final=True)
            self.assertEqual(next_context["revision"], 1)
            self.assertTrue(next_path.is_file())
            run = Path(folder) / "recorded-agent-test"
            self.assertTrue((run / "handoffs/proposal-001.json").is_file())
            manifest = json.loads((run / "manifest.json").read_text())
            self.assertEqual(manifest["final_step"], 1)
            self.assertEqual(manifest["steps"][1]["provenance"]["context_revision"], 1)
            self.assertEqual(manifest["steps"][1]["provenance"]["skill"], "remove_objects")


if __name__ == "__main__":
    unittest.main()
