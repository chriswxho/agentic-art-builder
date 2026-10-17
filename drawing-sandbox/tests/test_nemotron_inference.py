import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from drawing_sandbox.iteration_context import (
    EVALUATOR_MODEL,
    GENERATOR_MODEL,
    INFERENCE_MODEL,
    PROTOCOL_VERSION,
    create_context,
)
from drawing_sandbox.nemotron_inference import (
    infer_plan,
    infer_proposal,
    inference_provenance,
    load_crusoe_api_key,
    parse_proposal,
    write_inference_log,
)
from drawing_sandbox.planning import PLAN_PROTOCOL_VERSION


def context():
    nodes = [
        {
            "type": "ellipse", "cx": x, "cy": 32, "rx": 5, "ry": 5,
            "fill": "#00ffff", "stroke": "none", "width": 0, "opacity": 0.5,
        }
        for x in (12, 28)
    ]
    return create_context(
        {
            "version": 1,
            "canvas": {"width": 64, "height": 64, "background": "#ffffff"},
            "motifs": {},
            "nodes": nodes,
        },
        run_id="nemotron-test",
        prompt_set=["Refine the cyan study."],
        composition={"layers": ["marks"]},
        layer_specs=[{"layer_id": "marks", "label": "Marks", "end_index": 2}],
    )


class NemotronInferenceTests(unittest.TestCase):
    def test_loads_historical_bare_token_without_logging_it(self):
        with tempfile.TemporaryDirectory() as folder:
            env = Path(folder) / ".env"
            env.write_text("secret-token-value\n")
            self.assertEqual(load_crusoe_api_key(env), "secret-token-value")

    def test_loads_crusoe_api_token_assignment(self):
        with tempfile.TemporaryDirectory() as folder:
            env = Path(folder) / ".env"
            env.write_text('CRUSOE_API_TOKEN="preferred-token-value"\n')
            self.assertEqual(load_crusoe_api_key(env), "preferred-token-value")

    def test_parses_json_from_reasoning_wrapper(self):
        proposal = {"protocol_version": PROTOCOL_VERSION, "base_revision": 0}
        text = "analysis first\n```json\n" + json.dumps(proposal) + "\n```"
        self.assertEqual(parse_proposal(text), proposal)

    def test_forward_inference_is_validated_and_logged_without_credentials(self):
        proposal = {
            "protocol_version": PROTOCOL_VERSION,
            "agent_id": "nemotron-remove-objects-r001",
            "mutation_id": "nemotron-test-remove_objects-r001",
            "base_revision": 0,
            "critique": "The first dot is visually redundant.",
            "rationale": "Removing it gives the pair clearer rhythm.",
            "remove_objects": {"object_ids": ["obj-0000"]},
        }
        response = {
            "id": "completion-test",
            "created": 1,
            "model": INFERENCE_MODEL,
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": json.dumps(proposal)},
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140},
        }
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / "current.png"
            Image.new("RGB", (64, 64), "white").save(image)
            with patch("drawing_sandbox.nemotron_inference._call_crusoe", return_value=response) as call:
                accepted, log = infer_proposal(
                    context(), "remove_objects", api_key="must-not-be-recorded",
                    current_image=image, max_attempts=1,
                )
            self.assertEqual(accepted, proposal)
            body = call.call_args.args[1]
            self.assertEqual(body["model"], INFERENCE_MODEL)
            self.assertEqual(body["model"], GENERATOR_MODEL)
            self.assertEqual(body["messages"][1]["content"][1]["type"], "image_url")
            serialized = json.dumps(log)
            self.assertNotIn("must-not-be-recorded", serialized)
            path = write_inference_log(log, root=folder)
            self.assertTrue(path.is_file())
            self.assertEqual(inference_provenance(log)["response_id"], "completion-test")

    def test_planner_forward_inference_returns_regional_assignments(self):
        source = context()
        plan = {
            "protocol_version": PLAN_PROTOCOL_VERSION,
            "planner_id": "nemotron-planner-r000",
            "plan_id": "nemotron-test-plan-r000",
            "base_revision": 0,
            "base_state_sha256": source["state_sha256"],
            "critique": "Both halves need more deliberate local texture.",
            "strategy": "Assign one bounded detail pass to each half.",
            "assignments": [
                {
                    "assignment_id": "left-detail", "part_label": "Left detail",
                    "brief": "Add cyan ellipses in the left half.",
                    "region": {"x": 0, "y": 0, "width": 32, "height": 64},
                    "layer_id": "marks", "shape_types": ["ellipse"],
                    "colors": ["#00ffff"], "minimum_shapes": 1, "shape_budget": 2,
                },
                {
                    "assignment_id": "right-detail", "part_label": "Right detail",
                    "brief": "Add black ellipses in the right half.",
                    "region": {"x": 32, "y": 0, "width": 32, "height": 64},
                    "layer_id": "marks", "shape_types": ["ellipse"],
                    "colors": ["#000000"], "minimum_shapes": 1, "shape_budget": 2,
                },
            ],
        }
        response = {
            "id": "planner-completion-test", "created": 1, "model": EVALUATOR_MODEL,
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(plan)}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 80, "total_tokens": 200},
        }
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / "current.png"
            Image.new("RGB", (64, 64), "white").save(image)
            with patch("drawing_sandbox.nemotron_inference._call_crusoe", return_value=response) as call:
                accepted, log = infer_plan(
                    source, api_key="must-not-be-recorded", current_image=image, max_attempts=1,
                    minimum_total_shapes=2,
                )
            self.assertEqual(len(accepted["assignments"]), 2)
            self.assertEqual(call.call_args.args[1]["model"], EVALUATOR_MODEL)
            self.assertEqual(log["plan_id"], "nemotron-test-plan-r000")
            self.assertNotIn("must-not-be-recorded", json.dumps(log))


if __name__ == "__main__":
    unittest.main()
