from copy import deepcopy
import unittest

from drawing_sandbox.iteration_context import PROTOCOL_VERSION, create_context
from drawing_sandbox.planning import (
    PLAN_PROTOCOL_VERSION,
    PicturePlanError,
    validate_delegated_proposal,
    validate_plan,
)


def ellipse(x, y, color="#00ffff"):
    return {
        "type": "ellipse", "cx": x, "cy": y, "rx": 3, "ry": 3,
        "fill": color, "stroke": "none", "width": 0, "opacity": 0.5,
    }


def context():
    return create_context(
        {
            "version": 1,
            "canvas": {"width": 64, "height": 64, "background": "#ffffff"},
            "motifs": {},
            "nodes": [ellipse(16, 16), ellipse(48, 48, "#000000")],
        },
        run_id="planner-test",
        prompt_set=["Refine two picture regions."],
        composition={"layers": ["marks"]},
        layer_specs=[{"layer_id": "marks", "label": "Marks", "end_index": 2}],
    )


def plan(source):
    return {
        "protocol_version": PLAN_PROTOCOL_VERSION,
        "planner_id": "nemotron-planner-r000",
        "plan_id": "planner-test-plan-r000",
        "base_revision": 0,
        "base_state_sha256": source["state_sha256"],
        "critique": "The two regions need denser local detail.",
        "strategy": "Delegate one bounded ellipse pass to each half.",
        "assignments": [
            {
                "assignment_id": "left-detail",
                "part_label": "Left detail",
                "brief": "Add cyan texture around the left mark.",
                "region": {"x": 0, "y": 0, "width": 32, "height": 64},
                "layer_id": "marks",
                "shape_types": ["ellipse"],
                "colors": ["#00ffff"],
                "minimum_shapes": 2,
                "shape_budget": 4,
            },
            {
                "assignment_id": "right-detail",
                "part_label": "Right detail",
                "brief": "Add black texture around the right mark.",
                "region": {"x": 32, "y": 0, "width": 32, "height": 64},
                "layer_id": "marks",
                "shape_types": ["ellipse"],
                "colors": ["#000000"],
                "minimum_shapes": 1,
                "shape_budget": 3,
            },
        ],
    }


def proposal(nodes):
    return {
        "protocol_version": PROTOCOL_VERSION,
        "agent_id": "nemotron-left-detail-r001",
        "mutation_id": "planner-test-left-r001",
        "base_revision": 0,
        "critique": "The left region is sparse.",
        "rationale": "Two small marks add local density.",
        "add_objects": {
            "layer_id": "marks",
            "before_object_id": "",
            "objects": [
                {
                    "object_id": f"plan-left-detail-r001-{index + 1:02d}",
                    "renderer_node": node,
                    "intent": "left texture",
                    "tags": ["type:ellipse", "fill:#00ffff"],
                }
                for index, node in enumerate(nodes)
            ],
        },
    }


class PicturePlanningTests(unittest.TestCase):
    def test_valid_plan_and_delegated_proposal(self):
        source = context()
        accepted = validate_plan(source, plan(source))
        change = proposal([ellipse(10, 20), ellipse(24, 36)])
        validate_delegated_proposal(source, change, accepted["assignments"][0])

    def test_plan_rejects_stale_context_and_excess_budget(self):
        source = context()
        stale = plan(source)
        stale["base_state_sha256"] = "0" * 64
        with self.assertRaises(PicturePlanError):
            validate_plan(source, stale)
        excessive = plan(source)
        excessive["assignments"][0]["shape_budget"] = 49
        with self.assertRaises(PicturePlanError):
            validate_plan(source, excessive)

    def test_delegation_rejects_wrong_region_color_and_count(self):
        source = context()
        assignment = validate_plan(source, plan(source))["assignments"][0]
        outside = proposal([ellipse(10, 20), ellipse(40, 36)])
        with self.assertRaises(PicturePlanError):
            validate_delegated_proposal(source, outside, assignment)
        wrong_color = proposal([ellipse(10, 20), ellipse(24, 36, "#ff00ff")])
        with self.assertRaises(PicturePlanError):
            validate_delegated_proposal(source, wrong_color, assignment)
        too_few = proposal([ellipse(10, 20)])
        with self.assertRaises(PicturePlanError):
            validate_delegated_proposal(source, too_few, assignment)


if __name__ == "__main__":
    unittest.main()
