import unittest

from drawing_sandbox.evaluation import (
    EVALUATION_PROTOCOL_VERSION,
    PictureEvaluationError,
    validate_evaluation,
)


def verdict(improved=True):
    return {
        "protocol_version": EVALUATION_PROTOCOL_VERSION,
        "evaluator_id": "nemotron-super-vl-evaluator-r005",
        "evaluation_id": "run-evaluation-r005",
        "run_id": "run",
        "base_revision": 0,
        "final_revision": 5,
        "reference_match_before": 52,
        "reference_match_after": 67 if improved else 48,
        "composition_before": 58,
        "composition_after": 70 if improved else 55,
        "improved": improved,
        "summary": "The planned detail improves reference correspondence.",
        "strengths": ["More specific regional structure."],
        "issues": ["Some marks remain schematic."],
        "next_actions": ["Refine the foreground silhouette."],
    }


class PictureEvaluationTests(unittest.TestCase):
    def test_valid_typed_evaluation(self):
        result = validate_evaluation(
            verdict(), run_id="run", base_revision=0, final_revision=5,
            evaluator_id="nemotron-super-vl-evaluator-r005",
            evaluation_id="run-evaluation-r005",
        )
        self.assertTrue(result["improved"])
        self.assertEqual(result["reference_match_after"], 67.0)

    def test_improvement_flag_must_match_scores(self):
        result = verdict()
        result["improved"] = False
        with self.assertRaises(PictureEvaluationError):
            validate_evaluation(
                result, run_id="run", base_revision=0, final_revision=5,
                evaluator_id="nemotron-super-vl-evaluator-r005",
                evaluation_id="run-evaluation-r005",
            )


if __name__ == "__main__":
    unittest.main()
