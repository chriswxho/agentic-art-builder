"""Structural and image behavior tests for the bounded drawing language."""

from copy import deepcopy
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from drawing_sandbox.renderer import (
    Evaluation,
    MAX_DEPTH,
    MAX_EXPANDED_SHAPES,
    MAX_NODES,
    SceneValidationError,
    check_scene,
    expand_scene,
    render,
    to_svg,
    validate,
)
from drawing_sandbox.policy import composition_constraints


def ellipse(**changes):
    node = {"type": "ellipse", "cx": 64, "cy": 64, "rx": 20, "ry": 12,
            "fill": "#f00", "stroke": "none", "width": 0}
    node.update(changes)
    return node


def scene(nodes=None, motifs=None):
    return {"version": 1, "canvas": {"width": 128, "height": 128, "background": "#fff"},
            "nodes": [] if nodes is None else nodes, "motifs": {} if motifs is None else motifs}


class RendererTests(unittest.TestCase):
    def test_evaluation_defaults_are_independent(self):
        first, second = Evaluation(True), Evaluation(True)
        first.hard_checks["extra"] = True
        first.metrics["count"] = 1
        first.violations.append("example")
        self.assertEqual(second.hard_checks, {})
        self.assertEqual(second.metrics, {})
        self.assertEqual(second.violations, [])

    def test_valid_scene_normalizes_without_mutation(self):
        source = scene([ellipse(), {"type": "path", "points": [[1, 2], [20, 30]]}])
        source["description"] = "Optional scene metadata"
        before = deepcopy(source)
        result = validate(source)
        self.assertTrue(result.valid, result.violations)
        self.assertTrue(all(result.hard_checks.values()))
        self.assertEqual(result.metrics["expanded_shapes"], 2)
        expanded = expand_scene(source)
        self.assertEqual(expanded[0]["fill"], "#ff0000")
        self.assertEqual(expanded[1]["opacity"], 1)
        self.assertEqual(expanded[1]["matrix"], [1, 0, 0, 1, 0, 0])
        self.assertEqual(before, source)

    def test_group_and_reusable_motif_expand_identically(self):
        body = [{"type": "path", "points": [[0, 0], [8, -5], [20, 3]], "smooth": True}]
        transform = {"x": 40, "y": 60, "scale": 1.8, "rotate": 25}
        inline = scene([{"type": "group", "transform": transform, "children": body}])
        extracted = scene([{"type": "use", "motif": "branch", "transform": transform}],
                          {"branch": {"params": {}, "body": body, "description": "Reusable branch"}})
        self.assertEqual(expand_scene(inline), expand_scene(extracted))
        np.testing.assert_array_equal(np.asarray(render(inline)), np.asarray(render(extracted)))

    def test_nested_parameters_and_transform_order(self):
        leaf = ellipse(cx=0, cy=0, rx="$radius", ry=4)
        source = scene(
            [{"type": "use", "motif": "pair", "args": {"r": 7},
              "transform": {"x": 30, "y": 40, "scale": 2, "rotate": 90}}],
            {"dot": {"params": {"radius": 3}, "body": [leaf]},
             "pair": {"params": {"r": 5}, "body": [
                 {"type": "use", "motif": "dot", "args": {"radius": "$r"}, "transform": {"x": 10}}
             ]}},
        )
        shape = expand_scene(source)[0]
        self.assertEqual(shape["rx"], 7)
        self.assertAlmostEqual(shape["matrix"][4], 30)
        self.assertAlmostEqual(shape["matrix"][5], 60)
        self.assertAlmostEqual(shape["matrix"][1], 2)

    def test_parameter_scope_and_no_expressions(self):
        template = scene([{"type": "use", "motif": "dot"}],
                         {"dot": {"params": {"r": 5}, "body": [ellipse(rx="$r")]}})
        for value in ("$missing", "$r + 2", "__import__('os').getcwd()", "3.5"):
            with self.subTest(value=value):
                source = deepcopy(template)
                source["motifs"]["dot"]["body"][0]["rx"] = value
                self.assertFalse(validate(source).valid)
        bad_argument = deepcopy(template)
        bad_argument["nodes"][0]["args"] = {"unknown": 10}
        self.assertFalse(validate(bad_argument).valid)
        negative_argument = deepcopy(template)
        negative_argument["nodes"][0]["args"] = {"r": -5}
        self.assertFalse(validate(negative_argument).valid)

    def test_unknown_and_cyclic_motifs_including_unused_definitions(self):
        self.assertFalse(validate(scene([{"type": "use", "motif": "missing"}])).valid)
        cyclic = scene([], {"a": {"body": [{"type": "use", "motif": "b"}]},
                            "b": {"body": [{"type": "use", "motif": "a"}]}})
        result = validate(cyclic)
        self.assertFalse(result.valid)
        self.assertIn("cyclic", result.violations[0])
        unused_bad = scene([], {"unused": {"body": [{"type": "use", "motif": "missing"}]}})
        self.assertFalse(validate(unused_bad).valid)

    def test_nonfinite_and_boolean_numbers(self):
        for value in (float("nan"), float("inf"), -float("inf"), True, 10**1000):
            with self.subTest(value=repr(value)[:40]):
                source = scene([ellipse(cx=value)])
                result = validate(source)
                self.assertFalse(result.valid)
                self.assertTrue(result.violations)
                with self.assertRaises(SceneValidationError):
                    expand_scene(source)

    def test_canvas_and_color_restrictions(self):
        for width in (63, 1025, 128.0, True, "128"):
            source = scene()
            source["canvas"]["width"] = width
            self.assertFalse(validate(source).valid)
        for color in ("red", "url(https://example.com/x.svg)", "#abcd", "#gggggg", None):
            self.assertFalse(validate(scene([ellipse(fill=color)])).valid)
        for source in (None, [], {}, {"version": True}, scene([{}])):
            self.assertFalse(validate(source).valid)

    def test_shape_count_limits(self):
        self.assertEqual(MAX_EXPANDED_SHAPES, 2048)
        source = scene([ellipse() for _ in range(MAX_EXPANDED_SHAPES)])
        self.assertTrue(validate(source).valid)
        source["nodes"].append(ellipse())
        self.assertFalse(validate(source).valid)
        expanded = scene([{"type": "use", "motif": "pair"}
                          for _ in range(MAX_EXPANDED_SHAPES // 2 + 1)],
                         {"pair": {"body": [ellipse(), ellipse()]}})
        self.assertIn("shape count", validate(expanded).violations[0])

    def test_raw_and_expanded_node_limits(self):
        empty_group = {"type": "group", "children": []}
        self.assertFalse(validate(scene([deepcopy(empty_group) for _ in range(MAX_NODES + 1)])).valid)
        source = scene([{"type": "use", "motif": "empty_pair"}
                        for _ in range(MAX_NODES // 3 + 1)],
                       {"empty_pair": {"body": [deepcopy(empty_group), deepcopy(empty_group)]}})
        result = validate(source)
        self.assertFalse(result.valid)
        self.assertIn("expanded node", result.violations[0])

    def test_depth_and_transformed_bounds(self):
        node = ellipse()
        for _ in range(MAX_DEPTH - 1):
            node = {"type": "group", "children": [node]}
        self.assertTrue(validate(scene([node])).valid)
        self.assertFalse(validate(scene([{"type": "group", "children": [node]}])).valid)
        self.assertFalse(validate(scene([{"type": "group", "transform": {"scale": 64},
                                         "children": [ellipse(width=100)]}])).valid)
        self.assertFalse(validate(scene([ellipse(cx=65_535)])).valid)
        self.assertTrue(validate(scene([ellipse(cx=-100)])).valid)

    def test_closed_smooth_svg_and_filled_raster(self):
        source = scene([{"type": "path", "points": [[20, 64], [64, 25], [110, 64], [64, 90]],
                         "closed": True, "smooth": True, "fill": "#008844", "stroke": "#003322", "width": 2}])
        svg = to_svg(source)
        parsed = ET.fromstring(svg)
        path = parsed.find("{http://www.w3.org/2000/svg}path")
        self.assertIsNotNone(path)
        self.assertIn("C ", path.attrib["d"])
        self.assertTrue(path.attrib["d"].endswith("Z"))
        image = render(source)
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (128, 128))
        self.assertEqual(image.getpixel((64, 60)), (0, 136, 68))
        self.assertEqual(image.getpixel((2, 2)), (255, 255, 255))

    def test_opacity_is_composited_once(self):
        source = scene([ellipse(rx=40, ry=40, opacity=0.5)])
        source["canvas"]["background"] = "#00f"
        pixel = render(source, size=64).getpixel((32, 32))
        self.assertLessEqual(abs(pixel[0] - 128), 1)
        self.assertEqual(pixel[1], 0)
        self.assertLessEqual(abs(pixel[2] - 127), 1)

    def test_render_size_and_canvas_clipping(self):
        source = scene([{"type": "polygon", "points": [[-100, -100], [300, -100], [300, 300], [-100, 300]],
                         "fill": "#f00", "stroke": "none"}])
        source["canvas"]["height"] = 64
        image = render(source, size=64)
        self.assertEqual(image.size, (64, 64))
        self.assertEqual(image.getpixel((32, 5)), (255, 255, 255))
        self.assertEqual(image.getpixel((32, 32)), (255, 0, 0))
        self.assertEqual(render(source, size=(96, 48)).size, (96, 48))
        for size in (True, 0, 2048, (64,), (64, float("nan"))):
            with self.assertRaises(ValueError):
                render(source, size=size)

    def test_node_attributes_remain_strict(self):
        self.assertFalse(validate(scene([ellipse(unused_code="eval('...')")])).valid)
        self.assertFalse(validate(scene([{"type": "group", "transform": {"skew": 10}, "children": []}])).valid)
        self.assertFalse(validate(scene([{"type": "path", "points": [[1, 1], [2, 2]], "smooth": 1}])).valid)

    def test_optional_shape_limit_counts_expansion(self):
        source = scene([{"type": "use", "motif": "pair"}],
                       {"pair": {"body": [ellipse(), ellipse(cx=30)]}})
        self.assertEqual(check_scene(source), validate(source))
        self.assertTrue(check_scene(source, {"max_shapes": 2}).valid)
        result = check_scene(source, {"max_shapes": 1})
        self.assertFalse(result.valid)
        self.assertFalse(result.hard_checks["max_shapes"])
        self.assertEqual(result.metrics["expanded_shapes"], 2)
        self.assertTrue(check_scene(scene(), {"max_shapes": 0}).valid)
        for limit in (-1, True, 1.5, MAX_EXPANDED_SHAPES + 1):
            with self.subTest(limit=limit):
                self.assertFalse(check_scene(source, {"max_shapes": limit}).valid)

    def test_palette_normalizes_and_checks_active_paint_and_background(self):
        source = scene([ellipse(stroke="#000", width=0),
                        ellipse(fill="#00f", opacity=0)])
        result = check_scene(source, {"palette": ["#FFF", "#ff0000"]})
        self.assertTrue(result.valid, result.violations)
        self.assertEqual(result.metrics["palette_color_count"], 2)
        self.assertFalse(check_scene(source, {"palette": ["#fff"]}).valid)
        self.assertFalse(check_scene(source, {"palette": ["#f00"]}).valid)
        source["nodes"][0]["width"] = 2
        self.assertFalse(check_scene(source, {"palette": ["#fff", "#f00"]}).valid)
        self.assertTrue(check_scene(source, {"palette": ["#fff", "#f00", "#000"]}).valid)

    def test_cmyk_small_mark_policy(self):
        source = scene([ellipse(fill="#00ffff", opacity=.5)])
        constraints = composition_constraints()
        self.assertTrue(check_scene(source, constraints).valid)

        bad_color = deepcopy(source)
        bad_color["nodes"][0]["fill"] = "#ff0000"
        self.assertFalse(check_scene(bad_color, constraints).hard_checks["shape_palette"])

        bad_opacity = deepcopy(source)
        bad_opacity["nodes"][0]["opacity"] = .75
        self.assertFalse(check_scene(bad_opacity, constraints).hard_checks["required_opacity"])

        bad_background = deepcopy(source)
        bad_background["canvas"]["background"] = "#000000"
        self.assertFalse(check_scene(bad_background, constraints).hard_checks["background"])

        wide_stroke = scene([{"type": "path", "points": [[30, 30], [70, 30]],
                              "fill": "none", "stroke": "#000000", "width": 13,
                              "opacity": .5}])
        self.assertFalse(check_scene(wide_stroke, constraints).hard_checks["max_stroke_width"])

        large_mark = scene([ellipse(fill="#ffff00", opacity=.5, rx=49, ry=10)])
        self.assertFalse(check_scene(large_mark, constraints).hard_checks["max_shape_span"])

    def test_invalid_constraints_return_results_without_relaxing_hard_checks(self):
        for constraints in ([], {"unrecognized": True}, {"palette": "#fff"},
                            {"palette": ["red"]}, {"palette": ["none"]},
                            {"shape_palette": "#fff"}, {"background": "white"},
                            {"required_opacity": True}, {"max_stroke_width": -1},
                            {"max_shape_span": 0}):
            with self.subTest(constraints=constraints):
                result = check_scene(scene(), constraints)
                self.assertFalse(result.valid)
                self.assertTrue(result.violations)
        self.assertFalse(check_scene(scene([ellipse(cx=float("nan"))]), {}).valid)


if __name__ == "__main__":
    unittest.main()
