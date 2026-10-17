"""The enforced instruction set for agent-authored picture compositions."""

from __future__ import annotations

from copy import deepcopy


CMYK_INKS = {
    "cyan": "#00ffff",
    "magenta": "#ff00ff",
    "yellow": "#ffff00",
    "black": "#000000",
}

CMYK_MARK_CONSTRAINTS = {
    "shape_palette": list(CMYK_INKS.values()),
    "background": "#ffffff",
    "required_opacity": 0.5,
    "max_stroke_width": 12,
    "max_shape_span": 96,
}


def composition_constraints() -> dict:
    """Return a fresh copy so callers cannot mutate the global policy."""
    return deepcopy(CMYK_MARK_CONSTRAINTS)


def current_instruction_set() -> dict:
    """Return the human- and machine-readable picture-making contract."""
    return {
        "version": 1,
        "canvas": {
            "background": "#ffffff",
            "size_range": [64, 1024],
        },
        "marks": {
            "primitive_types": ["path", "polygon", "ellipse"],
            "assembly_types": ["group", "use"],
            "inks": dict(CMYK_INKS),
            "opacity": 0.5,
            "max_transformed_stroke_width": 12,
            "max_transformed_span": 96,
        },
        "composition": {
            "paint_order": "nodes are composited in array order",
            "blend_model": "source-over",
            "max_expanded_shapes": 512,
        },
        "recording": {
            "required_metadata": ["run_id", "prompt_set", "composition"],
            "saved_progress": ["scene_json", "svg", "png", "checks"],
            "saved_final": ["scene_json", "svg", "png"],
        },
    }


__all__ = [
    "CMYK_INKS", "CMYK_MARK_CONSTRAINTS", "composition_constraints",
    "current_instruction_set",
]
