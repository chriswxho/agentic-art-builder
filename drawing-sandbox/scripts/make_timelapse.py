#!/usr/bin/env python3
"""Replay a Drawing Sandbox scene in top-level node order as PNG frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from drawing_sandbox.renderer import render


STAGES = (
    (0, "Canvas"),
    (1, "Snowy ridge"),
    (9, "Lake layers"),
    (17, "Right shoreline"),
    (20, "Distant granite"),
    (29, "Central boulders"),
    (37, "Foreground rocks"),
    (44, "Shoreline pines"),
    (46, "Foreground anchor"),
)


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/SFNS.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def stage_for(layer: int) -> str:
    result = STAGES[0][1]
    for start, name in STAGES:
        if layer < start:
            break
        result = name
    return result


def label(image: Image.Image, layer: int, total: int, complete: bool = False) -> Image.Image:
    image = image.convert("RGB")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    panel_top = height - 67
    draw.rectangle((0, panel_top, width, height), fill=(10, 37, 48, 192))
    progress = width if complete else round(width * layer / total)
    draw.rectangle((0, panel_top, progress, panel_top + 4), fill=(139, 224, 211, 255))
    heading = "Tahoe Day Study — complete" if complete else stage_for(layer)
    detail = "Drawing Sandbox construction replay" if complete else f"Layer {layer:02d} / {total}"
    draw.text((24, panel_top + 13), heading, font=font(21, bold=True), fill=(245, 249, 246, 255))
    draw.text((24, panel_top + 39), detail, font=font(14), fill=(184, 220, 216, 255))
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("frames", type=Path)
    parser.add_argument("--transition-frames", type=int, default=2)
    parser.add_argument("--intro-frames", type=int, default=18)
    parser.add_argument("--final-frames", type=int, default=30)
    args = parser.parse_args()

    source = json.loads(args.scene.read_text())
    nodes = source["nodes"]
    total = len(nodes)
    args.frames.mkdir(parents=True, exist_ok=True)

    frame_number = 0

    def save(image: Image.Image, layer: int, complete: bool = False) -> None:
        nonlocal frame_number
        label(image, layer, total, complete).save(
            args.frames / f"frame_{frame_number:04d}.png", optimize=True
        )
        frame_number += 1

    previous_scene = {**source, "nodes": []}
    previous = render(previous_scene)
    for _ in range(args.intro_frames):
        save(previous, 0)

    previous_stage = stage_for(0)
    for layer in range(1, total + 1):
        candidate = {**source, "nodes": nodes[:layer]}
        current = render(candidate)
        for step in range(1, args.transition_frames + 1):
            save(Image.blend(previous, current, step / args.transition_frames), layer)
        current_stage = stage_for(layer)
        if current_stage != previous_stage:
            for _ in range(3):
                save(current, layer)
        previous = current
        previous_stage = current_stage

    for _ in range(args.final_frames):
        save(previous, total, complete=True)

    print(f"{frame_number} frames written to {args.frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
