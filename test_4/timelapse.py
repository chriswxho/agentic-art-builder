#!/usr/bin/env python3
"""Animated GIF of a test_4 run: reference beside the canvas after every commit.

Writes runs/<run_id>/timelapse.gif. Each frame is captioned with the commit number, the patch that
was painted (and by which agent), and the Lab score after it.

usage: drawing-sandbox/.venv/bin/python test_4/timelapse.py test_4/runs/easy-sea-sunset [more run dirs]
"""

import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont

W, H = 480, 320
FRAME_MS, FIRST_MS, LAST_MS = 700, 1200, 3000


def font(size: int):
    for candidate in ("/System/Library/Fonts/Supplemental/Arial.ttf",
                      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def build(run_dir: Path) -> Path:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    tree = json.loads((run_dir / "tree.json").read_text()) if (run_dir / "tree.json").exists() else {"nodes": []}
    nodes = {n["id"]: n for n in tree["nodes"]}
    scores = {s["commit"]: s for s in json.loads((run_dir / "scores.json").read_text())}
    reference = Image.open(manifest["composition"]["source_image"]).convert("RGB").resize((W, H))
    big, small = font(18), font(13)
    total = len(manifest["steps"]) - 1
    frames = []
    for index, step in enumerate(manifest["steps"]):
        provenance = step.get("provenance", {})
        node = nodes.get(provenance.get("tree_node"), {})
        score = scores.get(index, {})
        frame = Image.new("RGB", (2 * W + 30, H + 92), (24, 26, 30))
        frame.paste(reference, (10, 40))
        frame.paste(Image.open(run_dir / step["png"]).convert("RGB").resize((W, H)), (W + 20, 40))
        draw = ImageDraw.Draw(frame)
        draw.text((10, 12), f"{tree.get('run_id', run_dir.name)} · reference", font=big, fill=(230, 230, 230))
        title = "blank canvas" if index == 0 else f"commit {index} of {total}"
        draw.text((W + 20, 12), title, font=big, fill=(230, 230, 230))
        if index:
            what = f"{node.get('kind', '?')} (depth {node.get('depth', '?')}): {str(node.get('label', ''))[:60]}"
            delta = provenance.get("patch_delta_e")
            detail = f"patch color difference {delta[0]} → {delta[1]}" if delta else ""
        else:
            what, detail = "", ""
        draw.text((10, H + 48), what, font=small, fill=(200, 220, 230))
        draw.text((10, H + 68), detail, font=small, fill=(160, 180, 190))
        lab = score.get("lab")
        draw.text((W + 20, H + 48), f"Lab score {lab:.3f}" if lab is not None else "", font=big, fill=(0, 230, 230))
        bar = round((2 * W + 10) * (lab or 0))
        draw.rectangle((10, H + 88, 10 + bar, H + 91), fill=(0, 230, 230))
        frames.append(frame)
    durations = [FIRST_MS] + [FRAME_MS] * (len(frames) - 2) + [LAST_MS] if len(frames) > 1 else [LAST_MS]
    out = run_dir / "timelapse.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    return out


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(build(Path(arg)))
