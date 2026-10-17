#!/usr/bin/env python3
"""Create a labeled MP4/GIF replay from a recorded Luna iteration chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont


def font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def label_frame(image: Image.Image, revision: int, total: int, title: str, detail: str) -> Image.Image:
    image = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    top = height - 74
    draw.rectangle((0, top, width, height), fill=(8, 29, 38, 205))
    draw.rectangle((0, top, round(width * revision / max(1, total)), top + 5), fill=(0, 255, 255, 230))
    draw.text((22, top + 14), title, font=font(20, True), fill=(255, 255, 255, 255))
    draw.text((22, top + 43), detail, font=font(13), fill=(195, 233, 232, 255))
    return Image.alpha_composite(image, overlay).convert("RGB")


def step_label(run_dir: Path, step: dict) -> tuple[str, str]:
    revision = step.get("provenance", {}).get("context_revision", step["index"])
    provenance = step.get("provenance", {})
    if not provenance.get("proposal"):
        return f"Revision {revision} · Initial handoff", "Base CMYK Tahoe composition"
    proposal = json.loads((run_dir / provenance["proposal"]).read_text())
    skill = provenance.get("skill", "mutation")
    agent = provenance.get("agent_id", "Luna")
    critique = " ".join(proposal.get("critique", "").split())
    if len(critique) > 112:
        critique = critique[:109].rstrip() + "…"
    return f"Revision {revision} · {skill}", f"{agent} · {critique}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out-prefix", type=Path)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--hold", type=int, default=20)
    parser.add_argument("--fade", type=int, default=6)
    args = parser.parse_args()
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to encode the timelapse")

    manifest = json.loads(args.manifest.read_text())
    run_dir = args.manifest.parent
    prefix = args.out_prefix or (run_dir / "agent-by-agent-timelapse")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    labeled = []
    total = len(manifest["steps"]) - 1
    for step in manifest["steps"]:
        title, detail = step_label(run_dir, step)
        image = Image.open(run_dir / step["png"]).convert("RGB")
        labeled.append(label_frame(image, step["index"], total, title, detail))

    with tempfile.TemporaryDirectory(prefix="drawing-sandbox-luna-") as folder:
        frames = Path(folder)
        frame_number = 0

        def save(image: Image.Image) -> None:
            nonlocal frame_number
            image.save(frames / f"frame_{frame_number:04d}.png")
            frame_number += 1

        for index, image in enumerate(labeled):
            for _ in range(args.hold + (10 if index == len(labeled) - 1 else 0)):
                save(image)
            if index + 1 < len(labeled):
                for fade_index in range(1, args.fade + 1):
                    save(Image.blend(image, labeled[index + 1], fade_index / (args.fade + 1)))

        subprocess.run([
            "ffmpeg", "-loglevel", "error", "-y", "-framerate", str(args.fps),
            "-i", str(frames / "frame_%04d.png"), "-c:v", "libx264", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(prefix) + ".mp4",
        ], check=True)
        subprocess.run([
            "ffmpeg", "-loglevel", "error", "-y", "-framerate", str(args.fps),
            "-i", str(frames / "frame_%04d.png"), "-filter_complex",
            "[0:v]fps=10,scale=720:-1:flags=lanczos,split[s0][s1];"
            "[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3",
            "-loop", "0", str(prefix) + ".gif",
        ], check=True)
    print(json.dumps({"mp4": str(prefix) + ".mp4", "gif": str(prefix) + ".gif",
                      "steps": len(labeled), "frames": frame_number}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
