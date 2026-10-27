"""Assemble the README scene-explorer GIF from frames captured on the website (called by render.mjs).

    python report/figures/gif.py <frames dir> <output.gif>

The frames dir holds NNN.png images and durations.json (milliseconds per frame). All frames share one palette so
the outline colours stay stable while the before/after divider moves.
"""

import json
import sys
from pathlib import Path

from PIL import Image

WIDTH = 640  # README display width is 520 px; a little extra keeps outlines sharp on high-density screens


def main(frames_dir: Path, output: Path) -> None:
    durations = json.loads((frames_dir / "durations.json").read_text())
    frames = [Image.open(frames_dir / f"{i:03d}.png").convert("RGB") for i in range(len(durations))]
    height = round(frames[0].height * WIDTH / frames[0].width)
    frames = [frame.resize((WIDTH, height), Image.LANCZOS) for frame in frames]
    # One palette from a mosaic of every frame, so colours do not flicker between frames.
    mosaic = Image.new("RGB", (WIDTH, height * len(frames)))
    for i, frame in enumerate(frames):
        mosaic.paste(frame, (0, i * height))
    palette = mosaic.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    quantized = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    quantized[0].save(output, save_all=True, append_images=quantized[1:], duration=durations, loop=0, optimize=True)
    print(f"scene-explorer.gif: {len(frames)} frames, {WIDTH}x{height}, {output.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
