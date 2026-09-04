#!/usr/bin/env python3
"""Assemble a clean preview from provenance-bearing map-native panels.

Raw panels and their manifests remain the source of truth. This script only
creates a layout preview for manuscript composition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BACKGROUND = (255, 255, 255)
INK = (27, 33, 38)
MUTED = (83, 91, 98)
RULE = (205, 211, 215)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        ("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _panel(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), BACKGROUND)
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def _paste_row(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    top: int,
    tag: str,
    title: str,
    panels: list[tuple[str, Path]],
    left: int,
    panel_size: int,
    gap: int,
) -> None:
    tag_font = _font(24, bold=True)
    title_font = _font(22, bold=True)
    label_font = _font(16, bold=True)
    draw.text((left, top), tag, fill=INK, font=tag_font)
    draw.text((left + 34, top + 2), title, fill=INK, font=title_font)
    image_top = top + 48
    for index, (label, path) in enumerate(panels):
        x = left + index * (panel_size + gap)
        draw.text((x, image_top - 22), label, fill=INK, font=label_font)
        canvas.paste(_panel(path, panel_size), (x, image_top))
        draw.rectangle((x, image_top, x + panel_size - 1, image_top + panel_size - 1), outline=RULE, width=1)


def _paste_comparison(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    top: int,
    left: int,
    panel_size: int,
    gap: int,
    case: Path,
) -> None:
    """Show the DELETE error explicitly; blank panels no longer read as missing output."""
    label_font = _font(16, bold=True)
    note_font = _font(14, bold=True)
    panels = [
        ("Direct: missed DELETE", case / "zoom01" / "direct_writeback.png", (213, 94, 0)),
        ("ActiveMap: correct DELETE", case / "zoom01" / "activemap_writeback.png", (0, 114, 178)),
    ]
    for index, (label, path, accent) in enumerate(panels):
        x = left + index * (panel_size + gap)
        draw.text((x, top - 24), label, fill=INK, font=label_font)
        canvas.paste(_panel(path, panel_size), (x, top))
        draw.rectangle((x, top, x + panel_size - 1, top + panel_size - 1), outline=RULE, width=1)
        draw.rectangle((x, top + panel_size + 8, x + 14, top + panel_size + 22), fill=accent)
        text = "stale road retained" if index == 0 else "stale road removed"
        draw.text((x + 20, top + panel_size + 5), text, fill=MUTED, font=note_font)


def assemble(muno_root: Path, space_root: Path, output: Path) -> dict:
    muno_case = muno_root / "01_improved_writeback_task-9773234aa3b1b826"
    space_case = space_root / "01_rank100_changer_validation_case"
    for manifest in (muno_case / "manifest.json", space_case / "manifest.json"):
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("split") != "val" or payload.get("test_assets_read") is not False:
            raise ValueError(f"not a validation-only asset: {manifest}")

    # The MUNO row uses a zoomed matched comparison.  A correct DELETE is
    # mostly empty by construction, so the direct-vs-ActiveMap labels make the
    # counterfactual error legible without adding a decorative heatmap.
    panel_size, gap, left = 190, 18, 36
    width = left * 2 + 4 * panel_size + 3 * gap
    height = 815
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    _paste_row(
        canvas,
        draw,
        top=24,
        tag="a",
        title="Road-map DELETE audit (MUNO21)",
        panels=[
            ("Aerial context", muno_case / "current_rgb.png"),
            ("Prior map", muno_case / "zoom01" / "prior_map.png"),
            ("Reference map", muno_case / "zoom01" / "reference_map.png"),
            ("Residual: direct policy", muno_case / "zoom01" / "residual_direct.png"),
        ],
        left=left,
        panel_size=panel_size,
        gap=gap,
    )
    _paste_comparison(
        canvas,
        draw,
        top=290,
        left=left + (width - 2 * panel_size - gap - 2 * left) // 2,
        panel_size=170,
        gap=34,
        case=muno_case,
    )
    draw.line((left, 500, width - left, 500), fill=RULE, width=1)
    _paste_row(
        canvas,
        draw,
        top=526,
        tag="b",
        title="Evidence selection with deferred writeback (SpaceNet8)",
        panels=[
            ("First POST", space_case / "zoom01" / "first_post_rgb.png"),
            ("Selected POST", space_case / "zoom01" / "selected_post_rgb.png"),
            ("Reference change", space_case / "zoom01" / "reference_change.png"),
            ("Safe Commit: defer", space_case / "zoom01" / "safe_commit_writeback.png"),
        ],
        left=left,
        panel_size=panel_size,
        gap=gap,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return {
        "schema_version": "activemap-cross-domain-qualitative-preview-v1",
        "split": "val",
        "test_assets_read": False,
        "output": str(output),
        "inputs": [str(muno_case), str(space_case)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("muno_root", type=Path)
    parser.add_argument("spacenet_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(assemble(args.muno_root, args.spacenet_root, args.output), indent=2))


if __name__ == "__main__":
    main()
