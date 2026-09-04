#!/usr/bin/env python3
"""Assemble a map-native SN7 perception qualitative figure.

The figure deliberately shows editable-map writebacks and residuals rather
than enlarged binary masks.  Source panels are validation-only assets rendered
from the fixed updater protocol; this script changes layout only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BACKGROUND = (255, 255, 255)
INK = (28, 35, 42)
MUTED = (92, 101, 110)
RULE = (211, 216, 220)
TP = (0, 158, 115)
FP = (213, 94, 0)
FN = (86, 180, 233)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def panel(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), BACKGROUND)
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def assert_validation_cases(root: Path, cases: list[str]) -> None:
    for case in cases:
        manifest = json.loads((root / case / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("test_assets_read") is not False:
            raise ValueError(f"{case} is not validation-only")
        if set(manifest.get("methods", ())) != {"changemamba", "ban"}:
            raise ValueError(f"unexpected perception methods for {case}")


def assemble(root: Path, output: Path) -> dict[str, object]:
    cases = ["008_add_success", "015_delete_success", "020_reshape_success"]
    assert_validation_cases(root, cases)
    headers = [
        "Current context",
        "Reference map",
        "ChangeMamba writeback",
        "ChangeMamba residual",
        "Open-CD BAN residual",
    ]
    operations = ["ADD", "DELETE", "RESHAPE"]
    panel_size, gap, left = 235, 20, 150
    header_h, row_h, bottom = 90, 278, 82
    width = left + len(headers) * panel_size + (len(headers) - 1) * gap + 44
    height = header_h + len(cases) * row_h + bottom
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    title_font = font(27, bold=True)
    head_font = font(16, bold=True)
    row_font = font(19, bold=True)
    note_font = font(14)

    draw.text((left, 20), "SN7 editable-map writeback diagnostics", fill=INK, font=title_font)
    for index, header in enumerate(headers):
        x = left + index * (panel_size + gap)
        draw.text((x, 60), header, fill=INK, font=head_font)

    source_names = [
        "01_context.png",
        "03_reference_vector.png",
        "04_changemamba_writeback.png",
        "05_changemamba_residual.png",
        "05_ban_residual.png",
    ]
    for row, (case, operation) in enumerate(zip(cases, operations, strict=True)):
        top = header_h + row * row_h
        draw.text((26, top + panel_size // 2 - 12), operation, fill=INK, font=row_font)
        for column, name in enumerate(source_names):
            x = left + column * (panel_size + gap)
            canvas.paste(panel(root / case / name, panel_size), (x, top))
            draw.rectangle((x, top, x + panel_size - 1, top + panel_size - 1), outline=RULE, width=1)

    legend_y = height - 50
    draw.text((left, legend_y), "Residual:", fill=MUTED, font=note_font)
    cursor = left + 76
    for color, text in ((TP, "correct update"), (FP, "false write"), (FN, "missed update")):
        draw.rectangle((cursor, legend_y + 4, cursor + 15, legend_y + 19), fill=color)
        draw.text((cursor + 21, legend_y), text, fill=MUTED, font=note_font)
        cursor += 132
    draw.text((cursor + 10, legend_y), "Validation-only; boundary rendering, not attribution.", fill=MUTED, font=note_font)

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return {
        "schema_version": "sn7-perception-writeback-figure-v1",
        "split": "val",
        "test_assets_read": False,
        "cases": cases,
        "methods": ["ChangeMamba", "Open-CD BAN"],
        "output": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(assemble(args.root, args.output), indent=2))


if __name__ == "__main__":
    main()
