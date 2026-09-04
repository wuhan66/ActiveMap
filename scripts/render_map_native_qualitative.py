#!/usr/bin/env python3
"""Render map-update-native qualitative panels from aligned validation assets.

The input is the validation-only per-case layout written by
``render_sn7_updater_qualitative.py``.  Unlike a segmentation mosaic, this
renderer exports independent cartographic panels: editable prior boundaries,
reference boundaries, per-method writeback boundaries, and false-positive /
false-negative residuals.  A paper layout can then place only the panels that
answer one map-update question, without baking labels into the imagery.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


RGB = tuple[int, int, int]
PRIOR: RGB = (244, 162, 0)
TARGET: RGB = (0, 158, 115)
PREDICTION: RGB = (0, 114, 178)
TRUE_POSITIVE: RGB = (0, 158, 115)
FALSE_POSITIVE: RGB = (213, 94, 0)
FALSE_NEGATIVE: RGB = (86, 180, 233)


def _load_mask(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGBA")
    pixels = np.asarray(image)
    return (pixels[..., 3] > 0) & (pixels[..., :3].max(axis=-1) > 127)


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    result = mask.copy()
    for _ in range(radius):
        padded = np.pad(result, 1, mode="edge")
        result = np.logical_or.reduce(
            [
                padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
                for dy in range(3)
                for dx in range(3)
            ]
        )
    return result


def _outline(mask: np.ndarray, width: int = 1) -> np.ndarray:
    padded = np.pad(mask, 1, mode="edge")
    eroded = np.logical_and.reduce(
        [
            padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
            for dy in range(3)
            for dx in range(3)
        ]
    )
    return _dilate(mask & ~eroded, width)


def _base(rgb: np.ndarray, dim: float) -> np.ndarray:
    return np.clip(rgb.astype(np.float32) * dim, 0, 255).astype(np.uint8)


def _paint(canvas: np.ndarray, mask: np.ndarray, color: RGB) -> None:
    canvas[mask] = np.asarray(color, dtype=np.uint8)


def _save(array: np.ndarray, destination: Path, size: int) -> None:
    image = Image.fromarray(array, mode="RGB")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    image.save(destination)


def _extent(mask: np.ndarray, margin: int) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    height, width = mask.shape
    if not len(xs):
        return 0, 0, width, height
    x0 = max(0, int(xs.min()) - margin)
    y0 = max(0, int(ys.min()) - margin)
    x1 = min(width, int(xs.max()) + margin + 1)
    y1 = min(height, int(ys.max()) + margin + 1)
    return x0, y0, x1, y1


def _crop(array: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bounds
    return array[y0:y1, x0:x1]


def _components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """Return connected-component boxes as (area, x0, y0, x1, y1)."""
    height, width = mask.shape
    remaining = mask.copy()
    components: list[tuple[int, int, int, int, int]] = []
    for start_y, start_x in zip(*np.where(remaining), strict=True):
        if not remaining[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        remaining[start_y, start_x] = False
        area = 0
        min_x = max_x = int(start_x)
        min_y = max_y = int(start_y)
        while stack:
            y, x = stack.pop()
            area += 1
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and remaining[ny, nx]:
                    remaining[ny, nx] = False
                    stack.append((ny, nx))
        components.append((area, min_x, min_y, max_x + 1, max_y + 1))
    return sorted(components, reverse=True)


def _zoom_bounds(mask: np.ndarray, *, count: int = 2, margin: int = 7) -> list[tuple[int, int, int, int]]:
    boxes = _components(mask)
    selected: list[tuple[int, int, int, int]] = []
    for _, x0, y0, x1, y1 in boxes:
        candidate = _extent(
            np.pad(
                np.ones((y1 - y0, x1 - x0), dtype=bool),
                ((y0, mask.shape[0] - y1), (x0, mask.shape[1] - x1)),
            ),
            margin,
        )
        if not selected or all(
            abs((candidate[0] + candidate[2]) - (prior[0] + prior[2]))
            + abs((candidate[1] + candidate[3]) - (prior[1] + prior[3]))
            > 6
            for prior in selected
        ):
            selected.append(candidate)
        if len(selected) == count:
            return selected
    fallback = _extent(mask, margin)
    while len(selected) < count:
        selected.append(fallback)
    return selected


def _render_case(case_dir: Path, output_dir: Path, *, size: int, zoom_size: int) -> dict:
    rgb = _load_rgb(case_dir / "01_current_rgb.png")
    prior = _load_mask(case_dir / "02_prior_mask.png")
    target = _load_mask(case_dir / "03_target_mask.png")
    prior_outline = _outline(prior)
    target_outline = _outline(target)

    output_dir.mkdir(parents=True, exist_ok=False)
    _save(rgb, output_dir / "01_context.png", size)

    prior_panel = _base(rgb, 0.72)
    _paint(prior_panel, prior_outline, PRIOR)
    _save(prior_panel, output_dir / "02_prior_vector.png", size)

    target_panel = _base(rgb, 0.72)
    _paint(target_panel, target_outline, TARGET)
    _save(target_panel, output_dir / "03_reference_vector.png", size)

    methods = []
    for committed_path in sorted(case_dir.glob("*_committed_mask.png")):
        method = committed_path.name.removesuffix("_committed_mask.png").split("_", 1)[1]
        committed = _load_mask(committed_path)
        committed_outline = _outline(committed)

        writeback = _base(rgb, 0.72)
        _paint(writeback, target_outline, TARGET)
        _paint(writeback, committed_outline, PREDICTION)
        _save(writeback, output_dir / f"04_{method}_writeback.png", size)

        true_positive = committed & target
        false_positive = committed & ~target
        false_negative = target & ~committed
        residual = _base(rgb, 0.45)
        _paint(residual, _outline(true_positive), TRUE_POSITIVE)
        _paint(residual, _outline(false_positive), FALSE_POSITIVE)
        _paint(residual, _outline(false_negative), FALSE_NEGATIVE)
        _save(residual, output_dir / f"05_{method}_residual.png", size)

        focus = np.logical_xor(prior, target) | false_positive | false_negative
        for zoom_index, bounds in enumerate(_zoom_bounds(focus), start=1):
            for name, panel in (
                ("prior", prior_panel),
                ("reference", target_panel),
                ("writeback", writeback),
                ("residual", residual),
            ):
                cropped = _crop(panel, bounds)
                image = Image.fromarray(cropped, mode="RGB").resize(
                    (zoom_size, zoom_size), Image.Resampling.LANCZOS
                )
                image.save(output_dir / f"zoom{zoom_index:02d}_{method}_{name}.png")
        methods.append(method)

    manifest = {
        "schema_version": "map-native-qualitative-v1",
        "source_case": str(case_dir),
        "source_protocol": "aligned validation-only SN7 updater assets",
        "test_assets_read": False,
        "rendering": {
            "prior_boundary_rgb": PRIOR,
            "reference_boundary_rgb": TARGET,
            "writeback_boundary_rgb": PREDICTION,
            "residual_true_positive_rgb": TRUE_POSITIVE,
            "residual_false_positive_rgb": FALSE_POSITIVE,
            "residual_false_negative_rgb": FALSE_NEGATIVE,
            "representation": "raster masks rendered as editable boundary geometry; no feature attribution is implied",
        },
        "methods": methods,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--cases", nargs="*", default=["008_add_success", "015_delete_success", "020_reshape_success"])
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--zoom-size", type=int, default=768)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    rendered = []
    for case in args.cases:
        source = args.input_dir / case
        if not source.is_dir():
            raise FileNotFoundError(source)
        rendered.append(_render_case(source, args.output_dir / case, size=args.size, zoom_size=args.zoom_size))
    selection = args.input_dir / "selection.jsonl"
    if selection.is_file():
        shutil.copy2(selection, args.output_dir / "source_selection.jsonl")
    (args.output_dir / "README.md").write_text(
        "# Map-native validation visuals\n\n"
        "Each case is exported as independent image panels for manuscript layout. "
        "Orange = prior map boundary; green = reference boundary; blue = method writeback; "
        "residual green/orange/light-blue = true-positive/false-positive/false-negative. "
        "These are validation-only aligned baseline assets, not an ActiveMap policy qualitative claim.\n",
        encoding="utf-8",
    )
    print(json.dumps({"cases": len(rendered), "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
