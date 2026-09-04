#!/usr/bin/env python3
"""Render ranked updater failures from frozen predictions without new inference."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from affine import Affine
from PIL import Image, ImageDraw
from render_vector_update_examples import (
    COLORS,
    _badge,
    _draw_geometry,
    _font,
    _geometry,
    _load_image,
)
from shapely.geometry import shape

from activemap.models import EditOperation
from activemap.updater_records import load_updater_samples


def _zoom_box(
    sample: Any,
    prediction: dict[str, Any],
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    transform = Affine(*(sample.crop_transform or [1, 0, 0, 0, 1, 0]))
    inverse = ~transform
    geometries = [_geometry(sample.prior_geometry), _geometry(sample.target_geometry)]
    geometry_payload = prediction.get("predicted_geometry")
    if geometry_payload is not None:
        geometries.append(shape(geometry_payload))
    pixels: list[tuple[float, float]] = []
    for geometry in geometries:
        if geometry is None or geometry.is_empty:
            continue
        min_x, min_y, max_x, max_y = geometry.bounds
        pixels.extend(
            inverse * coordinate
            for coordinate in (
                (min_x, min_y),
                (min_x, max_y),
                (max_x, min_y),
                (max_x, max_y),
            )
        )
    if not pixels:
        return (0, 0, width, height)
    xs, ys = zip(*pixels, strict=True)
    center_x = 0.5 * (min(xs) + max(xs))
    center_y = 0.5 * (min(ys) + max(ys))
    extent = max(max(xs) - min(xs), max(ys) - min(ys))
    side = min(max(32.0, extent * 3.0), float(min(width, height)))
    left = min(max(center_x - side / 2.0, 0.0), width - side)
    top = min(max(center_y - side / 2.0, 0.0), height - side)
    return (round(left), round(top), round(left + side), round(top + side))


def _prediction_panel(base: Image.Image, sample: Any, prediction: dict[str, Any]) -> Image.Image:
    transform = Affine(*(sample.crop_transform or [1, 0, 0, 0, 1, 0]))
    operation = EditOperation(prediction["predicted_edit"])
    prior = _geometry(sample.prior_geometry)
    geometry_payload = prediction.get("predicted_geometry")
    predicted = shape(geometry_payload) if geometry_payload is not None else None
    if operation == EditOperation.DELETE:
        panel = _draw_geometry(base, prior, transform, color=COLORS["delete"], width=3)
    elif operation == EditOperation.KEEP:
        panel = _draw_geometry(base, prior, transform, color=COLORS["prediction"], width=3)
    else:
        panel = _draw_geometry(base, predicted, transform, color=COLORS["prediction"], width=3)
    _badge(
        panel,
        operation.value,
        COLORS["delete"] if operation == EditOperation.DELETE else COLORS["prediction"],
    )
    return panel


def _render_category(
    category: str,
    cases: list[dict[str, Any]],
    sample_by_id: dict[str, Any],
    output: Path,
    *,
    tile: int,
) -> None:
    header = 42
    caption = 44
    canvas = Image.new("RGB", (tile * 4, header + len(cases) * (tile + caption)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), category.replace("_", " ").upper(), fill="#202020", font=_font(20))
    labels = ("New image (zoom)", "Old editable vector", "Frozen prediction", "Target")
    for column, label in enumerate(labels):
        draw.text((column * tile + 8, header - 18), label, fill="#404040", font=_font(13))
    for row, case in enumerate(cases):
        prediction = case["prediction"]
        sample = sample_by_id[prediction["sample_id"]]
        array = _load_image(sample.image_path)
        base = Image.fromarray((array * 255).astype("uint8"))
        transform = Affine(*(sample.crop_transform or [1, 0, 0, 0, 1, 0]))
        prior = _geometry(sample.prior_geometry)
        target = _geometry(sample.target_geometry)
        old_panel = _draw_geometry(base, prior, transform, color=COLORS["prior"], width=3)
        predicted_panel = _prediction_panel(base, sample, prediction)
        target_panel = _draw_geometry(base, target, transform, color=COLORS["target"], width=3)
        _badge(target_panel, prediction["target_edit"], COLORS["target"])
        zoom = _zoom_box(sample, prediction, width=base.width, height=base.height)
        y = header + row * (tile + caption)
        for column, panel in enumerate((base, old_panel, predicted_panel, target_panel)):
            canvas.paste(panel.crop(zoom).resize((tile, tile)), (column * tile, y))
        polygon_iou = prediction.get("polygon_iou")
        iou_text = "N/A" if polygon_iou is None else f"{float(polygon_iou):.3f}"
        text = (
            f"{prediction['sample_id']}  target={prediction['target_edit']}  "
            f"pred={prediction['predicted_edit']}  confidence={prediction['confidence']:.3f}  "
            f"polygon_iou={iou_text}"
        )
        draw.text((8, y + tile + 8), text, fill="#202020", font=_font(12))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("failure_manifest", type=Path)
    parser.add_argument("samples", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--per-category", type=int, default=4)
    parser.add_argument("--tile", type=int, default=256)
    args = parser.parse_args()
    payload = json.loads(args.failure_manifest.read_text(encoding="utf-8"))
    samples = load_updater_samples(args.samples, split=args.split)
    sample_by_id = {sample.sample_id: sample for sample in samples}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in payload["cases"]:
        grouped[case["failure_category"]].append(case)
    for category, cases in grouped.items():
        _render_category(
            category,
            cases[: args.per_category],
            sample_by_id,
            args.output_dir / f"{category}.png",
            tile=args.tile,
        )


if __name__ == "__main__":
    main()
