"""Render updater predictions as editable before/after vector-map operations."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from affine import Affine
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry

from activemap.inference import UpdaterPredictor
from activemap.models import EditOperation
from activemap.updater_records import UpdaterSample, load_updater_samples
from activemap.vector_map import topology_is_valid, vectorize_mask

EDIT_ORDER = list(EditOperation)
COLORS = {
    "prior": (255, 191, 0, 255),
    "prediction": (0, 214, 201, 255),
    "target": (60, 220, 92, 255),
    "delete": (247, 76, 76, 255),
}


def _load_image(path: str) -> np.ndarray:
    image = np.load(path).astype(np.float32)
    if image.shape[0] in {1, 3, 4}:
        image = np.moveaxis(image[:3], 0, -1)
    if image.max(initial=0.0) > 1.0:
        image /= 255.0
    return np.clip(image, 0.0, 1.0)


def _load_mask(path: str) -> np.ndarray:
    return np.asarray(np.load(path), dtype=np.float32).squeeze()


def _iter_polygons(geometry: BaseGeometry | None) -> list[BaseGeometry]:
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    if geometry.geom_type == "MultiPolygon":
        return list(geometry.geoms)
    if geometry.geom_type == "GeometryCollection":
        return [part for item in geometry.geoms for part in _iter_polygons(item)]
    return []


def _draw_geometry(
    image: Image.Image,
    geometry: BaseGeometry | None,
    transform: Affine,
    *,
    color: tuple[int, int, int, int],
    width: int = 2,
    fill_alpha: int = 0,
) -> Image.Image:
    overlay = image.convert("RGBA")
    draw = ImageDraw.Draw(overlay, "RGBA")
    inverse = ~transform
    for polygon in _iter_polygons(geometry):
        exterior = [inverse * (x, y) for x, y in polygon.exterior.coords]
        if fill_alpha:
            draw.polygon(exterior, fill=(*color[:3], fill_alpha))
        draw.line(exterior, fill=color, width=width, joint="curve")
        for interior in polygon.interiors:
            ring = [inverse * (x, y) for x, y in interior.coords]
            draw.line(ring, fill=color, width=width, joint="curve")
    return overlay.convert("RGB")


def _mask_overlay(image: np.ndarray, probability: np.ndarray) -> Image.Image:
    output = image.copy()
    mask = probability >= 0.5
    output[mask] = output[mask] * 0.45 + np.asarray([0.95, 0.15, 0.12]) * 0.55
    return Image.fromarray((np.clip(output, 0.0, 1.0) * 255).astype(np.uint8))


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _geometry(sample_geometry: Any) -> BaseGeometry | None:
    if sample_geometry is None:
        return None
    return shape(sample_geometry.model_dump(mode="json"))


def _predict(
    predictor: UpdaterPredictor,
    sample: UpdaterSample,
    *,
    threshold: float,
) -> dict[str, Any]:
    image = _load_image(sample.image_path)
    prior_mask = _load_mask(sample.prior_mask_path)
    result = predictor.predict(image, prior_mask)
    probabilities = np.asarray(result["edit_probabilities"])
    predicted_edit = EDIT_ORDER[int(np.argmax(probabilities))]
    mask_probability = np.asarray(result["mask_probability"])
    transform = Affine(*(sample.crop_transform or [1, 0, 0, 0, 1, 0]))
    predicted_geometry = vectorize_mask(mask_probability, transform)
    target_geometry = _geometry(sample.target_geometry)
    if predicted_geometry is None and target_geometry is None:
        polygon_iou = 1.0
    elif predicted_geometry is None or target_geometry is None:
        polygon_iou = 0.0
    else:
        union = predicted_geometry.union(target_geometry).area
        polygon_iou = (
            predicted_geometry.intersection(target_geometry).area / union
            if union > 0.0
            else 0.0
        )
    confidence = float(result["confidence"])
    topology_valid = (
        topology_is_valid(predicted_geometry)
        if predicted_edit in {EditOperation.ADD, EditOperation.RESHAPE}
        else True
    )
    committed = confidence >= threshold and topology_valid
    return {
        "image": image,
        "mask_probability": mask_probability,
        "predicted_edit": predicted_edit,
        "edit_probability": float(probabilities.max()),
        "confidence": confidence,
        "predicted_geometry": predicted_geometry,
        "polygon_iou": polygon_iou,
        "topology_valid": topology_valid,
        "committed": committed,
        "transform": transform,
    }


def _select_examples(
    predictor: UpdaterPredictor,
    samples: list[UpdaterSample],
    *,
    threshold: float,
    candidates_per_edit: int,
) -> list[tuple[UpdaterSample, dict[str, Any]]]:
    grouped: dict[EditOperation, list[UpdaterSample]] = defaultdict(list)
    for sample in sorted(samples, key=lambda item: item.sample_id):
        grouped[sample.edit_type].append(sample)
    selected: list[tuple[UpdaterSample, dict[str, Any]]] = []
    for operation in EDIT_ORDER:
        candidates: list[tuple[UpdaterSample, dict[str, Any]]] = []
        for sample in grouped[operation][:candidates_per_edit]:
            prediction = _predict(predictor, sample, threshold=threshold)
            candidates.append((sample, prediction))
        clear = [item for item in candidates if (item[0].clear_fraction or 0.0) >= 0.95]
        correct = [
            item for item in clear if item[1]["predicted_edit"] == operation
        ] or [item for item in candidates if item[1]["predicted_edit"] == operation]
        pool = correct or candidates
        if pool:
            selected.append(max(pool, key=_example_score))
    return selected


def _example_score(item: tuple[UpdaterSample, dict[str, Any]]) -> float:
    sample, prediction = item
    reference = _geometry(sample.target_geometry) or _geometry(sample.prior_geometry)
    transform = prediction["transform"]
    pixel_area = (
        reference.area / max(abs(transform.a * transform.e), 1e-6)
        if reference is not None
        else 0.0
    )
    clear_fraction = sample.clear_fraction or 0.0
    geometry_quality = 0.25 + float(prediction["polygon_iou"])
    if sample.edit_type == EditOperation.DELETE:
        geometry_quality = 1.0
    return float(
        np.log1p(pixel_area)
        * (0.5 + prediction["confidence"])
        * clear_fraction
        * geometry_quality**2
    )


def _object_roi(
    geometries: list[BaseGeometry | None], transform: Affine, size: int
) -> tuple[int, int, int, int]:
    inverse = ~transform
    pixel_bounds = []
    for geometry in geometries:
        if geometry is None or geometry.is_empty:
            continue
        min_x, min_y, max_x, max_y = geometry.bounds
        corners = [
            inverse * (min_x, min_y),
            inverse * (min_x, max_y),
            inverse * (max_x, min_y),
            inverse * (max_x, max_y),
        ]
        pixel_bounds.extend(corners)
    if not pixel_bounds:
        return (0, 0, size, size)
    xs = [point[0] for point in pixel_bounds]
    ys = [point[1] for point in pixel_bounds]
    center_x = (min(xs) + max(xs)) / 2.0
    center_y = (min(ys) + max(ys)) / 2.0
    side = max(max(xs) - min(xs), max(ys) - min(ys)) + 20.0
    side = min(max(side, 44.0), float(size))
    left = min(max(center_x - side / 2.0, 0.0), size - side)
    top = min(max(center_y - side / 2.0, 0.0), size - side)
    return (
        int(np.floor(left)),
        int(np.floor(top)),
        int(np.ceil(left + side)),
        int(np.ceil(top + side)),
    )


def _render_base_and_transform(
    image: np.ndarray,
    transform: Affine,
    roi: tuple[int, int, int, int],
    tile: int,
) -> tuple[Image.Image, Affine]:
    left, top, right, bottom = roi
    source = Image.fromarray((image * 255).astype(np.uint8))
    base = source.crop(roi).resize((tile, tile), Image.Resampling.LANCZOS)
    panel_transform = (
        transform
        * Affine.translation(left, top)
        * Affine.scale((right - left) / tile, (bottom - top) / tile)
    )
    return base, panel_transform


def _render_mask_panel(
    base: Image.Image,
    probability: np.ndarray,
    roi: tuple[int, int, int, int],
    tile: int,
) -> Image.Image:
    probability_image = Image.fromarray(probability.astype(np.float32), mode="F")
    resized = np.asarray(
        probability_image.crop(roi).resize((tile, tile), Image.Resampling.BILINEAR)
    )
    return _mask_overlay(np.asarray(base, dtype=np.float32) / 255.0, resized)


def _badge(image: Image.Image, label: str, color: tuple[int, int, int, int]) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    font = _font(18)
    box = draw.textbbox((0, 0), label, font=font)
    width = box[2] - box[0] + 18
    draw.rounded_rectangle((8, 8, 8 + width, 38), radius=4, fill=color)
    draw.text((17, 12), label, font=font, fill="white")


def _committed_geometry(
    sample: UpdaterSample, prediction: dict[str, Any]
) -> BaseGeometry | None:
    prior = _geometry(sample.prior_geometry)
    if not prediction["committed"]:
        return prior
    operation = prediction["predicted_edit"]
    if operation == EditOperation.DELETE:
        return None
    if operation in {EditOperation.ADD, EditOperation.RESHAPE}:
        return prediction["predicted_geometry"]
    return prior


def render_examples(
    examples: list[tuple[UpdaterSample, dict[str, Any]]], output_path: Path
) -> None:
    tile = 384
    header = 76
    caption = 58
    columns = (
        "New image",
        "Old editable vector",
        "Predicted mask",
        "Candidate vector edit",
        "Committed map / target",
    )
    canvas = Image.new(
        "RGB", (tile * len(columns), header + len(examples) * (tile + caption)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    heading_font = _font(20)
    caption_font = _font(18)
    for column, heading in enumerate(columns):
        draw.text((column * tile + 12, 20), heading, fill="black", font=heading_font)
    draw.text(
        (12, 50),
        "yellow=prior   red=mask/delete   cyan=committed prediction   green=target",
        fill=(70, 70, 70),
        font=_font(15),
    )
    for row, (sample, prediction) in enumerate(examples):
        image_array = prediction["image"]
        transform = prediction["transform"]
        prior = _geometry(sample.prior_geometry)
        target = _geometry(sample.target_geometry)
        predicted = prediction["predicted_geometry"]
        committed = _committed_geometry(sample, prediction)
        roi = _object_roi([prior, target, predicted], transform, image_array.shape[0])
        base, panel_transform = _render_base_and_transform(image_array, transform, roi, tile)
        old_panel = _draw_geometry(
            base, prior, panel_transform, color=COLORS["prior"], width=5, fill_alpha=28
        )
        if prior is None:
            _badge(old_panel, "NO PRIOR OBJECT", (100, 100, 100, 220))
        mask_panel = _render_mask_panel(
            base, prediction["mask_probability"], roi, tile
        )
        candidate_panel = _draw_geometry(
            base, prior, panel_transform, color=COLORS["prior"], width=5, fill_alpha=20
        )
        operation = prediction["predicted_edit"]
        if operation == EditOperation.DELETE:
            candidate_panel = _draw_geometry(
                base,
                prior,
                panel_transform,
                color=COLORS["delete"],
                width=6,
                fill_alpha=35,
            )
        elif operation in {EditOperation.ADD, EditOperation.RESHAPE}:
            candidate_panel = _draw_geometry(
                candidate_panel,
                predicted,
                panel_transform,
                color=COLORS["prediction"],
                width=6,
                fill_alpha=28,
            )
        badge_color = (
            COLORS["delete"]
            if operation == EditOperation.DELETE
            else COLORS["prediction"]
        )
        _badge(candidate_panel, operation.value, badge_color)
        committed_panel = _draw_geometry(
            base,
            committed,
            panel_transform,
            color=COLORS["prediction"],
            width=7,
            fill_alpha=22,
        )
        committed_panel = _draw_geometry(
            committed_panel,
            target,
            panel_transform,
            color=COLORS["target"],
            width=4,
        )
        if operation == EditOperation.DELETE and prediction["committed"]:
            _badge(committed_panel, "OBJECT REMOVED", COLORS["delete"])
        panels = (base, old_panel, mask_panel, candidate_panel, committed_panel)
        y = header + row * (tile + caption)
        for column, panel in enumerate(panels):
            canvas.paste(panel, (column * tile, y))
        status = "COMMIT" if prediction["committed"] else "REJECT"
        polygon_iou = (
            "N/A"
            if operation == EditOperation.DELETE
            else f"{prediction['polygon_iou']:.3f}"
        )
        text = (
            f"target={sample.edit_type.value}  pred={prediction['predicted_edit'].value}  "
            f"edit_p={prediction['edit_probability']:.3f}  "
            f"confidence={prediction['confidence']:.3f}  decision={status}"
            f"  polygon_iou={polygon_iou}"
        )
        draw.text((12, y + tile + 14), text, fill="black", font=caption_font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, dpi=(300, 300))


def write_geojson(
    examples: list[tuple[UpdaterSample, dict[str, Any]]], output_path: Path
) -> None:
    features = []
    for sample, prediction in examples:
        geometries = {
            "prior": _geometry(sample.prior_geometry),
            "predicted": prediction["predicted_geometry"],
            "target": _geometry(sample.target_geometry),
            "committed": _committed_geometry(sample, prediction),
        }
        for role, geometry in geometries.items():
            if geometry is None or geometry.is_empty:
                continue
            features.append(
                {
                    "type": "Feature",
                    "geometry": mapping(geometry),
                    "properties": {
                        "sample_id": sample.sample_id,
                        "role": role,
                        "target_edit": sample.edit_type.value,
                        "predicted_edit": prediction["predicted_edit"].value,
                        "confidence": prediction["confidence"],
                        "committed": prediction["committed"],
                        "object_id": sample.object_id,
                        "crs": sample.crs,
                    },
                }
            )
    output_path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("samples", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--commit-threshold", type=float, default=0.5)
    parser.add_argument("--candidates-per-edit", type=int, default=128)
    args = parser.parse_args()
    predictor = UpdaterPredictor(args.checkpoint, device=args.device)
    samples = load_updater_samples(args.samples, split=args.split)
    examples = _select_examples(
        predictor,
        samples,
        threshold=args.commit_threshold,
        candidates_per_edit=args.candidates_per_edit,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.output_dir / "vector_update_examples.png"
    geojson_path = args.output_dir / "vector_update_examples.geojson"
    render_examples(examples, figure_path)
    write_geojson(examples, geojson_path)
    print(
        json.dumps(
            {
                "figure": str(figure_path.resolve()),
                "geojson": str(geojson_path.resolve()),
                "samples": [sample.sample_id for sample, _ in examples],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
