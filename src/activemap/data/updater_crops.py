"""Create portable updater crops from typed vector-map snapshot differences."""

from __future__ import annotations

import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry

from activemap.data.edits import EditEvent
from activemap.data.progress import write_progress
from activemap.data.raster_masks import rasterize_geometry_mask
from activemap.data.sn7_pairs import iter_snapshot_pairs
from activemap.models import GeoJSONGeometry
from activemap.updater_records import UpdaterSample


def _square_bounds(
    geometry: BaseGeometry,
    *,
    pixel_size: float,
    context_pixels: int,
    minimum_pixels: int,
) -> tuple[float, float, float, float]:
    min_x, min_y, max_x, max_y = geometry.bounds
    width = max(max_x - min_x, pixel_size * minimum_pixels)
    height = max(max_y - min_y, pixel_size * minimum_pixels)
    side = max(width, height) + 2.0 * context_pixels * pixel_size
    center_x = (min_x + max_x) * 0.5
    center_y = (min_y + max_y) * 0.5
    half = side * 0.5
    return center_x - half, center_y - half, center_x + half, center_y + half


def _window_transform(dataset: rasterio.io.DatasetReader, window: Window, size: int) -> Affine:
    native = dataset.window_transform(window)
    return native * Affine.scale(float(window.width) / size, float(window.height) / size)


def _normalize_image(array: np.ndarray) -> np.ndarray:
    output = array.astype(np.float32)
    finite = output[np.isfinite(output)]
    maximum = float(finite.max(initial=0.0)) if finite.size else 0.0
    if maximum > 1.0:
        denominator = 255.0 if maximum <= 255.0 else 10000.0 if maximum <= 10000 else maximum
        output /= denominator
    return np.clip(np.nan_to_num(output), 0.0, 1.0)


def _read_image_crop(
    dataset: rasterio.io.DatasetReader,
    bounds: tuple[float, float, float, float],
    size: int,
) -> tuple[np.ndarray, Window, Affine]:
    window = from_bounds(*bounds, transform=dataset.transform)
    band_indexes = list(range(1, min(dataset.count, 3) + 1))
    if not band_indexes:
        raise ValueError(f"raster has no bands: {dataset.name}")
    image = dataset.read(
        band_indexes,
        window=window,
        out_shape=(len(band_indexes), size, size),
        boundless=True,
        fill_value=0,
        resampling=Resampling.bilinear,
    )
    while image.shape[0] < 3:
        image = np.concatenate([image, image[-1:]], axis=0)
    return _normalize_image(image[:3]), window, _window_transform(dataset, window, size)


def _valid_mask(
    udm_path: Path | None,
    bounds: tuple[float, float, float, float],
    size: int,
    clear_value: int,
    image_valid: np.ndarray,
) -> tuple[np.ndarray, str]:
    if udm_path is None or not udm_path.is_file():
        return image_valid.astype(np.float32), "masked_image"
    with rasterio.open(udm_path) as dataset:
        window = from_bounds(*bounds, transform=dataset.transform)
        values = dataset.read(
            1,
            window=window,
            out_shape=(size, size),
            boundless=True,
            fill_value=255,
            resampling=Resampling.nearest,
        )
    valid = (values == clear_value) & (image_valid >= 0.5)
    return valid.astype(np.float32), "udm+masked_image"


def _normalized_bounds(
    geometry: BaseGeometry | None,
    crop_bounds: tuple[float, float, float, float],
) -> np.ndarray:
    if geometry is None or geometry.is_empty:
        return np.zeros(4, dtype=np.float32)
    crop_min_x, crop_min_y, crop_max_x, crop_max_y = crop_bounds
    width = max(crop_max_x - crop_min_x, 1e-6)
    height = max(crop_max_y - crop_min_y, 1e-6)
    min_x, min_y, max_x, max_y = geometry.bounds
    return np.asarray(
        [
            (min_x - crop_min_x) / width,
            (min_y - crop_min_y) / height,
            (max_x - crop_min_x) / width,
            (max_y - crop_min_y) / height,
        ],
        dtype=np.float32,
    )


def geometry_delta(event: EditEvent, crop_bounds: tuple[float, float, float, float]) -> np.ndarray:
    target = _normalized_bounds(event.new_geometry, crop_bounds)
    prior = _normalized_bounds(event.old_geometry, crop_bounds)
    return np.concatenate([target, target - prior]).astype(np.float32)


def build_updater_crops(
    manifest: pd.DataFrame,
    output_dir: Path,
    *,
    image_size: int = 128,
    context_pixels: int = 32,
    clear_value: int = 0,
    id_column: str | None = None,
    keep_iou_min: float = 0.80,
    fallback_match_iou_min: float = 0.20,
    max_centroid_distance: float | None = None,
    min_area: float = 0.0,
    max_events_per_operation: int | None = None,
    max_invalid_fraction: float = 0.50,
    max_month_gap: int | None = None,
    min_change_persistence: int = 1,
    sampling_seed: int = 20260710,
    derivation_version: str = "sn7-adjacent-v3-distance-gated",
    include_prior_image: bool = False,
) -> dict[str, object]:
    """Write current image/prior/target arrays and optional aligned old imagery.

    The optional old image is deliberately stored separately from the current
    image.  A temporal updater must opt in at loading time, which keeps legacy
    v4 manifests and checkpoints byte-for-byte compatible.
    """

    if image_size < 16:
        raise ValueError("image_size must be at least 16")
    output_dir.mkdir(parents=True, exist_ok=True)
    array_dir = output_dir / "arrays"
    array_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / "updater_samples.jsonl"
    samples: list[UpdaterSample] = []
    operation_counts: Counter[str] = Counter()
    skipped_low_valid: Counter[str] = Counter()
    pair_count = 0
    progress_path = output_dir / "progress.json"
    write_progress(
        progress_path,
        {"status": "running", "snapshot_pairs": 0, "samples": 0},
    )

    pairs = iter_snapshot_pairs(
        manifest,
        id_column=id_column,
        keep_iou_min=keep_iou_min,
        fallback_match_iou_min=fallback_match_iou_min,
        max_centroid_distance=max_centroid_distance,
        min_area=min_area,
        max_events_per_operation=max_events_per_operation,
        max_month_gap=max_month_gap,
        min_change_persistence=min_change_persistence,
        sampling_seed=sampling_seed,
    )
    for pair in pairs:
        pair_count += 1
        with ExitStack() as stack:
            dataset = stack.enter_context(rasterio.open(pair.image_path))
            old_dataset = (
                stack.enter_context(rasterio.open(pair.old_image_path))
                if include_prior_image
                else None
            )
            if old_dataset is not None and old_dataset.crs != dataset.crs:
                raise ValueError(
                    "paired temporal crops require old and current images to share a CRS: "
                    f"{pair.old_image_path} vs {pair.image_path}"
                )
            pixel_size = max(abs(float(dataset.transform.a)), abs(float(dataset.transform.e)))
            for event_index, event in enumerate(pair.events):
                reference = (
                    event.new_geometry if event.new_geometry is not None else event.old_geometry
                )
                if reference is None or reference.is_empty:
                    continue
                crop_bounds = _square_bounds(
                    reference,
                    pixel_size=pixel_size,
                    context_pixels=context_pixels,
                    minimum_pixels=image_size // 4,
                )
                image, _, transform = _read_image_crop(dataset, crop_bounds, image_size)
                prior_image = (
                    _read_image_crop(old_dataset, crop_bounds, image_size)[0]
                    if old_dataset is not None
                    else None
                )
                prior = rasterize_geometry_mask(event.old_geometry, transform, image_size)
                target = rasterize_geometry_mask(event.new_geometry, transform, image_size)
                image_valid = np.any(image > 1e-6, axis=0).astype(np.float32)
                valid, quality_source = _valid_mask(
                    pair.udm_path,
                    crop_bounds,
                    image_size,
                    clear_value=clear_value,
                    image_valid=image_valid,
                )
                if prior_image is not None:
                    prior_image_valid = np.any(prior_image > 1e-6, axis=0).astype(np.float32)
                    prior_valid, prior_quality_source = _valid_mask(
                        pair.old_udm_path,
                        crop_bounds,
                        image_size,
                        clear_value=clear_value,
                        image_valid=prior_image_valid,
                    )
                    valid = valid * prior_valid
                    quality_source = f"{quality_source}+prior_{prior_quality_source}"
                clear_fraction = float(np.mean(valid >= 0.5))
                if 1.0 - clear_fraction > max_invalid_fraction:
                    skipped_low_valid[event.op.value] += 1
                    continue
                stem = (
                    f"{pair.aoi_id}-{pair.old_timestamp}-{pair.new_timestamp}-"
                    f"{event.op.value.lower()}-{event_index:06d}"
                )
                paths = {
                    "image": array_dir / f"{stem}-image.npy",
                    "prior_image": array_dir / f"{stem}-prior-image.npy",
                    "prior": array_dir / f"{stem}-prior.npy",
                    "target": array_dir / f"{stem}-target.npy",
                    "valid": array_dir / f"{stem}-valid.npy",
                }
                np.save(paths["image"], image)
                if prior_image is not None:
                    np.save(paths["prior_image"], prior_image)
                np.save(paths["prior"], prior)
                np.save(paths["target"], target)
                np.save(paths["valid"], valid)
                samples.append(
                    UpdaterSample(
                        sample_id=stem,
                        aoi_id=pair.aoi_id,
                        split=pair.split,
                        image_path=paths["image"].relative_to(output_dir).as_posix(),
                        prior_image_path=(
                            paths["prior_image"].relative_to(output_dir).as_posix()
                            if prior_image is not None
                            else None
                        ),
                        prior_mask_path=paths["prior"].relative_to(output_dir).as_posix(),
                        target_mask_path=paths["target"].relative_to(output_dir).as_posix(),
                        valid_mask_path=paths["valid"].relative_to(output_dir).as_posix(),
                        edit_type=event.op,
                        geometry_delta=geometry_delta(event, crop_bounds).tolist(),
                        object_id=event.object_id,
                        crop_transform=list(transform)[:6],
                        crs=dataset.crs.to_string() if dataset.crs is not None else None,
                        prior_geometry=(
                            GeoJSONGeometry.model_validate(mapping(event.old_geometry))
                            if event.old_geometry is not None
                            else None
                        ),
                        target_geometry=(
                            GeoJSONGeometry.model_validate(mapping(event.new_geometry))
                            if event.new_geometry is not None
                            else None
                        ),
                        clear_fraction=clear_fraction,
                        quality_source=quality_source,
                        dataset_name="spacenet7",
                        geometry_family="polygon",
                        supervision_type="real_temporal",
                        source_metadata={
                            "old_timestamp": pair.old_timestamp,
                            "new_timestamp": pair.new_timestamp,
                            "month_gap": pair.month_gap,
                        },
                    )
                )
                operation_counts[event.op.value] += 1
        write_progress(
            progress_path,
            {
                "status": "running",
                "snapshot_pairs": pair_count,
                "samples": len(samples),
                "operations": dict(sorted(operation_counts.items())),
                "skipped_low_valid": dict(sorted(skipped_low_valid.items())),
                "current_aoi": pair.aoi_id,
                "current_old_timestamp": pair.old_timestamp,
                "current_new_timestamp": pair.new_timestamp,
            },
        )

    with sample_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(sample.model_dump_json() + "\n")
    summary: dict[str, object] = {
        "snapshot_pairs": pair_count,
        "samples": len(samples),
        "operations": dict(sorted(operation_counts.items())),
        "skipped_low_valid": dict(sorted(skipped_low_valid.items())),
        "manifest": str(sample_path.resolve()),
        "image_size": image_size,
        "context_pixels": context_pixels,
        "max_month_gap": max_month_gap,
        "min_change_persistence": min_change_persistence,
        "max_events_per_operation": max_events_per_operation,
        "max_invalid_fraction": max_invalid_fraction,
        "sampling_seed": sampling_seed,
        "derivation_version": derivation_version,
        "temporal_pair_input": include_prior_image,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_progress(progress_path, {"status": "complete", **summary})
    return summary
