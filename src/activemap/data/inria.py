"""High-resolution Inria building masks adapted as synthetic-prior updater data."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.windows import Window
from shapely import make_valid
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry

from activemap.data.perturb import distort_geometry
from activemap.data.progress import write_progress
from activemap.data.raster_masks import rasterize_geometry_mask
from activemap.data.updater_crops import (
    _normalized_bounds,
    _read_image_crop,
    _square_bounds,
)
from activemap.models import EditOperation, GeoJSONGeometry
from activemap.updater_records import UpdaterSample


def _city(stem: str) -> str:
    match = re.match(r"([A-Za-z-]+)", stem)
    if match is None:
        raise ValueError(f"cannot infer Inria city from filename: {stem}")
    return match.group(1).lower()


def discover_inria_pairs(dataset_root: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for image_path in sorted(dataset_root.rglob("*.tif")):
        if image_path.parent.name.lower() != "images":
            continue
        gt_dir = image_path.parent.parent / "gt"
        candidates = [gt_dir / image_path.name, gt_dir / f"{image_path.stem}.tif"]
        label_path = next((path for path in candidates if path.is_file()), None)
        if label_path is not None:
            pairs.append((image_path, label_path))
    if not pairs:
        raise FileNotFoundError(
            f"no Inria train/images + train/gt GeoTIFF pairs found under {dataset_root}"
        )
    return pairs


def _rank_geometry(geometry: BaseGeometry, key: str, seed: int) -> str:
    return hashlib.sha1(f"{seed}:{key}:{geometry.wkb_hex}".encode()).hexdigest()


def _split_for_city(city: str, validation_cities: set[str], test_cities: set[str]) -> str:
    if city in test_cities:
        return "test"
    if city in validation_cities:
        return "val"
    return "train"


def _extract_objects(
    label_path: Path,
    *,
    min_area_pixels: int,
    maximum: int,
    seed: int,
) -> list[BaseGeometry]:
    with rasterio.open(label_path) as dataset:
        values = dataset.read(1)
        foreground = values > 0
        pixel_area = abs(float(dataset.transform.a * dataset.transform.e))
        minimum_area = min_area_pixels * pixel_area
        objects = [
            make_valid(shape(payload))
            for payload, value in shapes(
                foreground.astype(np.uint8),
                mask=foreground,
                transform=dataset.transform,
            )
            if int(value) == 1
        ]
    objects = [
        geometry
        for geometry in objects
        if not geometry.is_empty and geometry.area >= minimum_area
    ]
    return sorted(
        objects,
        key=lambda geometry: _rank_geometry(geometry, label_path.stem, seed),
    )[:maximum]


def build_inria_updater(
    dataset_root: Path,
    output_dir: Path,
    *,
    image_size: int = 256,
    context_pixels: int = 48,
    max_objects_per_tile: int = 64,
    min_area_pixels: int = 16,
    min_valid_fraction: float = 0.5,
    validation_cities: set[str] | None = None,
    test_cities: set[str] | None = None,
    seed: int = 20260710,
) -> dict[str, Any]:
    """Create KEEP/ADD/RESHAPE samples without claiming temporal supervision."""

    if not 0.0 <= min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must be between 0 and 1")
    validation_cities = {value.lower() for value in (validation_cities or {"vienna"})}
    test_cities = {value.lower() for value in (test_cities or {"kitsap"})}
    if validation_cities & test_cities:
        raise ValueError("Inria validation and test city sets must be disjoint")
    pairs = discover_inria_pairs(dataset_root)
    array_dir = output_dir / "arrays"
    array_dir.mkdir(parents=True, exist_ok=True)
    samples: list[UpdaterSample] = []
    operation_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    city_counts: Counter[str] = Counter()
    skipped_low_valid = 0
    progress_path = output_dir / "progress.json"

    for tile_index, (image_path, label_path) in enumerate(pairs):
        city = _city(image_path.stem)
        split = _split_for_city(city, validation_cities, test_cities)
        objects = _extract_objects(
            label_path,
            min_area_pixels=min_area_pixels,
            maximum=max_objects_per_tile,
            seed=seed,
        )
        with rasterio.open(image_path) as image_dataset:
            pixel_size = max(
                abs(float(image_dataset.transform.a)),
                abs(float(image_dataset.transform.e)),
            )
            for object_index, target_geometry in enumerate(objects):
                bounds = _square_bounds(
                    target_geometry,
                    pixel_size=pixel_size,
                    context_pixels=context_pixels,
                    minimum_pixels=image_size // 2,
                )
                image, _, transform = _read_image_crop(image_dataset, bounds, image_size)
                target = rasterize_geometry_mask(target_geometry, transform, image_size)
                valid = np.any(image > 1e-6, axis=0).astype(np.float32)
                if float(valid.mean()) < min_valid_fraction:
                    skipped_low_valid += 1
                    continue
                base_stem = f"inria-{city}-{image_path.stem}-{object_index:05d}"
                shared_paths = {
                    "image": array_dir / f"{base_stem}-image.npy",
                    "target": array_dir / f"{base_stem}-target.npy",
                    "valid": array_dir / f"{base_stem}-valid.npy",
                }
                np.save(shared_paths["image"], image)
                np.save(shared_paths["target"], target)
                np.save(shared_paths["valid"], valid)
                rng = np.random.default_rng(
                    int(
                        hashlib.sha1(f"{seed}:{base_stem}".encode()).hexdigest()[:8],
                        16,
                    )
                )
                reshape_prior = make_valid(
                    distort_geometry(
                        target_geometry,
                        shift_meters=pixel_size * float(rng.uniform(3.0, 10.0)),
                        area_scale=float(rng.uniform(0.80, 1.25)),
                        rng=rng,
                    )
                )
                operations = {EditOperation.KEEP: target_geometry}
                if split == "train":
                    operations.update(
                        {
                            EditOperation.ADD: None,
                            EditOperation.RESHAPE: reshape_prior,
                        }
                    )
                for operation, prior_geometry in operations.items():
                    prior = rasterize_geometry_mask(prior_geometry, transform, image_size)
                    stem = f"{base_stem}-{operation.value.lower()}"
                    prior_path = array_dir / f"{stem}-prior.npy"
                    np.save(prior_path, prior)
                    target_bounds = _normalized_bounds(target_geometry, bounds)
                    prior_bounds = _normalized_bounds(prior_geometry, bounds)
                    samples.append(
                        UpdaterSample(
                            sample_id=stem,
                            aoi_id=f"inria-{city}-{image_path.stem}",
                            split=split,
                            image_path=shared_paths["image"].relative_to(output_dir).as_posix(),
                            prior_mask_path=prior_path.relative_to(output_dir).as_posix(),
                            target_mask_path=shared_paths["target"].relative_to(
                                output_dir
                            ).as_posix(),
                            valid_mask_path=shared_paths["valid"].relative_to(
                                output_dir
                            ).as_posix(),
                            edit_type=operation,
                            geometry_delta=np.concatenate(
                                [target_bounds, target_bounds - prior_bounds]
                            ).tolist(),
                            object_id=f"{image_path.stem}-{object_index}",
                            crop_transform=list(transform)[:6],
                            crs=(
                                image_dataset.crs.to_string()
                                if image_dataset.crs is not None
                                else None
                            ),
                            prior_geometry=(
                                GeoJSONGeometry.model_validate(mapping(prior_geometry))
                                if prior_geometry is not None
                                else None
                            ),
                            target_geometry=GeoJSONGeometry.model_validate(
                                mapping(target_geometry)
                            ),
                            clear_fraction=float(valid.mean()),
                            quality_source="inria_official_mask",
                            dataset_name="inria_aerial",
                            geometry_family="polygon",
                            supervision_type=(
                                "single_timestamp"
                                if operation == EditOperation.KEEP
                                else "synthetic_prior"
                            ),
                            source_metadata={
                                "city": city,
                                "tile": image_path.stem,
                                "source_image": str(image_path),
                                "source_label": str(label_path),
                                "synthetic_operation": operation.value,
                            },
                        )
                    )
                    operation_counts[operation.value] += 1
                    split_counts[split] += 1
                    city_counts[city] += 1
        write_progress(
            progress_path,
            {
                "status": "running",
                "tiles_processed": tile_index + 1,
                "tiles_total": len(pairs),
                "samples": len(samples),
                "operations": dict(operation_counts),
                "splits": dict(split_counts),
            },
        )

    manifest_path = output_dir / "updater_samples.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(sample.model_dump_json() + "\n")
    summary = {
        "dataset": "Inria Aerial Image Labeling",
        "samples": len(samples),
        "operations": dict(sorted(operation_counts.items())),
        "splits": dict(sorted(split_counts.items())),
        "cities": dict(sorted(city_counts.items())),
        "validation_cities": sorted(validation_cities),
        "test_cities": sorted(test_cities),
        "image_size": image_size,
        "max_objects_per_tile": max_objects_per_tile,
        "min_valid_fraction": min_valid_fraction,
        "skipped_low_valid": skipped_low_valid,
        "seed": seed,
        "manifest": str(manifest_path.resolve()),
        "claim_boundary": (
            "single-timestamp boundary pretraining with train-only synthetic priors; "
            "not real temporal map-update evidence"
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_progress(progress_path, {"status": "complete", **summary})
    return summary


def build_inria_segmentation(
    dataset_root: Path,
    output_dir: Path,
    *,
    image_size: int = 256,
    window_size: int = 512,
    stride: int = 512,
    min_valid_fraction: float = 0.5,
    validation_cities: set[str] | None = None,
    test_cities: set[str] | None = None,
    seed: int = 20260721,
) -> dict[str, Any]:
    """Create unambiguous scene crops supervised by all official buildings."""

    if image_size < 64 or window_size < image_size or stride < 1:
        raise ValueError("invalid Inria segmentation image/window/stride sizes")
    if not 0.0 <= min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must be between 0 and 1")
    validation_cities = {value.lower() for value in (validation_cities or {"vienna"})}
    test_cities = {value.lower() for value in (test_cities or {"kitsap"})}
    if validation_cities & test_cities:
        raise ValueError("Inria validation and test city sets must be disjoint")

    pairs = discover_inria_pairs(dataset_root)
    array_dir = output_dir / "arrays"
    array_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "updater_samples.jsonl"
    progress_path = output_dir / "progress.json"
    samples: list[UpdaterSample] = []
    split_counts: Counter[str] = Counter()
    city_counts: Counter[str] = Counter()
    foreground_counts: Counter[str] = Counter()
    skipped_low_valid = 0

    for pair_index, (image_path, label_path) in enumerate(pairs):
        city = _city(image_path.stem)
        split = _split_for_city(city, validation_cities, test_cities)
        with (
            rasterio.open(image_path) as image_dataset,
            rasterio.open(label_path) as label_dataset,
        ):
            for top in range(0, image_dataset.height, stride):
                for left in range(0, image_dataset.width, stride):
                    window = Window(left, top, window_size, window_size)
                    image = image_dataset.read(
                        indexes=(1, 2, 3),
                        window=window,
                        out_shape=(3, image_size, image_size),
                        boundless=True,
                        fill_value=0,
                        resampling=Resampling.bilinear,
                    ).astype(np.float32)
                    if image.max(initial=0.0) > 1.0:
                        image /= 255.0
                    valid = (
                        image_dataset.read_masks(
                            1,
                            window=window,
                            out_shape=(image_size, image_size),
                            boundless=True,
                            resampling=Resampling.nearest,
                        )
                        > 0
                    ).astype(np.float32)
                    valid *= np.any(image > 1e-6, axis=0).astype(np.float32)
                    if float(valid.mean()) < min_valid_fraction:
                        skipped_low_valid += 1
                        continue
                    target = (
                        label_dataset.read(
                            1,
                            window=window,
                            out_shape=(image_size, image_size),
                            boundless=True,
                            fill_value=0,
                            resampling=Resampling.nearest,
                        )
                        > 0
                    ).astype(np.float32)
                    stem = f"inria-seg-{city}-{image_path.stem}-{top:05d}-{left:05d}"
                    image_output = array_dir / f"{stem}-image.npy"
                    target_output = array_dir / f"{stem}-target.npy"
                    valid_output = array_dir / f"{stem}-valid.npy"
                    np.save(image_output, image)
                    np.save(target_output, target)
                    np.save(valid_output, valid)
                    samples.append(
                        UpdaterSample(
                            sample_id=stem,
                            aoi_id=f"inria-{city}-{image_path.stem}",
                            split=split,
                            image_path=image_output.relative_to(output_dir).as_posix(),
                            prior_mask_path=target_output.relative_to(output_dir).as_posix(),
                            target_mask_path=target_output.relative_to(output_dir).as_posix(),
                            valid_mask_path=valid_output.relative_to(output_dir).as_posix(),
                            edit_type=EditOperation.KEEP,
                            geometry_delta=[0.0] * 8,
                            clear_fraction=float(valid.mean()),
                            quality_source="inria_official_full_scene_mask",
                            dataset_name="inria_aerial_segmentation",
                            geometry_family="polygon",
                            supervision_type="single_timestamp",
                            source_metadata={
                                "city": city,
                                "tile": image_path.stem,
                                "window": [left, top, window_size, window_size],
                                "contains_building": bool(target.any()),
                                "source_image": str(image_path),
                                "source_label": str(label_path),
                            },
                        )
                    )
                    split_counts[split] += 1
                    city_counts[city] += 1
                    foreground_counts["positive" if target.any() else "empty"] += 1
        write_progress(
            progress_path,
            {
                "status": "running",
                "pairs_processed": pair_index + 1,
                "pairs_total": len(pairs),
                "samples": len(samples),
                "splits": dict(split_counts),
                "foreground": dict(foreground_counts),
            },
        )

    with manifest_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(sample.model_dump_json() + "\n")
    summary = {
        "dataset": "Inria Aerial Image Labeling semantic scenes",
        "samples": len(samples),
        "splits": dict(sorted(split_counts.items())),
        "cities": dict(sorted(city_counts.items())),
        "foreground": dict(sorted(foreground_counts.items())),
        "validation_cities": sorted(validation_cities),
        "test_cities": sorted(test_cities),
        "image_size": image_size,
        "window_size": window_size,
        "stride": stride,
        "min_valid_fraction": min_valid_fraction,
        "skipped_low_valid": skipped_low_valid,
        "seed": seed,
        "manifest": str(manifest_path.resolve()),
        "claim_boundary": "single-timestamp semantic building segmentation only",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_progress(progress_path, {"status": "complete", **summary})
    return summary
