"""Structural and semantic audits for generated updater crop datasets."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import shape

from activemap.models import EditOperation
from activemap.updater_records import load_updater_samples


def _mask(path: str) -> np.ndarray:
    values = np.load(path).astype(np.float32).squeeze()
    if values.ndim != 2:
        raise ValueError(f"mask must be two-dimensional after squeeze: {path}")
    return values


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    left_mask = left >= 0.5
    right_mask = right >= 0.5
    union = np.sum(left_mask | right_mask)
    return float(np.sum(left_mask & right_mask) / union) if union else 1.0


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def audit_updater_dataset(
    samples_path: Path,
    *,
    max_invalid_fraction: float = 0.50,
    keep_iou_min: float = 0.70,
    reshape_iou_max: float = 0.95,
    max_reshape_centroid_distance: float | None = 20.0,
    allow_empty_keep: bool = False,
    allow_nonlocal_polyline_reshape: bool = False,
) -> dict[str, Any]:
    if not 0.0 <= max_invalid_fraction <= 1.0:
        raise ValueError("max_invalid_fraction must be between zero and one")
    samples = load_updater_samples(samples_path)
    errors: list[str] = []
    warnings: list[str] = []
    sample_ids = [sample.sample_id for sample in samples]
    duplicates = [name for name, count in Counter(sample_ids).items() if count > 1]
    if duplicates:
        errors.append(f"duplicate sample IDs: {duplicates[:10]}")

    aoi_splits: dict[str, set[str]] = defaultdict(set)
    source_scenario_counts: Counter[str] = Counter()
    valid_fractions: list[float] = []
    prior_pixels: list[float] = []
    target_pixels: list[float] = []
    mask_ious: list[float] = []
    added_pixel_counts: list[float] = []
    removed_pixel_counts: list[float] = []
    black_fractions: list[float] = []
    image_shapes: Counter[str] = Counter()
    quality_sources: Counter[str] = Counter()
    low_valid_count = 0
    valid_black_mismatch_count = 0
    empty_required_count = 0
    empty_keep_count = 0
    temporal_contract_violation_count = 0
    semantic_violation_count = 0
    reshape_distance_violation_count = 0
    nonlocal_polyline_reshape_count = 0
    reshape_centroid_distances: list[float] = []
    temporal_pair_sample_count = sum(
        sample.prior_image_path is not None for sample in samples
    )
    temporal_pair_expected = temporal_pair_sample_count > 0
    temporal_pair_missing_count = 0
    temporal_pair_shape_mismatch_count = 0
    temporal_valid_mismatch_count = 0

    for sample in samples:
        aoi_splits[sample.aoi_id or sample.sample_id].add(sample.split)
        annotation_index = sample.source_metadata.get("annotation_index")
        source_scenario_key = (
            f"{sample.dataset_name}:{sample.aoi_id}:{annotation_index}"
            if annotation_index is not None
            else f"{sample.dataset_name}:{sample.object_id or sample.sample_id}"
        )
        source_scenario_counts[source_scenario_key] += 1
        try:
            image = np.load(sample.image_path)
            prior_image = (
                np.load(sample.prior_image_path)
                if sample.prior_image_path is not None
                else None
            )
            prior = _mask(sample.prior_mask_path)
            target = _mask(sample.target_mask_path)
            valid = (
                _mask(sample.valid_mask_path)
                if sample.valid_mask_path is not None
                else np.ones_like(target)
            )
        except Exception as exc:
            errors.append(f"{sample.sample_id}: failed to load arrays: {exc}")
            continue
        image_shapes[str(tuple(image.shape))] += 1
        quality_sources[sample.quality_source or "unspecified"] += 1
        if image.ndim != 3 or image.shape[-2:] != prior.shape:
            errors.append(
                f"{sample.sample_id}: image/mask shape mismatch {image.shape} vs {prior.shape}"
            )
            continue
        if temporal_pair_expected and prior_image is None:
            temporal_pair_missing_count += 1
            errors.append(f"{sample.sample_id}: temporal pair is missing prior_image_path")
            continue
        if prior_image is not None:
            if prior_image.shape != image.shape:
                temporal_pair_shape_mismatch_count += 1
                errors.append(
                    f"{sample.sample_id}: old/current image shapes differ "
                    f"{prior_image.shape} vs {image.shape}"
                )
                continue
            if not np.isfinite(prior_image).all():
                errors.append(f"{sample.sample_id}: prior image contains non-finite values")
            if (
                float(prior_image.min(initial=0.0)) < 0.0
                or float(prior_image.max(initial=0.0)) > 1.0
            ):
                errors.append(f"{sample.sample_id}: prior image values fall outside [0,1]")
        if prior.shape != target.shape or target.shape != valid.shape:
            errors.append(f"{sample.sample_id}: prior/target/valid shapes differ")
            continue
        arrays = (("image", image), ("prior", prior), ("target", target), ("valid", valid))
        for name, values in arrays:
            if not np.isfinite(values).all():
                errors.append(f"{sample.sample_id}: {name} contains non-finite values")
            if float(values.min(initial=0.0)) < 0.0 or float(values.max(initial=0.0)) > 1.0:
                errors.append(f"{sample.sample_id}: {name} values fall outside [0,1]")

        valid_fraction = float(np.mean(valid >= 0.5))
        channel_axis = 0 if image.shape[0] in {1, 3, 4} else -1
        image_nonblack = np.any(image > 1e-6, axis=channel_axis)
        black_fraction = float(np.mean(~image_nonblack))
        black_fractions.append(black_fraction)
        if np.any((valid >= 0.5) & ~image_nonblack):
            valid_black_mismatch_count += 1
            warnings.append(f"{sample.sample_id}: valid mask includes all-black pixels")
        if prior_image is not None:
            prior_image_nonblack = np.any(prior_image > 1e-6, axis=channel_axis)
            if np.any((valid >= 0.5) & ~(image_nonblack & prior_image_nonblack)):
                temporal_valid_mismatch_count += 1
                warnings.append(
                    f"{sample.sample_id}: temporal valid mask includes a black old/current pixel"
                )
        prior_count = float(np.sum(prior >= 0.5))
        target_count = float(np.sum(target >= 0.5))
        prior_binary = prior >= 0.5
        target_binary = target >= 0.5
        added_count = float(np.sum(target_binary & ~prior_binary))
        removed_count = float(np.sum(prior_binary & ~target_binary))
        iou = _iou(prior, target)
        valid_fractions.append(valid_fraction)
        prior_pixels.append(prior_count)
        target_pixels.append(target_count)
        mask_ious.append(iou)
        added_pixel_counts.append(added_count)
        removed_pixel_counts.append(removed_count)
        if 1.0 - valid_fraction > max_invalid_fraction:
            low_valid_count += 1
            warnings.append(f"{sample.sample_id}: invalid fraction {1.0 - valid_fraction:.3f}")

        full_scene_temporal = sample.supervision_type == "full_scene_temporal"
        required_empty = False
        required_nonempty = False
        if full_scene_temporal:
            if sample.edit_type == EditOperation.ADD:
                missing_temporal_evidence = added_count == 0
            elif sample.edit_type == EditOperation.DELETE:
                missing_temporal_evidence = removed_count == 0
            elif sample.edit_type == EditOperation.RESHAPE:
                missing_temporal_evidence = added_count == 0 or removed_count == 0
            else:
                missing_temporal_evidence = False
            if missing_temporal_evidence:
                temporal_contract_violation_count += 1
                errors.append(
                    f"{sample.sample_id}: full-scene temporal evidence missing for "
                    f"{sample.edit_type.value}"
                )
        else:
            required_empty = (sample.edit_type == EditOperation.ADD and prior_count > 0) or (
                sample.edit_type == EditOperation.DELETE and target_count > 0
            )
        if sample.edit_type == EditOperation.KEEP:
            if prior_count == 0 and target_count == 0:
                empty_keep_count += 1
            required_nonempty = (
                (prior_count == 0) != (target_count == 0)
                if allow_empty_keep
                else prior_count == 0 or target_count == 0
            )
        elif not full_scene_temporal:
            required_nonempty = (
                sample.edit_type in {EditOperation.DELETE, EditOperation.RESHAPE}
                and prior_count == 0
            ) or (
                sample.edit_type in {EditOperation.ADD, EditOperation.RESHAPE} and target_count == 0
            )
        if required_empty or required_nonempty:
            empty_required_count += 1
            errors.append(
                f"{sample.sample_id}: empty/non-empty mask contract violated for "
                f"{sample.edit_type.value}"
            )
        if sample.edit_type == EditOperation.KEEP and iou < keep_iou_min:
            semantic_violation_count += 1
            warnings.append(f"{sample.sample_id}: KEEP raster IoU is {iou:.3f}")
        if sample.edit_type == EditOperation.RESHAPE and iou >= reshape_iou_max:
            semantic_violation_count += 1
            warnings.append(f"{sample.sample_id}: RESHAPE raster IoU is {iou:.3f}")
        if (
            sample.edit_type == EditOperation.RESHAPE
            and not full_scene_temporal
            and sample.prior_geometry is not None
            and sample.target_geometry is not None
        ):
            prior_geometry = shape(sample.prior_geometry.model_dump(mode="json"))
            target_geometry = shape(sample.target_geometry.model_dump(mode="json"))
            distance = float(prior_geometry.centroid.distance(target_geometry.centroid))
            reshape_centroid_distances.append(distance)
            if (
                max_reshape_centroid_distance is not None
                and distance > max_reshape_centroid_distance
            ):
                if allow_nonlocal_polyline_reshape and sample.geometry_family == "polyline":
                    nonlocal_polyline_reshape_count += 1
                else:
                    reshape_distance_violation_count += 1
                    errors.append(
                        f"{sample.sample_id}: RESHAPE centroid distance "
                        f"{distance:.3f} exceeds "
                        f"{max_reshape_centroid_distance:.3f}"
                    )

    leaked = {aoi: sorted(splits) for aoi, splits in aoi_splits.items() if len(splits) > 1}
    if leaked:
        errors.append(f"AOIs occur in multiple splits: {leaked}")
    summary: dict[str, Any] = {
        "source": str(samples_path.resolve()),
        "allow_empty_keep": allow_empty_keep,
        "allow_nonlocal_polyline_reshape": allow_nonlocal_polyline_reshape,
        "sample_count": len(samples),
        "aoi_count": len(aoi_splits),
        "source_scenario_count": len(source_scenario_counts),
        "patches_per_source_scenario": (
            _quantiles([float(value) for value in source_scenario_counts.values()])
            if source_scenario_counts
            else None
        ),
        "largest_source_scenarios": [
            {"source_scenario": key, "patches": count}
            for key, count in source_scenario_counts.most_common(10)
        ],
        "split_counts": dict(sorted(Counter(sample.split for sample in samples).items())),
        "operation_counts": dict(
            sorted(Counter(sample.edit_type.value for sample in samples).items())
        ),
        "image_shapes": dict(sorted(image_shapes.items())),
        "temporal_pair_input": temporal_pair_expected,
        "temporal_pair_sample_count": temporal_pair_sample_count,
        "temporal_pair_missing_count": temporal_pair_missing_count,
        "temporal_pair_shape_mismatch_count": temporal_pair_shape_mismatch_count,
        "temporal_valid_mismatch_count": temporal_valid_mismatch_count,
        "quality_sources": dict(sorted(quality_sources.items())),
        "black_fraction": _quantiles(black_fractions) if black_fractions else None,
        "valid_fraction": _quantiles(valid_fractions) if valid_fractions else None,
        "prior_foreground_pixels": _quantiles(prior_pixels) if prior_pixels else None,
        "target_foreground_pixels": _quantiles(target_pixels) if target_pixels else None,
        "prior_target_iou": _quantiles(mask_ious) if mask_ious else None,
        "added_pixels": _quantiles(added_pixel_counts) if added_pixel_counts else None,
        "removed_pixels": (_quantiles(removed_pixel_counts) if removed_pixel_counts else None),
        "low_valid_count": low_valid_count,
        "valid_black_mismatch_count": valid_black_mismatch_count,
        "empty_contract_violations": empty_required_count,
        "empty_keep_count": empty_keep_count,
        "temporal_contract_violations": temporal_contract_violation_count,
        "semantic_warnings": semantic_violation_count,
        "reshape_centroid_distance": (
            _quantiles(reshape_centroid_distances) if reshape_centroid_distances else None
        ),
        "reshape_distance_violations": reshape_distance_violation_count,
        "nonlocal_polyline_reshape_count": nonlocal_polyline_reshape_count,
        "max_reshape_centroid_distance": max_reshape_centroid_distance,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors[:100],
        "warnings": warnings[:100],
        "passed": not errors,
    }
    return summary


def write_updater_audit(summary: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
