"""Deterministically refresh updater masks from stored vector provenance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from affine import Affine
from shapely.geometry import shape

from activemap.data.raster_masks import rasterize_geometry_mask
from activemap.updater_records import load_updater_samples


def refresh_updater_masks(
    samples_path: Path,
    report_path: Path,
    *,
    write: bool = False,
) -> dict[str, Any]:
    """Re-rasterize masks using geometry and affine fields stored in each sample."""

    samples = load_updater_samples(samples_path)
    changed_prior = 0
    changed_target = 0
    skipped_missing_provenance = 0
    affected_ids: list[str] = []
    for sample in samples:
        if sample.crop_transform is None or (
            sample.prior_geometry is None and sample.target_geometry is None
        ):
            skipped_missing_provenance += 1
            continue
        prior_path = Path(sample.prior_mask_path)
        target_path = Path(sample.target_mask_path)
        old_prior = np.load(prior_path).astype(np.float32).squeeze()
        old_target = np.load(target_path).astype(np.float32).squeeze()
        if old_prior.ndim != 2 or old_target.shape != old_prior.shape:
            raise ValueError(f"invalid stored mask shapes for {sample.sample_id}")
        if old_prior.shape[0] != old_prior.shape[1]:
            raise ValueError(f"mask must be square for {sample.sample_id}")
        transform = Affine(*sample.crop_transform)
        prior_geometry = (
            shape(sample.prior_geometry.model_dump(mode="json"))
            if sample.prior_geometry is not None
            else None
        )
        target_geometry = (
            shape(sample.target_geometry.model_dump(mode="json"))
            if sample.target_geometry is not None
            else None
        )
        new_prior = rasterize_geometry_mask(prior_geometry, transform, old_prior.shape[0])
        new_target = rasterize_geometry_mask(target_geometry, transform, old_target.shape[0])
        prior_changed = not np.array_equal(old_prior, new_prior)
        target_changed = not np.array_equal(old_target, new_target)
        if prior_changed:
            changed_prior += 1
        if target_changed:
            changed_target += 1
        if prior_changed or target_changed:
            affected_ids.append(sample.sample_id)
            if write:
                np.save(prior_path, new_prior)
                np.save(target_path, new_target)

    summary: dict[str, Any] = {
        "source": str(samples_path.resolve()),
        "sample_count": len(samples),
        "changed_prior_masks": changed_prior,
        "changed_target_masks": changed_target,
        "changed_samples": len(affected_ids),
        "skipped_missing_provenance": skipped_missing_provenance,
        "write": write,
        "affected_sample_ids": affected_ids[:100],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
