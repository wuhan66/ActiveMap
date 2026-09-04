#!/usr/bin/env python3
"""Audit MUNO21 prior/target alignment before prior-conditioned SAM training."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from activemap.integrations.sam_road_training import (
    load_road_records,
    union_segmentations,
)
from activemap.updater_records import load_updater_samples


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binary_mask(path: str, expected_shape: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(np.load(path)).squeeze()
    if mask.shape != expected_shape:
        raise ValueError(f"mask {path} has shape {mask.shape}, expected {expected_shape}")
    if not np.all(np.isfinite(mask)):
        raise ValueError(f"mask {path} contains non-finite values")
    return mask > 0.5


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.logical_and(first, second).sum())
    union = int(np.logical_or(first, second).sum())
    return intersection / max(union, 1)


def audit(
    dataset_root: Path,
    updater_manifest: Path,
    *,
    splits: tuple[str, ...] = ("train", "val"),
) -> dict[str, Any]:
    samples = {row.sample_id: row for row in load_updater_samples(updater_manifest)}
    if len(samples) != len(load_updater_samples(updater_manifest)):
        raise ValueError("updater manifest contains duplicate sample IDs")
    summaries = {}
    errors = []
    for split in splits:
        records = load_road_records(
            dataset_root / "annotations" / f"{split}.json",
            dataset_root / "images" / split,
        )
        counts: Counter[str] = Counter()
        fractions: dict[str, list[float]] = defaultdict(list)
        target_ious = []
        matched_ids = set()
        for record in tqdm(records, desc=f"audit {split}", unit="image"):
            sample = samples.get(record.sample_id)
            if sample is None:
                errors.append(f"missing updater sample: {record.sample_id}")
                continue
            matched_ids.add(record.sample_id)
            if sample.split != split:
                errors.append(
                    f"split mismatch {record.sample_id}: {sample.split} != {split}"
                )
                continue
            expected_shape = (record.height, record.width)
            try:
                prior = _binary_mask(sample.prior_mask_path, expected_shape)
                updater_target = _binary_mask(sample.target_mask_path, expected_shape)
                valid = (
                    _binary_mask(sample.valid_mask_path, expected_shape)
                    if sample.valid_mask_path is not None
                    else np.ones(expected_shape, dtype=bool)
                )
                rle_target = union_segmentations(
                    record.segmentations,
                    height=record.height,
                    width=record.width,
                ).astype(bool)
            except Exception as exc:
                errors.append(f"{record.sample_id}: {exc}")
                continue
            prior &= valid
            updater_target &= valid
            rle_target &= valid
            add = np.logical_and(rle_target, ~prior)
            remove = np.logical_and(prior, ~rle_target)
            target_ious.append(_iou(rle_target, updater_target))
            counts[f"edit:{record.edit_type}"] += 1
            counts["matched"] += 1
            counts["empty_prior"] += int(not prior.any())
            counts["empty_target"] += int(not rle_target.any())
            valid_count = max(int(valid.sum()), 1)
            fractions["prior"].append(float(prior.sum() / valid_count))
            fractions["target"].append(float(rle_target.sum() / valid_count))
            fractions["add"].append(float(add.sum() / valid_count))
            fractions["remove"].append(float(remove.sum() / valid_count))
            fractions[f"add:{record.edit_type}"].append(float(add.sum() / valid_count))
            fractions[f"remove:{record.edit_type}"].append(
                float(remove.sum() / valid_count)
            )
        summaries[split] = {
            "records": len(records),
            "matched_unique_ids": len(matched_ids),
            "counts": dict(sorted(counts.items())),
            "rle_vs_updater_target_iou": {
                "minimum": min(target_ious) if target_ious else None,
                "mean": float(np.mean(target_ious)) if target_ious else None,
                "below_0_99": sum(value < 0.99 for value in target_ious),
            },
            "fractions": {
                name: {
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "maximum": max(values),
                    "positive_count": sum(value > 0 for value in values),
                }
                for name, values in sorted(fractions.items())
            },
        }
    return {
        "schema_version": "muno21-prior-conditioning-audit-v1",
        "dataset_root": str(dataset_root.resolve()),
        "updater_manifest": str(updater_manifest.resolve()),
        "sources": {
            "updater_manifest_sha256": _sha256(updater_manifest),
            **{
                f"{split}_annotations_sha256": _sha256(
                    dataset_root / "annotations" / f"{split}.json"
                )
                for split in splits
            },
        },
        "splits": summaries,
        "errors": errors,
        "passed": not errors
        and all(
            summary["records"] == summary["matched_unique_ids"]
            for summary in summaries.values()
        ),
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("updater_manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = audit(args.dataset_root, args.updater_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
