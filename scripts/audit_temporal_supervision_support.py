"""Audit whether temporal updater samples contain pixel support for change heads."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from activemap.models import EditOperation
from activemap.updater_records import UpdaterSample, load_updater_samples


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pixel_counts(sample: UpdaterSample) -> tuple[int, int]:
    prior = np.load(sample.prior_mask_path) >= 0.5
    target = np.load(sample.target_mask_path) >= 0.5
    valid = (
        np.load(sample.valid_mask_path) >= 0.5
        if sample.valid_mask_path is not None
        else np.ones_like(target, dtype=bool)
    )
    if prior.shape != target.shape or target.shape != valid.shape:
        raise ValueError(f"mask shape mismatch for {sample.sample_id}")
    return (
        int(np.count_nonzero(target & ~prior & valid)),
        int(np.count_nonzero(prior & ~target & valid)),
    )


def _operation_has_expected_support(
    operation: EditOperation, added_pixels: int, removed_pixels: int
) -> bool:
    if operation == EditOperation.ADD:
        return added_pixels > 0
    if operation == EditOperation.DELETE:
        return removed_pixels > 0
    if operation == EditOperation.RESHAPE:
        return added_pixels > 0 and removed_pixels > 0
    return added_pixels == 0 and removed_pixels == 0


def audit_temporal_supervision_support(
    samples_path: Path,
    output_path: Path,
    *,
    allowed_splits: set[str],
) -> dict[str, Any]:
    samples = load_updater_samples(samples_path)
    unexpected = sorted({sample.split for sample in samples} - allowed_splits)
    if unexpected:
        raise ValueError(f"samples include disallowed splits: {unexpected}")

    rows: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    violations: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for sample in samples:
        added_pixels, removed_pixels = _pixel_counts(sample)
        key = (sample.split, sample.supervision_type, sample.edit_type.value)
        row = rows[key]
        row["samples"] += 1
        row["added_positive_samples"] += int(added_pixels > 0)
        row["removed_positive_samples"] += int(removed_pixels > 0)
        row["both_change_positive_samples"] += int(added_pixels > 0 and removed_pixels > 0)
        row["added_pixels"] += added_pixels
        row["removed_pixels"] += removed_pixels
        if not _operation_has_expected_support(sample.edit_type, added_pixels, removed_pixels):
            violations[key].append(sample.sample_id)

    report_rows = []
    for key in sorted(rows):
        split, supervision_type, operation = key
        row = rows[key]
        report_rows.append(
            {
                "split": split,
                "supervision_type": supervision_type,
                "operation": operation,
                **dict(sorted(row.items())),
                "expected_support_violations": len(violations[key]),
                "violation_sample_ids": violations[key][:20],
            }
        )
    report = {
        "status": "passed",
        "purpose": "temporal change-head supervision support audit",
        "samples": str(samples_path.resolve()),
        "samples_sha256": _sha256(samples_path),
        "allowed_splits": sorted(allowed_splits),
        "test_assets_read": False,
        "sample_count": len(samples),
        "rows": report_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--allowed-splits", default="train,val")
    args = parser.parse_args()
    allowed_splits = {value.strip() for value in args.allowed_splits.split(",") if value.strip()}
    if not allowed_splits:
        raise ValueError("--allowed-splits must not be empty")
    report = audit_temporal_supervision_support(
        args.samples,
        args.output,
        allowed_splits=allowed_splits,
    )
    print(json.dumps({"status": report["status"], "sample_count": report["sample_count"]}))


if __name__ == "__main__":
    main()
