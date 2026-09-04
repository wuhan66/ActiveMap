#!/usr/bin/env python3
"""Aggregate fixed-protocol SN7 concat U-Net validation results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


METRICS = (
    "edit_accuracy",
    "macro_f1",
    "update_f1",
    "false_edit_rate",
    "missed_update_rate",
    "mean_raster_iou",
    "mean_polygon_iou",
    "topology_valid_rate",
    "ece",
)
EDIT_CLASSES = ("KEEP", "ADD", "DELETE", "RESHAPE")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate(inputs: list[tuple[int, Path]]) -> dict[str, Any]:
    if len(inputs) != 3 or len({seed for seed, _ in inputs}) != 3:
        raise ValueError("exactly three unique seeds are required")
    rows = []
    sample_count = None
    aoi_count = None
    for seed, path in sorted(inputs):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("split") != "val":
            raise ValueError(f"validation-only summary required: {path}")
        if "/test" in str(payload.get("prediction_path", "")).lower():
            raise ValueError(f"test-like prediction path rejected: {path}")
        current_samples = int(payload["sample_count"])
        current_aois = int(payload["aoi_count"])
        sample_count = current_samples if sample_count is None else sample_count
        aoi_count = current_aois if aoi_count is None else aoi_count
        if current_samples != sample_count or current_aois != aoi_count:
            raise ValueError("seed summaries have different validation support")
        row: dict[str, Any] = {
            "seed": seed,
            "source": str(path.resolve()),
            "sha256": _sha256(path),
            "checkpoint": payload["checkpoint"],
        }
        for metric in METRICS:
            row[metric] = float(payload[metric])
        for edit in EDIT_CLASSES:
            row[f"{edit.lower()}_f1"] = float(payload["per_edit"][edit]["f1"])
        rows.append(row)

    metric_names = (*METRICS, *(f"{edit.lower()}_f1" for edit in EDIT_CLASSES))
    aggregate_metrics = {}
    for metric in metric_names:
        values = [float(row[metric]) for row in rows]
        aggregate_metrics[metric] = {
            "mean": mean(values),
            "sample_std": stdev(values),
            "per_seed": {str(row["seed"]): float(row[metric]) for row in rows},
        }
    return {
        "schema_version": "sn7-concat-unet-three-seed-validation-v1",
        "split": "val",
        "test_assets_read": False,
        "seed_count": 3,
        "seeds": [row["seed"] for row in rows],
        "sample_count_per_seed": sample_count,
        "aoi_count": aoi_count,
        "protocol": {
            "checkpoint_selection": "best_quality",
            "thresholds": {
                "commit": 0.0,
                "change": 0.5,
                "add": 0.5,
                "remove": 0.5,
            },
            "edit_decoding": "auto",
            "bootstrap_draws_per_seed": 5000,
        },
        "per_seed": rows,
        "aggregate": aggregate_metrics,
    }


def _write_csv(path: Path, payload: dict[str, Any]) -> None:
    fields = ["seed", *METRICS, *(f"{edit.lower()}_f1" for edit in EDIT_CLASSES)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in payload["per_seed"]:
            writer.writerow({field: row[field] for field in fields})


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    metrics = (
        ("macro_f1", "Macro-F1"),
        ("edit_accuracy", "Edit Acc."),
        ("false_edit_rate", "False Edit"),
        ("missed_update_rate", "Missed Edit"),
        ("mean_raster_iou", "Raster IoU"),
        ("mean_polygon_iou", "Polygon IoU"),
    )
    lines = [
        "# SN7 Concat U-Net Three-Seed Validation",
        "",
        "| Metric | Mean | Seed std |",
        "| --- | ---: | ---: |",
    ]
    for key, label in metrics:
        item = payload["aggregate"][key]
        lines.append(f"| {label} | {item['mean']:.6f} | {item['sample_std']:.6f} |")
    lines.extend(["", "Validation only; test assets were not read.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--summary", action="append", required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and (args.output_dir / "aggregate.json").exists():
        raise FileExistsError(args.output_dir / "aggregate.json")
    inputs = []
    for value in args.summary:
        raw_seed, separator, raw_path = value.partition("=")
        if not separator:
            parser.error("--summary must use SEED=PATH")
        inputs.append((int(raw_seed), Path(raw_path)))
    payload = aggregate(inputs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "aggregate.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(args.output_dir / "per_seed.csv", payload)
    _write_markdown(args.output_dir / "summary.md", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
