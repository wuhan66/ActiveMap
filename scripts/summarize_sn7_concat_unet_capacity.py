#!/usr/bin/env python3
"""Summarize the controlled SN7 concat U-Net width ablation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


METRICS = (
    "macro_f1",
    "edit_accuracy",
    "false_edit_rate",
    "missed_update_rate",
    "mean_raster_iou",
    "mean_polygon_iou",
    "topology_valid_rate",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_spec(value: str) -> tuple[int, int, Path, Path]:
    parts = value.split("=", 1)
    if len(parts) != 2:
        raise ValueError("--run must use WIDTH,PARAMETERS=SUMMARY")
    metadata, raw_path = parts
    width, parameters = (int(item) for item in metadata.split(",", 1))
    return width, parameters, Path(raw_path), Path(raw_path).parent


def summarize(specs: list[tuple[int, int, Path, Path]]) -> dict[str, Any]:
    if len(specs) != 3 or {item[0] for item in specs} != {32, 48, 64}:
        raise ValueError("exactly width 32, 48, and 64 runs are required")
    rows: list[dict[str, Any]] = []
    support: tuple[int, int] | None = None
    for width, parameters, summary_path, _ in sorted(specs):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if payload.get("split") != "val":
            raise ValueError(f"validation-only result required: {summary_path}")
        current_support = (int(payload["sample_count"]), int(payload["aoi_count"]))
        support = current_support if support is None else support
        if current_support != support:
            raise ValueError("capacity runs do not share validation support")
        row: dict[str, Any] = {
            "base_channels": width,
            "parameter_count": parameters,
            "summary": str(summary_path.resolve()),
            "sha256": _sha256(summary_path),
        }
        for metric in METRICS:
            row[metric] = float(payload[metric])
        rows.append(row)

    baseline = next(row for row in rows if row["base_channels"] == 32)
    for row in rows:
        for metric in METRICS:
            row[f"delta_{metric}_vs_width32"] = row[metric] - baseline[metric]
    candidates = [row for row in rows if row["base_channels"] != 32]
    for row in candidates:
        gates = {
            "raster_iou_gain_at_least_0_003": (
                row["delta_mean_raster_iou_vs_width32"] >= 0.003
            ),
            "false_edit_regression_at_most_0_005": (
                row["delta_false_edit_rate_vs_width32"] <= 0.005
            ),
            "operation_accuracy_regression_at_most_0_005": (
                row["delta_edit_accuracy_vs_width32"] >= -0.005
            ),
        }
        row["pilot_promotion_gates"] = gates
        row["pilot_promotion_passed"] = all(gates.values())
    passing = [row for row in candidates if row["pilot_promotion_passed"]]
    selected = (
        max(passing, key=lambda row: row["mean_raster_iou"]) if passing else baseline
    )
    return {
        "schema_version": "sn7-concat-unet-capacity-ablation-v1",
        "split": "val",
        "test_assets_read": False,
        "controlled_seed": 20260716,
        "sample_count": support[0] if support else None,
        "aoi_count": support[1] if support else None,
        "fixed_factors": [
            "data split",
            "seed",
            "optimization",
            "loss",
            "augmentation",
            "early stopping",
            "edit decoding",
        ],
        "varied_factor": "base_channels",
        "promotion_protocol": {
            "status": "predeclared_before_validation_completion",
            "minimum_raster_iou_gain": 0.003,
            "maximum_false_edit_regression": 0.005,
            "maximum_operation_accuracy_regression": 0.005,
            "single_seed_scope": (
                "A pass only triggers a three-seed confirmation; it cannot "
                "replace the formal width-32 baseline by itself."
            ),
        },
        "selected_width": int(selected["base_channels"]),
        "three_seed_confirmation_required": bool(passing),
        "runs": rows,
    }


def _write_csv(path: Path, payload: dict[str, Any]) -> None:
    fields = [
        "base_channels",
        "parameter_count",
        *METRICS,
        *(f"delta_{metric}_vs_width32" for metric in METRICS),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in payload["runs"]:
            writer.writerow({field: row[field] for field in fields})


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# SN7 Concat U-Net Capacity Ablation",
        "",
        "| Base channels | Parameters | Macro-F1 | Raster IoU | False edit | Missed edit |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["runs"]:
        lines.append(
            f"| {row['base_channels']} | {row['parameter_count']:,} | "
            f"{row['macro_f1']:.6f} | {row['mean_raster_iou']:.6f} | "
            f"{row['false_edit_rate']:.6f} | {row['missed_update_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"Pilot selection: width {payload['selected_width']}.",
            (
                "A wider passing model requires three-seed confirmation."
                if payload["three_seed_confirmation_required"]
                else "No wider model passed the predeclared promotion gate."
            ),
            "",
            "Controlled single-seed validation diagnostic; test assets were not read.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--run", action="append", required=True)
    args = parser.parse_args()
    specs = [_parse_spec(value) for value in args.run]
    payload = summarize(specs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(args.output_dir / "capacity_ablation.csv", payload)
    _write_markdown(args.output_dir / "capacity_ablation.md", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
