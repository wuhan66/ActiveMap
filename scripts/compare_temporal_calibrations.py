#!/usr/bin/env python3
"""Compare updater temporal calibrations produced under one frozen protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compare(runs: list[tuple[str, Path]]) -> dict[str, Any]:
    if len(runs) < 2:
        raise ValueError("at least two calibrations are required")
    loaded = [(name, path, _load(path)) for name, path in runs]
    reference = loaded[0][2]
    protocol_fields = {
        "split": reference.get("split"),
        "samples": reference.get("samples"),
        "sample_count": reference.get("sample_count"),
        "max_stable_false_positive": reference.get("max_stable_false_positive"),
        "grid_steps": len(reference.get("add_sweep", [])),
    }
    if protocol_fields["split"] != "val":
        raise ValueError("calibrations must use the validation split")
    rows: dict[str, Any] = {}
    for name, path, report in loaded:
        candidate_fields = {
            "split": report.get("split"),
            "samples": report.get("samples"),
            "sample_count": report.get("sample_count"),
            "max_stable_false_positive": report.get("max_stable_false_positive"),
            "grid_steps": len(report.get("add_sweep", [])),
        }
        if candidate_fields != protocol_fields:
            raise ValueError(f"{name}: calibration protocol differs from reference")
        selected = report["selected"]
        rows[name] = {
            "calibration": str(path.resolve()),
            "checkpoint": report.get("checkpoint"),
            "checkpoint_epoch": int(report["checkpoint_epoch"]),
            "constraint_satisfied": bool(report["constraint_satisfied"]),
            "add_threshold": float(selected["add_threshold"]),
            "remove_threshold": float(selected["remove_threshold"]),
            "add_iou": float(selected["add"]["mean_positive_iou"]),
            "remove_iou": float(selected["remove"]["mean_positive_iou"]),
            "harmonic_iou": float(selected["harmonic_mean_positive_iou"]),
            "add_stable_false_positive": float(
                selected["add"]["stable_false_positive_fraction"]
            ),
            "remove_stable_false_positive": float(
                selected["remove"]["stable_false_positive_fraction"]
            ),
        }
    reference_name = loaded[0][0]
    reference_row = rows[reference_name]
    deltas = {
        name: {
            f"delta_{metric}": values[metric] - reference_row[metric]
            for metric in (
                "add_iou",
                "remove_iou",
                "harmonic_iou",
                "add_stable_false_positive",
                "remove_stable_false_positive",
            )
        }
        for name, values in rows.items()
        if name != reference_name
    }
    return {
        "protocol": "paired_validation_temporal_calibration_v1",
        "protocol_fields": protocol_fields,
        "reference": reference_name,
        "runs": rows,
        "deltas": deltas,
        "all_constraints_satisfied": all(
            row["constraint_satisfied"] for row in rows.values()
        ),
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=CALIBRATION_JSON",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = []
    for item in args.run:
        name, separator, raw_path = item.partition("=")
        if not separator or not name or not raw_path:
            parser.error(f"invalid --run value: {item}")
        runs.append((name, Path(raw_path)))
    report = compare(runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
