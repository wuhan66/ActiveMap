#!/usr/bin/env python3
"""Aggregate protocol-matched ChangeMamba safe-commit validations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

METRICS = (
    "committed_map_iou",
    "map_iou_delta",
    "operation_accuracy",
    "false_edit_rate",
    "missed_edit_rate",
    "wrong_edit_rate",
    "accepted_commit_rate",
    "candidate_acceptance_rate",
    "beneficial_commit_precision",
    "beneficial_commit_recall",
)


def aggregate_safe_commit(run_dirs: list[Path]) -> dict[str, Any]:
    if len(run_dirs) < 2:
        raise ValueError("at least two safe-commit runs are required")
    runs = []
    protocol = None
    for run_dir in run_dirs:
        summary = json.loads(
            (run_dir / "summary.json").read_text(encoding="utf-8")
        )
        if summary.get("test_assets_read") is not False:
            raise ValueError(f"safe-commit run is not test-free: {run_dir}")
        calibration = summary["calibration"]
        current_protocol = {
            "schema_version": summary["schema_version"],
            "feature_names": summary["feature_names"],
            "method": calibration["method"],
            "folds": calibration["folds"],
            "l2": calibration["l2"],
            "beneficial_definition": calibration["beneficial_definition"],
            "threshold_selection": calibration["threshold_selection"],
        }
        if protocol is None:
            protocol = current_protocol
        elif current_protocol != protocol:
            raise ValueError(f"safe-commit protocol mismatch: {run_dir}")
        validation = summary["validation"]
        runs.append(
            {
                "run_dir": str(run_dir),
                "selected_threshold": calibration["selected_threshold"],
                "always_commit": validation["always_commit"],
                "safe_commit": validation["safe_commit"],
                "safe_minus_always": validation["safe_minus_always"],
            }
        )

    aggregate = {}
    for section in ("always_commit", "safe_commit", "safe_minus_always"):
        aggregate[section] = {}
        for metric in METRICS:
            values = np.asarray(
                [run[section][metric] for run in runs], dtype=np.float64
            )
            aggregate[section][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
            }
    thresholds = np.asarray(
        [run["selected_threshold"] for run in runs], dtype=np.float64
    )
    return {
        "schema_version": "sn7-changemamba-safe-commit-aggregate-v1",
        "run_count": len(runs),
        "protocol": protocol,
        "selected_threshold": {
            "mean": float(thresholds.mean()),
            "std": float(thresholds.std(ddof=1)),
        },
        "runs": runs,
        "aggregate": aggregate,
        "test_assets_read": False,
    }


def markdown(result: dict[str, Any]) -> str:
    lines = [
        "| Metric | Always commit | Safe commit | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric in METRICS:
        always = result["aggregate"]["always_commit"][metric]
        safe = result["aggregate"]["safe_commit"][metric]
        delta = result["aggregate"]["safe_minus_always"][metric]
        lines.append(
            f"| {metric} | {always['mean']:.6f} +/- {always['std']:.6f} | "
            f"{safe['mean']:.6f} +/- {safe['std']:.6f} | "
            f"{delta['mean']:+.6f} +/- {delta['std']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_json", type=Path)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--output-markdown", type=Path)
    args = parser.parse_args()
    result = aggregate_safe_commit(args.run_dirs)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    if args.output_markdown is not None:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
