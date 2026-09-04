#!/usr/bin/env python3
"""Evaluate validation-derived operation-conditioned delta regularization."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.sweep_muno21_oracle_delta_components import (
    OPERATIONS,
    _paired_bootstrap,
    _replay,
    _sha256,
)


def _rates(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "mean_raster_iou": mean(row["raster_iou"] for row in rows),
        "mean_raster_iou_gain": mean(row["raster_iou_gain"] for row in rows),
        "false_edit_rate": mean(row["false_edit"] for row in rows),
        "missed_edit_rate": mean(row["missed_edit"] for row in rows),
        "wrong_edit_rate": mean(row["wrong_edit"] for row in rows),
        "mean_component_count_error": mean(
            abs(row["component_count"] - row["target_component_count"])
            for row in rows
        ),
    }


def _comparison(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    repetitions: int,
    rng: random.Random,
) -> dict[str, Any]:
    baseline_by_id = {row["task_id"]: row for row in baseline}
    candidate_by_id = {row["task_id"]: row for row in candidate}
    if baseline_by_id.keys() != candidate_by_id.keys():
        raise ValueError("baseline and candidate task IDs differ")
    task_ids = sorted(baseline_by_id)
    deltas = [
        candidate_by_id[task_id]["raster_iou"]
        - baseline_by_id[task_id]["raster_iou"]
        for task_id in task_ids
    ]
    baseline_rates = _rates(baseline)
    candidate_rates = _rates(candidate)
    interval = _paired_bootstrap(deltas, repetitions, rng)
    return {
        "sample_count": len(task_ids),
        "baseline": baseline_rates,
        "candidate": candidate_rates,
        "paired_raster_iou_delta": interval,
        "false_edit_rate_delta": (
            candidate_rates["false_edit_rate"] - baseline_rates["false_edit_rate"]
        ),
        "missed_edit_rate_delta": (
            candidate_rates["missed_edit_rate"] - baseline_rates["missed_edit_rate"]
        ),
        "wrong_edit_rate_delta": (
            candidate_rates["wrong_edit_rate"] - baseline_rates["wrong_edit_rate"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("writeback_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--add-threshold", type=int, default=1536)
    parser.add_argument("--delete-threshold", type=int, default=1024)
    parser.add_argument("--reshape-threshold", type=int, default=0)
    parser.add_argument("--selection-budget", type=float, default=3.0)
    parser.add_argument("--repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    thresholds = {
        "KEEP": 0,
        "ADD": args.add_threshold,
        "DELETE": args.delete_threshold,
        "RESHAPE": args.reshape_threshold,
    }
    if any(value < 0 for value in thresholds.values()):
        raise ValueError("component thresholds must be nonnegative")

    source_rows = [
        json.loads(line)
        for line in args.writeback_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not source_rows:
        raise ValueError("empty writeback input")
    for row in source_rows:
        if row.get("split") != "val" or row.get("test_assets_read") is not False:
            raise ValueError("operation-conditioned replay must remain validation-only")
        if not Path(row["mask_artifact"]).is_file():
            raise FileNotFoundError(row["mask_artifact"])

    selected_rows = [
        row for row in source_rows if float(row["budget"]) == args.selection_budget
    ]
    if not selected_rows:
        raise ValueError(f"no rows for selection budget {args.selection_budget}")

    baseline = [
        _replay(row, 0, "all", "preserve_largest") for row in selected_rows
    ]
    candidate = [
        _replay(
            row,
            thresholds[
                "KEEP"
                if row["target"] == "REJECT"
                else str(row["target"]).rsplit(":", 1)[-1]
            ],
            "all",
            "preserve_largest",
        )
        for row in selected_rows
    ]

    rng = random.Random(args.seed)
    overall = _comparison(baseline, candidate, args.repetitions, rng)
    by_operation: dict[str, dict[str, Any]] = {}
    for operation in OPERATIONS:
        operation_baseline = [
            row for row in baseline if row["operation"] == operation
        ]
        operation_candidate = [
            row for row in candidate if row["operation"] == operation
        ]
        if operation_baseline:
            by_operation[operation] = _comparison(
                operation_baseline,
                operation_candidate,
                args.repetitions,
                rng,
            )

    interval = overall["paired_raster_iou_delta"]
    promoted = (
        interval["ci95_low"] > 0
        and overall["false_edit_rate_delta"] <= 0
        and overall["missed_edit_rate_delta"] <= 0
    )
    result = {
        "schema_version": "muno21-operation-conditioned-delta-v1",
        "split": "val",
        "selection_status": "validation-derived-post-hoc-candidate",
        "selection_budget": args.selection_budget,
        "thresholds": thresholds,
        "pruning_policy": "preserve_largest",
        "bootstrap_unit": "task",
        "bootstrap_repetitions": args.repetitions,
        "overall": overall,
        "by_operation": by_operation,
        "promotion_gate_passed": promoted,
        "source": {
            "path": str(args.writeback_jsonl),
            "sha256": _sha256(args.writeback_jsonl),
        },
        "test_assets_read": False,
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    fields = (
        "operation",
        "threshold",
        "sample_count",
        "baseline_mean_gain",
        "candidate_mean_gain",
        "paired_delta",
        "ci95_low",
        "ci95_high",
        "false_edit_delta",
        "missed_edit_delta",
        "wrong_edit_delta",
    )
    with (args.output_dir / "table.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for operation, comparison in (("ALL", overall), *by_operation.items()):
            paired = comparison["paired_raster_iou_delta"]
            writer.writerow(
                {
                    "operation": operation,
                    "threshold": (
                        "mixed" if operation == "ALL" else thresholds[operation]
                    ),
                    "sample_count": comparison["sample_count"],
                    "baseline_mean_gain": comparison["baseline"][
                        "mean_raster_iou_gain"
                    ],
                    "candidate_mean_gain": comparison["candidate"][
                        "mean_raster_iou_gain"
                    ],
                    "paired_delta": paired["observed"],
                    "ci95_low": paired["ci95_low"],
                    "ci95_high": paired["ci95_high"],
                    "false_edit_delta": comparison["false_edit_rate_delta"],
                    "missed_edit_delta": comparison["missed_edit_rate_delta"],
                    "wrong_edit_delta": comparison["wrong_edit_rate_delta"],
                }
            )

    lines = [
        "# MUNO21 Operation-Conditioned Safe Delta",
        "",
        "Validation-derived post-hoc candidate; frozen test assets were not read.",
        "",
        "| Operation | Min pixels | N | Baseline gain | Candidate gain | Paired delta (95% CI) | False-edit delta | Missed-edit delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for operation, comparison in (("ALL", overall), *by_operation.items()):
        paired = comparison["paired_raster_iou_delta"]
        threshold = "mixed" if operation == "ALL" else str(thresholds[operation])
        lines.append(
            "| {operation} | {threshold} | {count} | {baseline:.6f} | "
            "{candidate:.6f} | {delta:+.6f} [{low:+.6f}, {high:+.6f}] | "
            "{false:+.6f} | {missed:+.6f} |".format(
                operation=operation,
                threshold=threshold,
                count=comparison["sample_count"],
                baseline=comparison["baseline"]["mean_raster_iou_gain"],
                candidate=comparison["candidate"]["mean_raster_iou_gain"],
                delta=paired["observed"],
                low=paired["ci95_low"],
                high=paired["ci95_high"],
                false=comparison["false_edit_rate_delta"],
                missed=comparison["missed_edit_rate_delta"],
            )
        )
    lines.extend(
        [
            "",
            f"Promotion gate: `{'PASS' if promoted else 'FAIL'}`.",
            "",
        ]
    )
    (args.output_dir / "table.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
