#!/usr/bin/env python3
"""Stratify MUNO21 oracle writeback quality by executable edit operation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


OPERATIONS = ("KEEP", "ADD", "DELETE", "RESHAPE")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty list")
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap_mean(
    values: list[float], repetitions: int, rng: random.Random
) -> dict[str, float]:
    observed = mean(values)
    estimates = [
        mean(rng.choice(values) for _ in values) for _ in range(repetitions)
    ]
    return {
        "observed": observed,
        "ci95_low": _percentile(estimates, 0.025),
        "ci95_high": _percentile(estimates, 0.975),
    }


def _target_operation(row: dict[str, Any]) -> str:
    target = str(row["target"])
    if target == "REJECT":
        operation = "KEEP"
    elif ":" in target:
        operation = target.rsplit(":", 1)[-1]
    else:
        operation = str(row.get("operation", target))
    if operation not in OPERATIONS:
        raise ValueError(f"unsupported target operation: {target}")
    return operation


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"no writeback records: {path}")
    for row in rows:
        if row.get("split") != "val" or row.get("test_assets_read") is not False:
            raise ValueError("oracle failure analysis must be validation-only")
        _target_operation(row)
    return rows


def _summarize(
    operation: str,
    budget: float,
    rows: list[dict[str, Any]],
    repetitions: int,
    rng: random.Random,
) -> dict[str, Any]:
    gains = [float(row["raster_iou_gain"]) for row in rows]
    component_deltas = [
        float(row["component_count_absolute_error"])
        - float(row["prior_component_count_absolute_error"])
        for row in rows
    ]
    return {
        "operation": operation,
        "budget": budget,
        "sample_count": len(rows),
        "mean_raster_iou": mean(float(row["raster_iou"]) for row in rows),
        "mean_prior_raster_iou": mean(
            float(row["prior_raster_iou"]) for row in rows
        ),
        "mean_raster_iou_gain": mean(gains),
        "median_raster_iou_gain": median(gains),
        "positive_gain_rate": mean(value > 0.0 for value in gains),
        "negative_gain_rate": mean(value < 0.0 for value in gains),
        "zero_gain_rate": mean(value == 0.0 for value in gains),
        "raster_iou_gain_bootstrap": _bootstrap_mean(gains, repetitions, rng),
        "mean_added_change_iou": mean(
            float(row["added_change_iou"]) for row in rows
        ),
        "mean_removed_change_iou": mean(
            float(row["removed_change_iou"]) for row in rows
        ),
        "mean_component_count_error": mean(
            float(row["component_count_absolute_error"]) for row in rows
        ),
        "mean_prior_component_count_error": mean(
            float(row["prior_component_count_absolute_error"]) for row in rows
        ),
        "mean_component_error_delta": mean(component_deltas),
        "component_error_worsened_rate": mean(value > 0.0 for value in component_deltas),
        "false_edit_rate": mean(bool(row["false_edit"]) for row in rows),
        "missed_edit_rate": mean(bool(row["missed_edit"]) for row in rows),
        "wrong_edit_rate": mean(bool(row["wrong_edit"]) for row in rows),
        "topology_valid_rate": mean(
            bool(row["vector_delta_topology_valid"]) for row in rows
        ),
    }


def _write_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fields = (
        "operation",
        "budget",
        "sample_count",
        "mean_raster_iou",
        "mean_prior_raster_iou",
        "mean_raster_iou_gain",
        "gain_ci95_low",
        "gain_ci95_high",
        "positive_gain_rate",
        "negative_gain_rate",
        "mean_added_change_iou",
        "mean_removed_change_iou",
        "mean_component_count_error",
        "mean_prior_component_count_error",
        "mean_component_error_delta",
        "component_error_worsened_rate",
        "false_edit_rate",
        "missed_edit_rate",
        "wrong_edit_rate",
        "topology_valid_rate",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    **{field: summary.get(field) for field in fields},
                    "gain_ci95_low": summary["raster_iou_gain_bootstrap"]["ci95_low"],
                    "gain_ci95_high": summary["raster_iou_gain_bootstrap"]["ci95_high"],
                }
            )


def _write_markdown(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# MUNO21 Oracle Writeback Failure Slices",
        "",
        "| Operation | Budget | N | Map IoU gain | 95% CI | Positive | Negative | Component error delta | Worsened components |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        interval = row["raster_iou_gain_bootstrap"]
        lines.append(
            "| {operation} | {budget:.1f} | {sample_count} | {gain:.4f} | "
            "[{low:.4f}, {high:.4f}] | {positive:.3f} | {negative:.3f} | "
            "{component:.3f} | {worsened:.3f} |".format(
                operation=row["operation"],
                budget=row["budget"],
                sample_count=row["sample_count"],
                gain=row["mean_raster_iou_gain"],
                low=interval["ci95_low"],
                high=interval["ci95_high"],
                positive=row["positive_gain_rate"],
                negative=row["negative_gain_rate"],
                component=row["mean_component_error_delta"],
                worsened=row["component_error_worsened_rate"],
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("writeback_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--worst-example-count", type=int, default=20)
    args = parser.parse_args()

    rows = _read_rows(args.writeback_jsonl)
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(_target_operation(row), float(row["budget"]))].append(row)

    rng = random.Random(args.seed)
    summaries = [
        _summarize(operation, budget, grouped[(operation, budget)], args.repetitions, rng)
        for operation in OPERATIONS
        for budget in sorted({key[1] for key in grouped})
        if grouped.get((operation, budget))
    ]
    worst_examples = sorted(
        (
            {
                "task_id": row["task_id"],
                "aoi_id": row["aoi_id"],
                "budget": row["budget"],
                "operation": _target_operation(row),
                "raster_iou_gain": row["raster_iou_gain"],
                "raster_iou": row["raster_iou"],
                "prior_raster_iou": row["prior_raster_iou"],
                "component_count_error": row["component_count_absolute_error"],
                "prior_component_count_error": row[
                    "prior_component_count_absolute_error"
                ],
                "mask_artifact": row["mask_artifact"],
            }
            for row in rows
        ),
        key=lambda row: float(row["raster_iou_gain"]),
    )[: args.worst_example_count]

    args.output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(args.output_dir / "slices.csv", summaries)
    _write_markdown(args.output_dir / "slices.md", summaries)
    result = {
        "schema_version": "muno21-oracle-writeback-failure-slices-v1",
        "split": "val",
        "record_count": len(rows),
        "task_count": len({row["task_id"] for row in rows}),
        "budgets": sorted({float(row["budget"]) for row in rows}),
        "operations": list(OPERATIONS),
        "bootstrap_repetitions": args.repetitions,
        "bootstrap_seed": args.seed,
        "summaries": summaries,
        "worst_examples": worst_examples,
        "source": {
            "path": str(args.writeback_jsonl),
            "sha256": _sha256(args.writeback_jsonl),
        },
        "test_assets_read": False,
    }
    (args.output_dir / "slices.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
