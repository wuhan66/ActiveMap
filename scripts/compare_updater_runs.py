#!/usr/bin/env python3
"""Create a common convergence report for several updater experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from activemap.training.curves import render_run_comparison

BASE_METRICS = (
    "loss",
    "iou",
    "edit_accuracy",
    "false_edit_rate",
    "missed_edit_rate",
    "delete_recall",
    "loss_confidence",
)
TEMPORAL_METRICS = ("added_change_iou", "removed_change_iou")


def metric_value(record: dict[str, Any], metric: str) -> float | None:
    val = record["val"]
    if metric == "temporal_change_harmonic_iou":
        added = val.get("added_change_iou")
        removed = val.get("removed_change_iou")
        if added is None or removed is None:
            return None
        added = float(added)
        removed = float(removed)
        denominator = added + removed
        return 0.0 if denominator <= 0.0 else 2.0 * added * removed / denominator
    value = val.get(metric)
    return None if value is None else float(value)


def load_history(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def summarize_history(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("history is empty")
    best = min(records, key=lambda record: float(record["val"]["loss"]))
    metrics = {
        key: value
        for key in ("iou", "edit_accuracy", "false_edit_rate", "delete_recall")
        if (value := metric_value(best, key)) is not None
    }
    for key in (*TEMPORAL_METRICS, "temporal_change_harmonic_iou"):
        value = metric_value(best, key)
        if value is not None:
            metrics[key] = value
    summary = {
        "epochs": len(records),
        "best_val_loss_epoch": int(best["epoch"]),
        "best_val_loss": float(best["val"]["loss"]),
        "metrics_at_best_val_loss": metrics,
    }
    temporal_records = [
        (record, value)
        for record in records
        if (value := metric_value(record, "temporal_change_harmonic_iou")) is not None
    ]
    if temporal_records:
        best_temporal, best_temporal_value = max(temporal_records, key=lambda item: item[1])
        summary["best_temporal_change_harmonic_iou_epoch"] = int(
            best_temporal["epoch"]
        )
        summary["best_temporal_change_harmonic_iou"] = best_temporal_value
        summary["metrics_at_best_temporal_change_harmonic_iou"] = {
            key: value
            for key in (
                *TEMPORAL_METRICS,
                "iou",
                "edit_accuracy",
                "false_edit_rate",
                "missed_edit_rate",
                "delete_recall",
            )
            if (value := metric_value(best_temporal, key)) is not None
        }
    return summary


def paired_epoch_deltas(
    reference: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return candidate-minus-reference validation deltas at shared epochs."""

    reference_by_epoch = {int(record["epoch"]): record for record in reference}
    metrics = (*BASE_METRICS, *TEMPORAL_METRICS, "temporal_change_harmonic_iou")
    deltas = []
    for record in candidate:
        epoch = int(record["epoch"])
        baseline = reference_by_epoch.get(epoch)
        if baseline is None:
            continue
        metric_deltas = {}
        for metric in metrics:
            candidate_value = metric_value(record, metric)
            baseline_value = metric_value(baseline, metric)
            if candidate_value is not None and baseline_value is not None:
                metric_deltas[f"delta_val_{metric}"] = (
                    candidate_value - baseline_value
                )
        deltas.append({"epoch": epoch, **metric_deltas})
    return deltas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=HISTORY_JSONL",
        help="repeat for each run",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    runs: dict[str, list[dict[str, Any]]] = {}
    for item in args.run:
        name, separator, raw_path = item.partition("=")
        if not separator or not name or not raw_path:
            parser.error(f"invalid --run value: {item}")
        runs[name] = load_history(Path(raw_path))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        name: summarize_history(records) for name, records in runs.items()
    }
    reference_name = next(iter(runs))
    summary["paired_reference"] = reference_name
    summary["paired_epoch_deltas"] = {
        name: paired_epoch_deltas(runs[reference_name], records)
        for name, records in runs.items()
        if name != reference_name
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    render_run_comparison(runs, args.output_dir / "validation_comparison.png")


if __name__ == "__main__":
    main()
