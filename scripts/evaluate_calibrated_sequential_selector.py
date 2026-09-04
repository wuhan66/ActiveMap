#!/usr/bin/env python3
"""Evaluate a training-calibrated selector threshold without refitting it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.calibrate_sequential_selector_threshold import apply_threshold
from scripts.evaluate_sequential_selector import selector_metrics, task_bootstrap


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("likelihood_traces", type=Path)
    parser.add_argument("calibration_summary", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--expected-records", type=int)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    traces = [
        json.loads(line)
        for line in args.likelihood_traces.read_text(encoding="utf-8").splitlines()
    ]
    if args.expected_records is not None and len(traces) != args.expected_records:
        raise ValueError(f"expected {args.expected_records} rows, found {len(traces)}")
    calibration = json.loads(args.calibration_summary.read_text(encoding="utf-8"))
    if calibration.get("selection_partition") != "held-out-training-tasks":
        raise ValueError("threshold was not selected on held-out training tasks")
    if calibration.get("validation_assets_read") is not False:
        raise ValueError("calibration summary reports validation access")
    threshold = float(calibration["threshold"])
    evaluated = apply_threshold(traces, threshold)
    metrics = selector_metrics(evaluated)
    bootstrap = task_bootstrap(
        evaluated, repetitions=args.bootstrap_repetitions, seed=args.seed
    )
    prevalence = metrics["target_call_rate"]
    always_stop_macro_f1 = (2.0 * (1.0 - prevalence)) / (2.0 - prevalence) / 2.0
    promotion = {
        "nonzero_calls": metrics["predicted_calls"] > 0,
        "call_rate_at_most_0_50": metrics["predicted_call_rate"] <= 0.5,
        "acquire_recall_at_least_0_10": metrics["acquire_recall"] >= 0.1,
        "precision_at_least_prevalence": metrics["acquire_precision"] >= prevalence,
        "utility_positive": metrics["realized_utility_sum"] > 0.0,
        "macro_f1_above_always_stop": metrics["macro_f1"] > always_stop_macro_f1,
        "utility_bootstrap_ci_above_zero": bootstrap["intervals"][
            "realized_utility_mean"
        ]["ci95_low"]
        > 0.0,
    }
    args.output_dir.mkdir(parents=True)
    traces_path = args.output_dir / "traces.jsonl"
    with traces_path.open("w", encoding="utf-8") as handle:
        for row in evaluated:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {
        "schema_version": "sequential-selector-calibrated-evaluation-v1",
        "threshold": threshold,
        "metrics": metrics,
        "always_stop_macro_f1": always_stop_macro_f1,
        "task_bootstrap": bootstrap,
        "promotion_gate": {**promotion, "passed": all(promotion.values())},
        "sources": {
            "likelihood_traces": {
                "path": str(args.likelihood_traces.resolve()),
                "sha256": sha256(args.likelihood_traces),
            },
            "calibration_summary": {
                "path": str(args.calibration_summary.resolve()),
                "sha256": sha256(args.calibration_summary),
            },
            "trace_sha256": sha256(traces_path),
        },
        "test_assets_read": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
