#!/usr/bin/env python3
"""Freeze a selector decision threshold on held-out training tasks only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from scripts.evaluate_sequential_selector import selector_metrics, task_bootstrap


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_traces(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise ValueError("selector likelihood traces are empty")
    required = {
        "task_id",
        "target_selection",
        "policy_relative_advantage",
        "decision_token_log_probability_margin",
    }
    if any(required - row.keys() for row in rows):
        raise ValueError("likelihood traces miss calibration fields")
    if {row.get("split") for row in rows} != {"train"}:
        raise ValueError("threshold calibration may only read training rows")
    return rows


def apply_threshold(rows: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "predicted_selection": (
                "ACQUIRE"
                if float(row["decision_token_log_probability_margin"]) > threshold
                else "STOP"
            ),
        }
        for row in rows
    ]


def threshold_grid(rows: list[dict[str, Any]]) -> list[float]:
    values = sorted(
        {float(row["decision_token_log_probability_margin"]) for row in rows}
    )
    if len(values) < 2:
        raise ValueError("calibration margins are constant")
    return [
        math.nextafter(values[0], -math.inf),
        *[(left + right) / 2.0 for left, right in zip(values, values[1:], strict=False)],
        math.nextafter(values[-1], math.inf),
    ]


def select_threshold(
    rows: list[dict[str, Any]],
    *,
    max_call_rate: float,
    max_false_call_rate: float,
    min_recall: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prevalence = sum(row["target_selection"] == "ACQUIRE" for row in rows) / len(rows)
    grid = []
    for threshold in threshold_grid(rows):
        metrics = selector_metrics(apply_threshold(rows, threshold))
        feasible = (
            metrics["predicted_calls"] > 0
            and metrics["predicted_call_rate"] <= max_call_rate
            and metrics["false_call_rate"] <= max_false_call_rate
            and metrics["acquire_recall"] >= min_recall
            and metrics["acquire_precision"] >= prevalence
            and metrics["realized_utility_sum"] > 0.0
        )
        grid.append({"threshold": threshold, "feasible": feasible, "metrics": metrics})
    feasible_rows = [row for row in grid if row["feasible"]]
    if not feasible_rows:
        raise ValueError("no train-calibration threshold satisfies safety constraints")
    selected = max(
        feasible_rows,
        key=lambda row: (
            row["metrics"]["realized_utility_sum"],
            -row["metrics"]["realized_risk_sum"],
            row["metrics"]["macro_f1"],
            -row["metrics"]["predicted_calls"],
        ),
    )
    return selected, grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("likelihood_traces", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--max-call-rate", type=float, default=0.5)
    parser.add_argument("--max-false-call-rate", type=float, default=0.1)
    parser.add_argument("--min-recall", type=float, default=0.1)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    rows = load_traces(args.likelihood_traces)
    selected, grid = select_threshold(
        rows,
        max_call_rate=args.max_call_rate,
        max_false_call_rate=args.max_false_call_rate,
        min_recall=args.min_recall,
    )
    selected_rows = apply_threshold(rows, float(selected["threshold"]))
    bootstrap = task_bootstrap(
        selected_rows, repetitions=args.bootstrap_repetitions, seed=args.seed
    )
    utility_interval = bootstrap["intervals"]["realized_utility_mean"]
    calibration_gate = {
        "positive_point_utility": selected["metrics"]["realized_utility_sum"] > 0.0,
        "bootstrap_probability_gt_zero_at_least_0_8": utility_interval[
            "bootstrap_probability_gt_zero"
        ]
        >= 0.8,
    }
    args.output_dir.mkdir(parents=True)
    grid_path = args.output_dir / "threshold_grid.json"
    grid_path.write_text(json.dumps(grid, indent=2) + "\n", encoding="utf-8")
    summary = {
        "schema_version": "sequential-selector-train-calibration-v1",
        "selection_partition": "held-out-training-tasks",
        "threshold": selected["threshold"],
        "metrics": selected["metrics"],
        "task_bootstrap": bootstrap,
        "constraints": {
            "max_call_rate": args.max_call_rate,
            "max_false_call_rate": args.max_false_call_rate,
            "min_recall": args.min_recall,
        },
        "calibration_gate": {
            **calibration_gate,
            "passed": all(calibration_gate.values()),
        },
        "sources": {
            "likelihood_traces": {
                "path": str(args.likelihood_traces.resolve()),
                "sha256": sha256(args.likelihood_traces),
            },
            "threshold_grid_sha256": sha256(grid_path),
        },
        "validation_assets_read": False,
        "test_assets_read": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
