#!/usr/bin/env python3
"""Aggregate the completed MUNO21 DPO action and recurrent rollout grid."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def average(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def three_budget_auc(rows: list[dict[str, Any]], key: str) -> float:
    ordered = sorted(rows, key=lambda row: float(row["budget"]))
    if len(ordered) != 3:
        raise ValueError("expected exactly three budget rows")
    values = [float(row[key]) for row in ordered]
    return 0.25 * values[0] + 0.5 * values[1] + 0.25 * values[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    per_run: list[dict[str, Any]] = []
    pattern = re.compile(r"b(?P<beta>\d{3})s(?P<seed>\d+)$")
    for run_dir in sorted(path for path in args.root.iterdir() if path.is_dir()):
        match = pattern.fullmatch(run_dir.name)
        if match is None:
            continue
        action = read_json(run_dir / "actions/summary.json")
        rollout = read_json(run_dir / "rollouts/summary.json")
        method_rows = [
            row for row in rollout["results"] if row["method"] == "qwen3_4b_sft"
        ]
        selector_rows = [
            row
            for row in rollout["results"]
            if row["method"] == "edit_conditioned_selector"
        ]
        validity = rollout["llm_validity_by_method"]["qwen3_4b_sft"]
        utility_auc = three_budget_auc(method_rows, "mean_quality_cost_utility")
        selector_auc = three_budget_auc(selector_rows, "mean_quality_cost_utility")
        per_run.append(
            {
                "beta": int(match.group("beta")) / 1000.0,
                "seed": int(match.group("seed")),
                "exact_action_accuracy": action["exact_action_accuracy"],
                "macro_f1": action["macro_f1"],
                "policy_utility": action["mean_policy_utility"],
                "regret": action["mean_regret"],
                "executable_rate": validity["executable_valid_rate"],
                "fallback_rate": validity["fallback_rate"],
                "terminal_accuracy": average(method_rows, "terminal_accuracy"),
                "false_edit_rate": average(method_rows, "false_edit_rate"),
                "missed_edit_rate": average(method_rows, "missed_edit_rate"),
                "mean_cost": average(method_rows, "mean_cost"),
                "mean_acquisitions": average(method_rows, "mean_acquisitions"),
                "utility_auc": utility_auc,
                "delta_vs_selector": utility_auc - selector_auc,
                "joint_utility": average(method_rows, "mean_joint_utility"),
                "selector_joint_utility": average(selector_rows, "mean_joint_utility"),
                "test_assets_read": rollout["protocol"]["test_assets_read"],
            }
        )

    fields = list(per_run[0])
    with (args.output / "per_run.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_run)

    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in per_run:
        grouped[float(row["beta"])].append(row)
    metrics = [
        "macro_f1",
        "terminal_accuracy",
        "false_edit_rate",
        "missed_edit_rate",
        "mean_cost",
        "utility_auc",
        "delta_vs_selector",
    ]
    aggregate: list[dict[str, Any]] = []
    for beta, rows in sorted(grouped.items()):
        record: dict[str, Any] = {"beta": beta, "seeds": len(rows)}
        for metric in metrics:
            values = [float(row[metric]) for row in rows]
            record[f"{metric}_mean"] = statistics.fmean(values)
            record[f"{metric}_std"] = statistics.pstdev(values)
        aggregate.append(record)
    aggregate_fields = list(aggregate[0])
    with (args.output / "aggregate.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregate)
    (args.output / "summary.json").write_text(
        json.dumps({"per_run": per_run, "aggregate": aggregate}, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "| Beta | Seeds | Macro-F1 | Terminal acc. | False edit | Missed edit | "
        "Cost | Utility AUC | Delta vs selector |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['beta']:.3f} | {row['seeds']} | "
            f"{row['macro_f1_mean']:.4f} +/- {row['macro_f1_std']:.4f} | "
            f"{row['terminal_accuracy_mean']:.4f} | "
            f"{row['false_edit_rate_mean']:.4f} | "
            f"{row['missed_edit_rate_mean']:.4f} | {row['mean_cost_mean']:.4f} | "
            f"{row['utility_auc_mean']:.4f} | {row['delta_vs_selector_mean']:+.4f} |"
        )
    (args.output / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
