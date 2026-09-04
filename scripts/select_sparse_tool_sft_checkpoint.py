#!/usr/bin/env python3
"""Select a sparse-tool SFT checkpoint using frozen validation-only gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def assess_sparse_tool_checkpoints(
    summaries: dict[str, dict[str, Any]],
    *,
    min_schema_valid: float = 0.99,
    min_executable_valid: float = 0.99,
    min_grounded_call_recall: float = 0.10,
    min_tool_positive_exact: float = 0.10,
    max_false_call_rate: float = 0.02,
) -> dict[str, Any]:
    assessed = []
    for label, summary in summaries.items():
        tools = summary.get("tool_metrics", {})
        row = {
            "label": label,
            "sample_count": int(summary["sample_count"]),
            "schema_valid_rate": float(summary["schema_valid_rate"]),
            "executable_valid_rate": float(summary["executable_valid_rate"]),
            "exact_action_accuracy": float(summary["exact_action_accuracy"]),
            "macro_f1": float(summary["macro_f1"]),
            "target_tool_calls": int(tools.get("target_call_count", 0)),
            "predicted_tool_calls": int(tools.get("predicted_call_count", 0)),
            "grounded_call_recall": float(tools.get("grounded_call_accuracy", 0.0)),
            "tool_positive_exact": float(
                tools.get("tool_positive_exact_accuracy", 0.0)
            ),
            "false_call_rate": float(tools.get("false_call_rate", 1.0)),
            "test_assets_read": summary.get("test_assets_read"),
        }
        gates = {
            "test_isolation": row["test_assets_read"] is False,
            "schema_valid": row["schema_valid_rate"] >= min_schema_valid,
            "executable_valid": row["executable_valid_rate"] >= min_executable_valid,
            "tool_targets_present": row["target_tool_calls"] > 0,
            "grounded_call_recall": (
                row["grounded_call_recall"] >= min_grounded_call_recall
            ),
            "tool_positive_exact": (
                row["tool_positive_exact"] >= min_tool_positive_exact
            ),
            "false_call_rate": row["false_call_rate"] <= max_false_call_rate,
        }
        assessed.append(
            {
                **row,
                "eligible": all(gates.values()),
                "failed_gates": [name for name, passed in gates.items() if not passed],
            }
        )

    def rank(row: dict[str, Any]) -> tuple[float, ...]:
        return (
            row["tool_positive_exact"],
            row["grounded_call_recall"],
            -row["false_call_rate"],
            row["macro_f1"],
            row["exact_action_accuracy"],
        )

    eligible = [row for row in assessed if row["eligible"]]
    selected = max(eligible, key=rank) if eligible else None
    best_observed = max(assessed, key=rank) if assessed else None
    return {
        "protocol": {
            "split": "val",
            "test_assets_read": False,
            "gates": {
                "min_schema_valid": min_schema_valid,
                "min_executable_valid": min_executable_valid,
                "min_grounded_call_recall": min_grounded_call_recall,
                "min_tool_positive_exact": min_tool_positive_exact,
                "max_false_call_rate": max_false_call_rate,
            },
            "selection_order": [
                "tool_positive_exact",
                "grounded_call_recall",
                "negative_false_call_rate",
                "macro_f1",
                "exact_action_accuracy",
            ],
        },
        "selection_passed": selected is not None,
        "selected_checkpoint": selected["label"] if selected else None,
        "best_observed_checkpoint": best_observed["label"] if best_observed else None,
        "checkpoints": assessed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--labels", required=True)
    args = parser.parse_args()

    labels = [value.strip() for value in args.labels.split(",") if value.strip()]
    summaries = {}
    for label in labels:
        path = args.evaluation_root / label / "actions" / "summary.json"
        if path.is_file():
            summaries[label] = json.loads(path.read_text(encoding="utf-8"))
    if not summaries:
        raise SystemExit("no static checkpoint summaries found")
    decision = assess_sparse_tool_checkpoints(summaries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
