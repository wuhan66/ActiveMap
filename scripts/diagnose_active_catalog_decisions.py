#!/usr/bin/env python3
"""Decompose Active-Catalog regret into call-gate and candidate-choice errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def decompose(
    traces: list[dict[str, Any]], evaluation_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    evaluation = {str(row["example_id"]): row for row in evaluation_rows}
    if len(evaluation) != len(evaluation_rows):
        raise ValueError("duplicate evaluation example_id")
    totals = {
        "total_regret": 0.0,
        "false_stop_regret": 0.0,
        "false_call_regret": 0.0,
        "candidate_choice_regret": 0.0,
        "unattributed_regret": 0.0,
    }
    counts = {
        "states": len(traces),
        "false_stops": 0,
        "false_calls": 0,
        "correct_gate_wrong_candidate": 0,
        "correct_exact_calls": 0,
    }
    oracle_candidate_utility_sum = 0.0
    original_utility_sum = 0.0
    for trace in traces:
        row = evaluation.get(str(trace["example_id"]))
        if row is None:
            raise ValueError(f"missing evaluation row: {trace['example_id']}")
        utilities = {str(item["evidence_id"]): float(item["utility"]) for item in row["candidates"]}
        stop = float(row["stop_utility"])
        best_id = max(utilities, key=lambda value: (utilities[value], value))
        best_utility = utilities[best_id]
        target_call = best_utility > stop
        predicted_call = trace["predicted_selection"] == "ACQUIRE"
        selected_id = trace.get("predicted_evidence_id")
        selected_utility = utilities.get(str(selected_id), stop) if predicted_call else stop
        oracle = max(stop, best_utility)
        regret = max(oracle - selected_utility, 0.0)
        totals["total_regret"] += regret
        original_utility_sum += selected_utility
        oracle_candidate_utility_sum += best_utility if predicted_call else stop
        if target_call and not predicted_call:
            totals["false_stop_regret"] += oracle - stop
            counts["false_stops"] += 1
        elif not target_call and predicted_call:
            totals["false_call_regret"] += stop - selected_utility
            counts["false_calls"] += 1
        elif target_call and predicted_call:
            candidate_regret = max(best_utility - selected_utility, 0.0)
            totals["candidate_choice_regret"] += candidate_regret
            if selected_id == best_id:
                counts["correct_exact_calls"] += 1
            else:
                counts["correct_gate_wrong_candidate"] += 1
    attributed = sum(
        value
        for key, value in totals.items()
        if key not in {"total_regret", "unattributed_regret"}
    )
    totals["unattributed_regret"] = max(totals["total_regret"] - attributed, 0.0)
    states = max(len(traces), 1)
    return {
        "schema_version": "active-catalog-decision-decomposition-v1",
        "counts": counts,
        "regret": {
            **totals,
            "false_stop_fraction": totals["false_stop_regret"] / max(totals["total_regret"], 1e-12),
            "false_call_fraction": totals["false_call_regret"] / max(totals["total_regret"], 1e-12),
            "candidate_choice_fraction": totals["candidate_choice_regret"]
            / max(totals["total_regret"], 1e-12),
        },
        "counterfactual_mean_utility": {
            "vlm_original": original_utility_sum / states,
            "vlm_gate_oracle_candidate": oracle_candidate_utility_sum / states,
        },
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", type=Path)
    parser.add_argument("evaluation_index", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = decompose(load_jsonl(args.traces), load_jsonl(args.evaluation_index))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
