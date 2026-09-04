#!/usr/bin/env python3
"""Diagnose realized value and cost of hierarchical tool calls."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw_gains = [
        float(row["hierarchical_utility"])
        + float(row["tool_cost"])
        - float(row["direct_utility"])
        for row in rows
    ]
    net_gains = [
        float(row["hierarchical_utility"]) - float(row["direct_utility"])
        for row in rows
    ]
    return {
        "count": len(rows),
        "raw_edit_gain_sum": sum(raw_gains),
        "raw_edit_gain_mean": _mean(raw_gains),
        "net_utility_gain_sum": sum(net_gains),
        "net_utility_gain_mean": _mean(net_gains),
        "raw_gain_positive": sum(value > 0 for value in raw_gains),
        "raw_gain_zero": sum(value == 0 for value in raw_gains),
        "raw_gain_negative": sum(value < 0 for value in raw_gains),
        "net_gain_positive": sum(value > 0 for value in net_gains),
        "net_gain_zero": sum(value == 0 for value in net_gains),
        "net_gain_negative": sum(value < 0 for value in net_gains),
    }


def analyze(rows: list[dict[str, Any]], costs: list[float]) -> dict[str, Any]:
    if not rows:
        raise ValueError("at least one trace is required")
    called = [row for row in rows if bool(row["predicted_use_tool"])]
    if not called:
        raise ValueError("trace contains no tool calls")
    if any(float(row["tool_cost"]) <= 0 for row in called):
        raise ValueError("called rows must record positive tool cost")
    raw_gains = [
        float(row["hierarchical_utility"])
        + float(row["tool_cost"])
        - float(row["direct_utility"])
        for row in called
    ]
    total_raw_gain = sum(raw_gains)
    current_cost = _mean([float(row["tool_cost"]) for row in called])
    transitions = Counter(
        f'{row["direct_operation"]}->{row["hierarchical_operation"]}' for row in called
    )
    sensitivity = []
    for cost in costs:
        total = total_raw_gain - cost * len(called)
        sensitivity.append(
            {
                "cost_per_call": cost,
                "total_utility_delta": total,
                "mean_utility_delta_over_all_states": total / len(rows),
            }
        )
    return {
        "schema_version": "hierarchical-call-value-diagnostic-v1",
        "sample_count": len(rows),
        "called_count": len(called),
        "call_rate": len(called) / len(rows),
        "target_call_count": sum(bool(row["target_use_tool"]) for row in rows),
        "true_call_count": sum(bool(row["target_use_tool"]) for row in called),
        "false_call_count": sum(not bool(row["target_use_tool"]) for row in called),
        "operation_changed_count": sum(
            row["direct_operation"] != row["hierarchical_operation"] for row in called
        ),
        "current_cost_per_call": current_cost,
        "current_total_cost": sum(float(row["tool_cost"]) for row in called),
        "break_even_cost_per_call": total_raw_gain / len(called),
        "all_calls": _group(called),
        "target_positive_calls": _group(
            [row for row in called if bool(row["target_use_tool"])]
        ),
        "target_negative_calls": _group(
            [row for row in called if not bool(row["target_use_tool"])]
        ),
        "operation_transitions": dict(sorted(transitions.items())),
        "fixed_call_set_cost_sensitivity": sensitivity,
        "selection_warning": (
            "Validation diagnostics must not select a new threshold or formal tool cost."
        ),
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cost", type=float, action="append")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    costs = args.cost or [0.0, 0.25, 0.5, 0.75, 1.0]
    if any(not math.isfinite(cost) or cost < 0 for cost in costs):
        raise ValueError("costs must be finite and non-negative")
    rows = [
        json.loads(line)
        for line in args.traces.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = analyze(rows, costs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
