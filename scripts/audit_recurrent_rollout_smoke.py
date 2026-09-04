#!/usr/bin/env python3
"""Fail-closed readiness audit for one recurrent rollout smoke shard."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            if value.get("split") == "test":
                raise ValueError("recurrent smoke audit refuses test rows")
            rows.append(value)
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def audit_smoke(
    trajectories: list[dict[str, Any]],
    calls: list[dict[str, Any]],
    *,
    minimum_keep: int,
    minimum_commit: int,
    minimum_tool: int,
    maximum_fallback_rate: float,
) -> dict[str, Any]:
    if min(minimum_keep, minimum_commit, minimum_tool) < 0:
        raise ValueError("minimum support values must be non-negative")
    if not 0.0 <= maximum_fallback_rate <= 1.0:
        raise ValueError("maximum fallback rate must be in [0, 1]")
    terminal = Counter(str(row.get("prediction", "UNKNOWN")) for row in trajectories)
    executed = Counter(str(row.get("executed_action", "UNKNOWN")) for row in calls)
    fallback = sum(bool(row.get("fallback_used", False)) for row in calls)
    overrides = sum(
        int("behavior_action" in row or "override_probability" in row) for row in calls
    )
    old_logprob_failures = 0
    for row in calls:
        payload = row.get("training_payload")
        if not isinstance(payload, dict):
            old_logprob_failures += 1
            continue
        values = payload.get("old_token_logprobs")
        sequence = payload.get("old_sequence_logprob")
        if (
            not isinstance(values, list)
            or not values
            or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values)
            or not isinstance(sequence, (int, float))
            or not math.isfinite(sequence)
        ):
            old_logprob_failures += 1
    commit = sum(count for key, count in terminal.items() if key.startswith("COMMIT"))
    tool_trajectories = sum(int(row.get("tool_calls", 0)) > 0 for row in trajectories)
    fallback_rate = fallback / max(len(calls), 1)
    gates = {
        "executed_keep_support": terminal["REJECT"] >= minimum_keep,
        "executed_commit_support": commit >= minimum_commit,
        "tool_trajectory_support": tool_trajectories >= minimum_tool,
        "fallback_rate": fallback_rate <= maximum_fallback_rate,
        "valid_old_logprobs": old_logprob_failures == 0,
        "no_behavior_override": overrides == 0,
    }
    return {
        "schema_version": "activemap-recurrent-smoke-audit-v1",
        "trajectory_count": len(trajectories),
        "call_count": len(calls),
        "terminal_actions": dict(sorted(terminal.items())),
        "executed_actions": dict(sorted(executed.items())),
        "executed_keep_trajectories": terminal["REJECT"],
        "executed_commit_trajectories": commit,
        "tool_trajectories": tool_trajectories,
        "fallback_calls": fallback,
        "fallback_rate": fallback_rate,
        "old_logprob_failures": old_logprob_failures,
        "behavior_override_calls": overrides,
        "thresholds": {
            "minimum_keep": minimum_keep,
            "minimum_commit": minimum_commit,
            "minimum_tool": minimum_tool,
            "maximum_fallback_rate": maximum_fallback_rate,
        },
        "gates": gates,
        "ready_for_four_gpu_collection": all(gates.values()),
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("trajectories", type=Path)
    parser.add_argument("calls", type=Path)
    parser.add_argument("--minimum-keep", type=int, default=16)
    parser.add_argument("--minimum-commit", type=int, default=16)
    parser.add_argument("--minimum-tool", type=int, default=16)
    parser.add_argument("--maximum-fallback-rate", type=float, default=0.01)
    args = parser.parse_args()
    report = audit_smoke(
        _read_jsonl(args.trajectories),
        _read_jsonl(args.calls),
        minimum_keep=args.minimum_keep,
        minimum_commit=args.minimum_commit,
        minimum_tool=args.minimum_tool,
        maximum_fallback_rate=args.maximum_fallback_rate,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["ready_for_four_gpu_collection"]:
        raise SystemExit(12)


if __name__ == "__main__":
    main()
