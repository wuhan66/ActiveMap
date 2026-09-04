#!/usr/bin/env python3
"""Gate executable writeback on real action support from fresh policy rollouts.

This is intentionally earlier than executable reward assembly. It prevents
spending updater/writeback compute on a policy that still executes only one
terminal action, while distinguishing model actions from behavior overrides.
"""

from __future__ import annotations

import argparse
import json
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
            rows.append(value)
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def _identity(row: dict[str, Any]) -> tuple[str, float]:
    if row.get("split") == "test":
        raise ValueError("coverage audit refuses test rows")
    return str(row["task_id"]), float(row["budget"])


def audit_executed_coverage(
    rollouts: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]],
    *,
    minimum_keep_trajectories: int,
    minimum_commit_trajectories: int,
    minimum_tool_trajectories: int,
    maximum_fallback_rate: float,
) -> dict[str, Any]:
    if len(rollouts) < 4:
        raise ValueError("recurrent coverage audit requires at least four rollout shards")
    if min(
        minimum_keep_trajectories,
        minimum_commit_trajectories,
        minimum_tool_trajectories,
    ) < 0:
        raise ValueError("minimum action coverage thresholds must be non-negative")
    if not 0.0 <= maximum_fallback_rate <= 1.0:
        raise ValueError("maximum fallback rate must be in [0, 1]")

    reference: set[tuple[str, float]] | None = None
    terminal_counts: Counter[str] = Counter()
    model_action_counts: Counter[str] = Counter()
    total_trajectories = 0
    tool_trajectories = 0
    calls = 0
    fallback_calls = 0
    behavior_override_calls = 0
    shard_reports = []
    for shard_id, (trajectory_rows, call_rows) in enumerate(rollouts):
        identities = {_identity(row) for row in trajectory_rows}
        if len(identities) != len(trajectory_rows):
            raise ValueError(f"rollout {shard_id} has duplicate task-budget rows")
        if reference is None:
            reference = identities
        elif identities != reference:
            raise ValueError(f"rollout {shard_id} does not share initial states with rollout 0")
        trajectory_counts = Counter(str(row.get("prediction", "UNKNOWN")) for row in trajectory_rows)
        terminal_counts.update(trajectory_counts)
        tool_count = sum(int(row.get("tool_calls", 0)) > 0 for row in trajectory_rows)
        tool_trajectories += tool_count
        total_trajectories += len(trajectory_rows)
        for row in call_rows:
            _identity(row)
            calls += 1
            fallback_calls += bool(row.get("fallback_used", False))
            behavior_override_calls += int("behavior_action" in row or "override_probability" in row)
            model_action_counts[str(row.get("executed_action", "UNKNOWN"))] += 1
        shard_reports.append(
            {
                "shard_id": shard_id,
                "trajectories": len(trajectory_rows),
                "terminal_actions": dict(sorted(trajectory_counts.items())),
                "tool_trajectories": tool_count,
                "calls": len(call_rows),
                "fallback_calls": sum(bool(row.get("fallback_used", False)) for row in call_rows),
            }
        )
    keep = terminal_counts["REJECT"]
    commit = sum(count for action, count in terminal_counts.items() if action.startswith("COMMIT"))
    fallback_rate = fallback_calls / max(calls, 1)
    gates = {
        "aligned_initial_states": reference is not None,
        "executed_keep_support": keep >= minimum_keep_trajectories,
        "executed_commit_support": commit >= minimum_commit_trajectories,
        "tool_trajectory_support": tool_trajectories >= minimum_tool_trajectories,
        "fallback_rate": fallback_rate <= maximum_fallback_rate,
        "no_behavior_override": behavior_override_calls == 0,
    }
    return {
        "schema_version": "activemap-recurrent-executed-coverage-v1",
        "rollout_shards": len(rollouts),
        "aligned_initial_state_groups": len(reference or set()),
        "trajectories": total_trajectories,
        "executed_terminal_action_counts": dict(sorted(terminal_counts.items())),
        "model_action_counts": dict(sorted(model_action_counts.items())),
        "executed_keep_trajectories": keep,
        "executed_commit_trajectories": commit,
        "tool_trajectories": tool_trajectories,
        "calls": calls,
        "fallback_calls": fallback_calls,
        "fallback_rate": fallback_rate,
        "behavior_override_calls": behavior_override_calls,
        "minimum_keep_trajectories": minimum_keep_trajectories,
        "minimum_commit_trajectories": minimum_commit_trajectories,
        "minimum_tool_trajectories": minimum_tool_trajectories,
        "maximum_fallback_rate": maximum_fallback_rate,
        "gates": gates,
        "ready_for_executable_writeback": all(gates.values()),
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "per_shard": shard_reports,
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--rollout", action="append", nargs=2, required=True,
        metavar=("TRAJECTORIES", "LLM_CALLS"),
    )
    parser.add_argument("--minimum-keep-trajectories", type=int, default=16)
    parser.add_argument("--minimum-commit-trajectories", type=int, default=16)
    parser.add_argument("--minimum-tool-trajectories", type=int, default=16)
    parser.add_argument("--maximum-fallback-rate", type=float, default=0.01)
    args = parser.parse_args()
    report = audit_executed_coverage(
        [(_read_jsonl(Path(pair[0])), _read_jsonl(Path(pair[1]))) for pair in args.rollout],
        minimum_keep_trajectories=args.minimum_keep_trajectories,
        minimum_commit_trajectories=args.minimum_commit_trajectories,
        minimum_tool_trajectories=args.minimum_tool_trajectories,
        maximum_fallback_rate=args.maximum_fallback_rate,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["ready_for_executable_writeback"]:
        raise SystemExit(12)


if __name__ == "__main__":
    main()
