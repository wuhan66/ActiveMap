#!/usr/bin/env python3
"""Audit on-policy rollout diversity before executable GRPO collection."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def _key(row: dict[str, Any]) -> tuple[str, float]:
    if "task_id" not in row or "budget" not in row:
        raise ValueError("rollout rows require task_id and budget")
    return str(row["task_id"]), float(row["budget"])


def _index_trajectories(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, float], dict[str, Any]]:
    indexed: dict[tuple[str, float], dict[str, Any]] = {}
    for row in rows:
        key = _key(row)
        if key in indexed:
            raise ValueError(f"duplicate trajectory key: {key}")
        indexed[key] = row
    return indexed


def _index_calls(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, float], list[dict[str, Any]]]:
    indexed: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        indexed[_key(row)].append(row)
    for values in indexed.values():
        values.sort(key=lambda row: int(row.get("step", 0)))
    return dict(indexed)


def _trajectory_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("prediction"),
        int(row.get("acquisitions", 0)),
        int(row.get("tool_calls", 0)),
        int(row.get("tool_successes", 0)),
        bool(row.get("terminal_edit_changed_after_tools", False)),
    )


def _call_signature(rows: list[dict[str, Any]]) -> tuple[Any, ...]:
    return tuple(
        (
            str(row.get("raw_output", "")),
            str(row.get("executed_action", "")),
            bool(row.get("fallback_used", False)),
        )
        for row in rows
    )


def _contains_nonstop(row: dict[str, Any]) -> bool:
    return bool(
        row.get(
            "contains_nonstop_action",
            int(row.get("acquisitions", 0)) > 0 or int(row.get("tool_calls", 0)) > 0,
        )
    )


def _logprob_status(row: dict[str, Any]) -> str:
    payload = row.get("training_payload")
    if not isinstance(payload, dict):
        return "missing_payload"
    values = payload.get("old_token_logprobs")
    if not isinstance(values, list) or not values:
        return "missing_logprobs"
    try:
        numeric = [float(value) for value in values]
    except (TypeError, ValueError):
        return "invalid_logprobs"
    if not all(value == value and abs(value) != float("inf") for value in numeric):
        return "invalid_logprobs"
    if all(abs(value) <= 1e-8 for value in numeric):
        return "all_zero_logprobs"
    return str(payload.get("old_logprob_source", "present"))


def audit_rollout_diversity(
    rollouts: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]],
    *,
    minimum_variable_group_rate: float = 0.20,
    minimum_nonstop_rate: float = 0.05,
) -> dict[str, Any]:
    """Return a model-free diversity audit for aligned initial states."""

    if len(rollouts) < 4:
        raise ValueError("recurrent GRPO diversity audit requires at least four rollouts")
    if not 0 <= minimum_variable_group_rate <= 1:
        raise ValueError("minimum variable group rate must be in [0, 1]")
    if not 0 <= minimum_nonstop_rate <= 1:
        raise ValueError("minimum nonstop rate must be in [0, 1]")

    indexed = []
    call_indexed = []
    per_rollout = []
    for rollout_id, (trajectory_rows, call_rows) in enumerate(rollouts):
        trajectories = _index_trajectories(trajectory_rows)
        calls = _index_calls(call_rows)
        if set(calls) - set(trajectories):
            raise ValueError(f"rollout {rollout_id} contains calls without trajectories")
        logprob_counts = Counter(_logprob_status(row) for row in call_rows)
        per_rollout.append(
            {
                "rollout_id": rollout_id,
                "trajectories": len(trajectories),
                "calls": len(call_rows),
                "action_counts": dict(
                    Counter(str(row.get("prediction", "UNKNOWN")) for row in trajectory_rows)
                ),
                "tool_calls": sum(int(row.get("tool_calls", 0)) for row in trajectory_rows),
                "nonstop_trajectories": sum(_contains_nonstop(row) for row in trajectory_rows),
                "fallback_calls": sum(bool(row.get("fallback_used", False)) for row in call_rows),
                "logprob_status": dict(logprob_counts),
            }
        )
        indexed.append(trajectories)
        call_indexed.append(calls)

    reference_keys = set(indexed[0])
    aligned = all(set(values) == reference_keys for values in indexed[1:])
    common_keys = sorted(
        set.intersection(*(set(values) for values in indexed)),
        key=lambda value: (value[0], value[1]),
    )
    raw_variable = 0
    action_variable = 0
    tool_variable = 0
    belief_variable = 0
    for key in common_keys:
        raw_signatures = {_call_signature(calls.get(key, [])) for calls in call_indexed}
        action_signatures = {_trajectory_signature(trajectories[key]) for trajectories in indexed}
        tool_counts = {int(trajectories[key].get("tool_calls", 0)) for trajectories in indexed}
        belief_values = {
            float(trajectories[key].get("mean_tool_belief_l1_delta", 0.0))
            for trajectories in indexed
        }
        raw_variable += len(raw_signatures) > 1
        action_variable += len(action_signatures) > 1
        tool_variable += len(tool_counts) > 1
        belief_variable += len(belief_values) > 1

    group_count = len(common_keys)
    total_trajectories = sum(len(values) for values in indexed)
    nonstop_rate = sum(item["nonstop_trajectories"] for item in per_rollout) / max(
        total_trajectories, 1
    )
    zero_or_invalid_logprob_calls = sum(
        item["logprob_status"].get("all_zero_logprobs", 0)
        + item["logprob_status"].get("missing_logprobs", 0)
        + item["logprob_status"].get("missing_payload", 0)
        + item["logprob_status"].get("invalid_logprobs", 0)
        for item in per_rollout
    )
    raw_rate = raw_variable / max(group_count, 1)
    action_rate = action_variable / max(group_count, 1)
    tool_rate = tool_variable / max(group_count, 1)
    gates = {
        "at_least_four_rollouts": len(rollouts) >= 4,
        "aligned_initial_states": aligned,
        "raw_completion_variation": raw_rate >= minimum_variable_group_rate,
        "executable_action_variation": action_rate >= minimum_variable_group_rate,
        "nonstop_exploration": nonstop_rate >= minimum_nonstop_rate,
        "valid_old_logprobs": zero_or_invalid_logprob_calls == 0,
    }
    recurrent_ready = all(gates.values())
    return {
        "schema_version": "activemap-recurrent-rollout-diversity-v1",
        "rollouts": len(rollouts),
        "common_initial_state_groups": group_count,
        "reference_group_count": len(reference_keys),
        "aligned_initial_states": aligned,
        "variable_raw_completion_groups": raw_variable,
        "variable_raw_completion_rate": raw_rate,
        "variable_action_groups": action_variable,
        "variable_action_rate": action_rate,
        "variable_tool_count_groups": tool_variable,
        "variable_tool_count_rate": tool_rate,
        "variable_belief_groups": belief_variable,
        "nonstop_trajectory_rate": nonstop_rate,
        "zero_or_invalid_logprob_calls": zero_or_invalid_logprob_calls,
        "tool_branch_reachable": tool_rate > 0 or any(
            item["tool_calls"] > 0 for item in per_rollout
        ),
        "gates": gates,
        "ready_for_recurrent_grpo": recurrent_ready,
        "ready_for_tool_belief_grpo": recurrent_ready and (
            tool_rate > 0 or any(item["tool_calls"] > 0 for item in per_rollout)
        ),
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "per_rollout": per_rollout,
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--rollout",
        action="append",
        nargs=2,
        required=True,
        metavar=("TRAJECTORIES", "LLM_CALLS"),
    )
    parser.add_argument("--minimum-variable-group-rate", type=float, default=0.20)
    parser.add_argument("--minimum-nonstop-rate", type=float, default=0.05)
    args = parser.parse_args()
    report = audit_rollout_diversity(
        [(_read_jsonl(Path(pair[0])), _read_jsonl(Path(pair[1]))) for pair in args.rollout],
        minimum_variable_group_rate=args.minimum_variable_group_rate,
        minimum_nonstop_rate=args.minimum_nonstop_rate,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["ready_for_recurrent_grpo"]:
        raise SystemExit(12)


if __name__ == "__main__":
    main()
