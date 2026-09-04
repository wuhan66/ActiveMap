#!/usr/bin/env python3
"""Audit that sparse tool supervision is reachable from acquisition trajectories."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            raise ValueError(f"{path}:{line_number} must contain three messages")
        row["_observation"] = json.loads(messages[1]["content"])
        row["_action"] = json.loads(messages[2]["content"])
        rows.append(row)
    if not rows:
        raise ValueError(f"no SFT rows in {path}")
    return rows


def _load_rollouts(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if path is None:
        return grouped
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            grouped[str(row["task_id"])].append(row)
    return grouped


def audit_reachability(
    composed_sft: Path, rollout_path: Path | None = None
) -> dict[str, Any]:
    rows = _load_jsonl(composed_sft)
    reachable_selected: dict[str, set[frozenset[str]]] = defaultdict(set)
    acquisition_targets: dict[str, set[str]] = defaultdict(set)
    source_counts: Counter[str] = Counter()

    for row in rows:
        source = str(row.get("composition_source", "unknown"))
        source_counts[source] += 1
        if source != "acquisition_agent":
            continue
        observation, action = row["_observation"], row["_action"]
        task_id = str(observation["task_id"])
        selected = frozenset(str(value) for value in observation["selected_evidence_ids"])
        reachable_selected[task_id].add(selected)
        if action.get("action") == "ACQUIRE":
            evidence_id = str(action["evidence_id"])
            acquisition_targets[task_id].add(evidence_id)
            reachable_selected[task_id].add(frozenset((*selected, evidence_id)))

    tool_rows = []
    tool_tasks: set[str] = set()
    reachable_tool_rows = 0
    grounded_tool_rows = 0
    task_reachability: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        action = row["_action"]
        if action.get("action") != "USE_TOOL":
            continue
        observation = row["_observation"]
        task_id = str(observation["task_id"])
        evidence_id = str(action["tool_call"]["inputs"]["evidence_id"])
        selected = {str(value) for value in observation["selected_evidence_ids"]}
        grounded = evidence_id in selected
        reachable = grounded and any(
            evidence_id in selected_state
            for selected_state in reachable_selected.get(task_id, set())
        )
        grounded_tool_rows += int(grounded)
        reachable_tool_rows += int(reachable)
        tool_tasks.add(task_id)
        task_reachability[task_id].append(reachable)
        tool_rows.append(
            {
                "task_id": task_id,
                "trajectory_id": str(row.get("trajectory_id", "")),
                "step": int(row.get("step", 0)),
                "evidence_id": evidence_id,
                "grounded_in_tool_observation": grounded,
                "reachable_after_acquisition": reachable,
            }
        )

    rollout_groups = _load_rollouts(rollout_path)
    rollout_rows = [row for task in tool_tasks for row in rollout_groups.get(task, [])]
    activated = sum(int(row.get("tool_calls", 0)) > 0 for row in rollout_rows)
    bridged_tasks = sum(all(values) for values in task_reachability.values())
    tasks_with_acquisition = sum(bool(acquisition_targets.get(task)) for task in tool_tasks)
    tool_count = len(tool_rows)
    report = {
        "schema_version": "activemap-agentic-reachability-audit-v1",
        "composed_sft": str(composed_sft.resolve()),
        "rollouts": str(rollout_path.resolve()) if rollout_path else None,
        "source_counts": dict(sorted(source_counts.items())),
        "tool_state_count": tool_count,
        "tool_task_count": len(tool_tasks),
        "grounded_tool_state_rate": grounded_tool_rows / max(tool_count, 1),
        "reachable_tool_state_rate": reachable_tool_rows / max(tool_count, 1),
        "fully_bridged_tool_task_rate": bridged_tasks / max(len(tool_tasks), 1),
        "tool_tasks_with_acquisition_target_rate": (
            tasks_with_acquisition / max(len(tool_tasks), 1)
        ),
        "rollout_tool_task_budget_count": len(rollout_rows),
        "rollout_tool_activation_rate": activated / max(len(rollout_rows), 1),
        "test_assets_read": False,
        "passed": bool(tool_count)
        and reachable_tool_rows == tool_count
        and bridged_tasks == len(tool_tasks),
        "unreachable_examples": [
            row for row in tool_rows if not row["reachable_after_acquisition"]
        ][:20],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("composed_sft", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rollouts", type=Path)
    parser.add_argument("--require-reachable", action="store_true")
    args = parser.parse_args()
    report = audit_reachability(args.composed_sft, args.rollouts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.require_reachable and not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
