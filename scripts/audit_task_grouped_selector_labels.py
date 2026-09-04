#!/usr/bin/env python3
"""Audit task-level support in a sequential SELECT SFT manifest.

The selector training manifest may contain intentional positive-example copies
for class balancing.  This audit removes those copies only for measurement,
then reports whether the states belonging to a task agree about acquisition.
It never writes labels or reads test data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _selection(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("missing messages")
    assistant = next(
        (message for message in reversed(messages) if message.get("role") == "assistant"),
        None,
    )
    if not isinstance(assistant, dict):
        raise ValueError("missing assistant target")
    content = assistant.get("content")
    if not isinstance(content, list) or len(content) != 1:
        raise ValueError("assistant target must contain one content block")
    target = json.loads(str(content[0]["text"]))
    selection = str(target.get("selection", ""))
    if target.get("stage") != "SELECT" or selection not in {"ACQUIRE", "STOP"}:
        raise ValueError("row is not a typed SELECT target")
    return selection


def _unique_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    by_trajectory: dict[str, dict[str, Any]] = {}
    duplicates = 0
    for row in rows:
        trajectory_id = str(row["trajectory_id"])
        signature = (
            str(row["task_id"]),
            _selection(row),
            float(row["policy_relative_advantage"]),
        )
        previous = by_trajectory.get(trajectory_id)
        if previous is None:
            by_trajectory[trajectory_id] = row
            continue
        previous_signature = (
            str(previous["task_id"]),
            _selection(previous),
            float(previous["policy_relative_advantage"]),
        )
        if signature != previous_signature:
            raise ValueError(f"conflicting copies for trajectory {trajectory_id}")
        duplicates += 1
    return list(by_trajectory.values()), duplicates


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not rows:
        raise ValueError("empty selector manifest")
    splits = {str(row["split"]) for row in rows}
    if len(splits) != 1 or not splits.issubset({"train", "val"}):
        raise ValueError("audit one train-only or validation-only manifest at a time")

    unique_rows, duplicate_count = _unique_rows(rows)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in unique_rows:
        by_task[str(row["task_id"])].append(row)

    task_rows: list[dict[str, Any]] = []
    for task_id, states in sorted(by_task.items()):
        advantages = [float(state["policy_relative_advantage"]) for state in states]
        selections = [_selection(state) for state in states]
        label_advantages_match = all(
            (selection == "ACQUIRE") == (advantage > 0.0)
            for selection, advantage in zip(selections, advantages, strict=True)
        )
        if not label_advantages_match:
            raise ValueError(f"selection/advantage mismatch for task {task_id}")
        positive_count = sum(selection == "ACQUIRE" for selection in selections)
        task_rows.append(
            {
                "task_id": task_id,
                "state_count": len(states),
                "acquire_count": positive_count,
                "acquire_rate": positive_count / len(states),
                "advantage_mean": _mean(advantages),
                "advantage_min": min(advantages),
                "advantage_max": max(advantages),
                "label_pattern": (
                    "all_acquire"
                    if positive_count == len(states)
                    else "all_stop"
                    if positive_count == 0
                    else "mixed"
                ),
            }
        )

    pattern_counts = Counter(row["label_pattern"] for row in task_rows)
    positive_state_count = sum(_selection(row) == "ACQUIRE" for row in unique_rows)
    report = {
        "schema_version": "task-grouped-selector-label-audit-v1",
        "role": "read-only training-label support audit",
        "split": next(iter(splits)),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": _sha256(args.manifest),
        "raw_record_count": len(rows),
        "unique_trajectory_count": len(unique_rows),
        "intentional_duplicate_count": duplicate_count,
        "task_count": len(task_rows),
        "state_count_histogram": dict(
            sorted(Counter(row["state_count"] for row in task_rows).items())
        ),
        "state_acquire_count": positive_state_count,
        "state_acquire_rate": positive_state_count / len(unique_rows),
        "task_label_patterns": dict(sorted(pattern_counts.items())),
        "task_advantage_mean": _mean([row["advantage_mean"] for row in task_rows]),
        "task_advantage_median": statistics.median(
            [row["advantage_mean"] for row in task_rows]
        ),
        "task_records": task_rows,
        "test_assets_read": False,
        "label_mutation_performed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    public_report = {key: value for key, value in report.items() if key != "task_records"}
    print(json.dumps(public_report, indent=2))


if __name__ == "__main__":
    main()
