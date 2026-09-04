#!/usr/bin/env python3
"""Create task-disjoint selector fit/calibration partitions from training only."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from activemap.agent.sequential_controller import (
    ControllerStage,
    SequentialControllerAction,
)
from activemap.agent.vlm_sft import load_vlm_sft_rows

FORBIDDEN_PROMPT_KEYS = {
    "ground_truth",
    "gt_edit",
    "oracle_use_tool",
    "policy_relative_advantage",
    "target_operation",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assistant_action(row: dict[str, Any]) -> SequentialControllerAction:
    text = next(
        part["text"]
        for part in row["messages"][2]["content"]
        if part.get("type") == "text"
    )
    return SequentialControllerAction.model_validate_json(text)


def audit_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    actions: Counter[str] = Counter()
    tasks = set()
    trajectories = set()
    for row in rows:
        if row["split"] != "train" or row["stage"] != ControllerStage.SELECT.value:
            raise ValueError("fit/calibration rows must be training SELECT records")
        action = assistant_action(row)
        actions[action.selection.value] += 1
        tasks.add(str(row["task_id"]))
        trajectory = str(row["trajectory_id"])
        if trajectory in trajectories and "augmentation_copy" not in row:
            raise ValueError("duplicate unaugmented selector trajectory")
        trajectories.add(trajectory)
        prompt = next(
            part["text"]
            for part in row["messages"][1]["content"]
            if part.get("type") == "text"
        )
        leaked = FORBIDDEN_PROMPT_KEYS.intersection(json.loads(prompt))
        if leaked:
            raise ValueError(f"selector prompt leaks target metadata: {sorted(leaked)}")
    return {
        "records": len(rows),
        "tasks": len(tasks),
        "trajectories": len(trajectories),
        "action_counts": dict(sorted(actions.items())),
    }


def split_by_task(
    rows: list[dict[str, Any]], *, calibration_fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not 0.05 <= calibration_fraction <= 0.4:
        raise ValueError("calibration fraction must be between 0.05 and 0.4")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_id"])].append(row)
    strata: dict[int, list[str]] = defaultdict(list)
    for task_id, task_rows in grouped.items():
        positive_count = sum(
            assistant_action(row).selection.value == "ACQUIRE" for row in task_rows
        )
        strata[positive_count].append(task_id)
    calibration_tasks = set()
    stratum_summary = {}
    for positive_count, task_ids in sorted(strata.items()):
        ordered = sorted(
            task_ids,
            key=lambda task_id: hashlib.sha256(
                f"{seed}:{positive_count}:{task_id}".encode()
            ).hexdigest(),
        )
        count = round(len(ordered) * calibration_fraction)
        if len(ordered) > 1:
            count = min(max(count, 1), len(ordered) - 1)
        calibration_tasks.update(ordered[:count])
        stratum_summary[str(positive_count)] = {
            "tasks": len(ordered),
            "calibration_tasks": count,
        }
    fit = [row for row in rows if str(row["task_id"]) not in calibration_tasks]
    calibration = [row for row in rows if str(row["task_id"]) in calibration_tasks]
    fit_tasks = {str(row["task_id"]) for row in fit}
    calibration_task_ids = {str(row["task_id"]) for row in calibration}
    if fit_tasks & calibration_task_ids:
        raise ValueError("fit and calibration task overlap")
    return fit, calibration, {
        "seed": seed,
        "calibration_fraction": calibration_fraction,
        "strata_by_positive_record_count": stratum_summary,
        "task_overlap": 0,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    return sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--positive-multiplier", type=int, default=9)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.positive_multiplier < 1:
        raise ValueError("positive multiplier must be positive")
    source_rows = load_vlm_sft_rows(args.source_jsonl)
    source_audit = audit_rows(source_rows)
    fit, calibration, split = split_by_task(
        source_rows,
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
    )
    balanced_fit = []
    for row in fit:
        positive = assistant_action(row).selection.value == "ACQUIRE"
        for copy_index in range(args.positive_multiplier if positive else 1):
            balanced_fit.append({**copy.deepcopy(row), "augmentation_copy": copy_index})
    args.output_dir.mkdir(parents=True)
    outputs = {
        "fit": fit,
        "fit_balanced": balanced_fit,
        "calibration": calibration,
    }
    files = {}
    for name, rows in outputs.items():
        path = args.output_dir / f"{name}.jsonl"
        files[name] = {
            "path": str(path.resolve()),
            "sha256": write_jsonl(path, rows),
            "audit": audit_rows(rows),
        }
    summary = {
        "schema_version": "sequential-selector-fit-calibration-v1",
        "source": {
            "path": str(args.source_jsonl.resolve()),
            "sha256": sha256(args.source_jsonl),
            "audit": source_audit,
        },
        "split": split,
        "positive_multiplier": args.positive_multiplier,
        "files": files,
        "validation_assets_read": False,
        "test_assets_read": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
