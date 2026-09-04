#!/usr/bin/env python3
"""Build task-disjoint SFT and rollout folds for policy-relative cross-fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def _write(path: Path, rows: list[dict[str, Any]]) -> str:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    return _sha256(path)


def _task_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row["task_id"]) for row in rows}


def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stages = Counter(str(row["stage"]) for row in rows)
    return {
        "records": len(rows),
        "tasks": len(_task_ids(rows)),
        "unique_examples": len({str(row["example_id"]) for row in rows}),
        "stage_counts": dict(stages),
        "pre_tool_positive_records": sum(
            row["stage"] == "PRE_TOOL" and bool(row["oracle_use_tool"]) for row in rows
        ),
    }


def build_folds(
    balanced_sft: Path,
    base_sft: Path,
    rollout_jsonl: Path,
    output_root: Path,
    *,
    folds: int,
    seed: int,
    inner_val_fraction: float,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    if folds < 2:
        raise ValueError("at least two outer folds are required")
    if not 0.0 < inner_val_fraction < 0.5:
        raise ValueError("inner validation fraction must be in (0, 0.5)")
    balanced_rows = _read(balanced_sft)
    base_rows = _read(base_sft)
    rollout_rows = _read(rollout_jsonl)
    if {str(row["split"]) for row in balanced_rows + base_rows + rollout_rows} != {"train"}:
        raise ValueError("cross-fit inputs must contain only split=train")

    pre_rows = [row for row in rollout_rows if row["stage"] == "PRE_TOOL"]
    post_rows = [row for row in rollout_rows if row["stage"] == "POST_TOOL"]
    pre_ids = {str(row["example_id"]) for row in pre_rows}
    post_ids = {str(row["example_id"]) for row in post_rows}
    if len(pre_rows) != len(pre_ids) or pre_ids != post_ids:
        raise ValueError("rollout must contain one complete PRE/POST pair per example")
    rollout_tasks = _task_ids(rollout_rows)
    if _task_ids(balanced_rows) != rollout_tasks or _task_ids(base_rows) != rollout_tasks:
        raise ValueError("SFT and rollout task sets differ")

    labels = np.asarray([int(row["oracle_use_tool"]) for row in pre_rows])
    groups = np.asarray([str(row["task_id"]) for row in pre_rows])
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_specs = []
    assignment: dict[str, int] = {}
    all_indices = np.arange(len(pre_rows))
    for fold, (_, holdout_indices) in enumerate(splitter.split(all_indices, labels, groups)):
        holdout_tasks = set(groups[holdout_indices])
        if assignment.keys() & holdout_tasks:
            raise RuntimeError("outer holdout tasks overlap")
        assignment.update({task_id: fold for task_id in holdout_tasks})
        remaining_tasks = sorted(rollout_tasks - holdout_tasks)
        task_positive = {
            task_id: any(
                bool(row["oracle_use_tool"])
                for row in pre_rows
                if str(row["task_id"]) == task_id
            )
            for task_id in remaining_tasks
        }
        stratify = [int(task_positive[task_id]) for task_id in remaining_tasks]
        fit_tasks_list, inner_val_tasks_list = train_test_split(
            remaining_tasks,
            test_size=inner_val_fraction,
            random_state=seed + fold,
            shuffle=True,
            stratify=stratify,
        )
        fit_tasks = set(fit_tasks_list)
        inner_val_tasks = set(inner_val_tasks_list)
        partitions_overlap = (
            bool(fit_tasks & inner_val_tasks)
            or bool(fit_tasks & holdout_tasks)
            or bool(inner_val_tasks & holdout_tasks)
        )
        if partitions_overlap:
            raise RuntimeError("cross-fit task partitions overlap")
        if fit_tasks | inner_val_tasks | holdout_tasks != rollout_tasks:
            raise RuntimeError("cross-fit task partitions do not cover the train split")

        fold_root = output_root / f"fold{fold}"
        fold_root.mkdir(parents=True)
        train_rows = [row for row in balanced_rows if str(row["task_id"]) in fit_tasks]
        val_rows = [row for row in base_rows if str(row["task_id"]) in inner_val_tasks]
        holdout_rows = [row for row in rollout_rows if str(row["task_id"]) in holdout_tasks]
        files = {
            "train_sft": {
                "path": str((fold_root / "train_sft.jsonl").resolve()),
                "sha256": _write(fold_root / "train_sft.jsonl", train_rows),
                **_stats(train_rows),
            },
            "inner_val_sft": {
                "path": str((fold_root / "inner_val_sft.jsonl").resolve()),
                "sha256": _write(fold_root / "inner_val_sft.jsonl", val_rows),
                **_stats(val_rows),
            },
            "holdout_rollout": {
                "path": str((fold_root / "holdout_rollout.jsonl").resolve()),
                "sha256": _write(fold_root / "holdout_rollout.jsonl", holdout_rows),
                **_stats(holdout_rows),
            },
        }
        fold_specs.append(
            {
                "fold": fold,
                "fit_task_count": len(fit_tasks),
                "inner_val_task_count": len(inner_val_tasks),
                "holdout_task_count": len(holdout_tasks),
                "holdout_static_positive_examples": sum(
                    bool(pre_rows[index]["oracle_use_tool"]) for index in holdout_indices
                ),
                "files": files,
            }
        )

    if set(assignment) != rollout_tasks:
        raise RuntimeError("outer folds do not cover every task exactly once")
    output_root.mkdir(parents=True, exist_ok=True)
    assignment_path = output_root / "outer_task_assignments.jsonl"
    _write(
        assignment_path,
        [
            {"task_id": task_id, "outer_fold": assignment[task_id]}
            for task_id in sorted(assignment)
        ],
    )
    summary = {
        "schema_version": "policy-relative-crossfit-folds-v1",
        "seed": seed,
        "fold_count": folds,
        "inner_val_fraction": inner_val_fraction,
        "source_examples": len(pre_rows),
        "source_tasks": len(rollout_tasks),
        "folds": fold_specs,
        "outer_assignment_sha256": _sha256(assignment_path),
        "sources": {
            "balanced_sft": {"path": str(balanced_sft.resolve()), "sha256": _sha256(balanced_sft)},
            "base_sft": {"path": str(base_sft.resolve()), "sha256": _sha256(base_sft)},
            "rollout": {"path": str(rollout_jsonl.resolve()), "sha256": _sha256(rollout_jsonl)},
        },
        "test_assets_read": False,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("balanced_sft", type=Path)
    parser.add_argument("base_sft", type=Path)
    parser.add_argument("rollout_jsonl", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--inner-val-fraction", type=float, default=0.10)
    args = parser.parse_args()
    result = build_folds(
        args.balanced_sft,
        args.base_sft,
        args.rollout_jsonl,
        args.output_root,
        folds=args.folds,
        seed=args.seed,
        inner_val_fraction=args.inner_val_fraction,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
