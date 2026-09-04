#!/usr/bin/env python3
"""Audit split isolation and persisted contracts for sequential-controller SFT."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from activemap.agent.sequential_controller import SequentialControllerTrajectory
from scripts.build_sequential_controller_sft import _jsonl, audit_prompt_contract


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_isolation(
    train: list[SequentialControllerTrajectory],
    val: list[SequentialControllerTrajectory],
) -> dict[str, Any]:
    train_tasks = {row.task_id for row in train}
    val_tasks = {row.task_id for row in val}
    train_ids = {row.trajectory_id for row in train}
    val_ids = {row.trajectory_id for row in val}
    if len(train_ids) != len(train) or len(val_ids) != len(val):
        raise ValueError("a sequential split contains duplicate trajectory IDs")
    task_overlap = sorted(train_tasks.intersection(val_tasks))
    trajectory_overlap = sorted(train_ids.intersection(val_ids))
    if task_overlap or trajectory_overlap:
        raise ValueError(
            f"sequential split overlap: tasks={task_overlap[:3]}, "
            f"trajectories={trajectory_overlap[:3]}"
        )
    snapshots = {row.policy_snapshot for row in [*train, *val]}
    if len(snapshots) != 1:
        raise ValueError("train and validation trajectories use different policy snapshots")
    return {
        "train_tasks": len(train_tasks),
        "val_tasks": len(val_tasks),
        "task_overlap": 0,
        "train_trajectories": len(train),
        "val_trajectories": len(val),
        "trajectory_overlap": 0,
        "policy_snapshot": next(iter(snapshots)),
    }


def _audit_root(root: Path, expected_split: str) -> tuple[dict[str, Any], list[Any]]:
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("test_assets_read") is not False or summary["split"] != expected_split:
        raise ValueError(f"invalid sequential {expected_split} summary")
    trajectories = [
        SequentialControllerTrajectory.model_validate_json(line)
        for line in (root / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(trajectories) != int(summary["trajectory_count"]):
        raise ValueError("trajectory count differs from summary")
    if {row.split for row in trajectories} != {expected_split}:
        raise ValueError("trajectory payload contains the wrong split")
    files = {}
    for name, expected in summary["files"].items():
        path = root / f"{name}.jsonl"
        if _sha256(path) != expected["sha256"]:
            raise ValueError(f"sequential SFT hash mismatch: {name}")
        rows = _jsonl(path)
        audit = audit_prompt_contract(rows)
        if audit != expected["audit"] or len(rows) != expected["records"]:
            raise ValueError(f"sequential SFT audit mismatch: {name}")
        files[name] = audit
    return {
        "summary_sha256": _sha256(root / "summary.json"),
        "trajectory_sha256": _sha256(root / "trajectories.jsonl"),
        "trajectory_count": len(trajectories),
        "files": files,
    }, trajectories


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_root", type=Path)
    parser.add_argument("val_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    train_audit, train = _audit_root(args.train_root, "train")
    val_audit, val = _audit_root(args.val_root, "val")
    result = {
        "schema_version": "sequential-controller-split-audit-v1",
        "isolation": split_isolation(train, val),
        "train": train_audit,
        "val": val_audit,
        "test_assets_read": False,
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
