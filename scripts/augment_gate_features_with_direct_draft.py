#!/usr/bin/env python3
"""Append runtime-observable direct-draft state to frozen visual gate features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from activemap.models import EditOperation
from scripts.assemble_policy_relative_crossfit_features import (
    _bundle,
    _jsonl,
    _sha256,
    _write_bundle,
)

OPERATIONS = tuple(operation.value for operation in EditOperation)


def direct_draft_vector(branch: dict[str, Any]) -> np.ndarray:
    direct = str(branch["direct_operation"])
    belief = str(branch["belief_operation"])
    if direct not in OPERATIONS or belief not in OPERATIONS:
        raise ValueError("branch contains an unsupported edit operation")
    direct_index = OPERATIONS.index(direct)
    belief_index = OPERATIONS.index(belief)
    direct_one_hot = np.eye(len(OPERATIONS), dtype=np.float32)[direct_index]
    belief_one_hot = np.eye(len(OPERATIONS), dtype=np.float32)[belief_index]
    transition = np.zeros(len(OPERATIONS) ** 2, dtype=np.float32)
    transition[belief_index * len(OPERATIONS) + direct_index] = 1.0
    scalars = np.asarray(
        [
            direct == belief,
            direct != EditOperation.KEEP.value,
            belief != EditOperation.KEEP.value,
            bool(branch["direct_terminal_valid"]),
            float(branch["tool_cost"]),
        ],
        dtype=np.float32,
    )
    return np.concatenate([direct_one_hot, belief_one_hot, transition, scalars])


def _augment(
    feature_root: Path,
    branch_root: Path,
    output_root: Path,
    *,
    split: str,
) -> dict[str, Any]:
    features, records, _ = _bundle(feature_root)
    branch_rows = _jsonl(branch_root / "traces.jsonl")
    branches = {str(row["example_id"]): row for row in branch_rows}
    if len(branches) != len(branch_rows):
        raise ValueError("direct branch cache contains duplicate example IDs")
    if {str(row["example_id"]) for row in records} != set(branches):
        raise ValueError("gate features and direct branch IDs differ")
    draft_blocks = []
    augmented_records = []
    for record in records:
        branch = branches[str(record["example_id"])]
        if record["split"] != split or branch["split"] != split:
            raise ValueError("draft-conditioned bundle contains the wrong split")
        if str(record["task_id"]) != str(branch["task_id"]):
            raise ValueError("gate feature and direct branch task IDs differ")
        draft_blocks.append(direct_draft_vector(branch))
        augmented_records.append(
            {
                **record,
                "direct_operation": branch["direct_operation"],
                "belief_operation": branch["belief_operation"],
                "direct_changed_belief": branch["direct_operation"]
                != branch["belief_operation"],
                "direct_terminal_valid": bool(branch["direct_terminal_valid"]),
                "tool_cost": float(branch["tool_cost"]),
            }
        )
    draft_features = np.stack(draft_blocks)
    augmented = np.concatenate([features.astype(np.float32), draft_features], axis=1)
    summary = _write_bundle(
        output_root,
        augmented,
        augmented_records,
        split=split,
        sources=[
            {
                "feature_summary_sha256": _sha256(feature_root / "summary.json"),
                "branch_summary_sha256": _sha256(branch_root / "summary.json"),
            }
        ],
    )
    summary["crossfit_protocol"] = "current-policy-direct-draft-conditioned"
    summary["base_feature_dim"] = int(features.shape[1])
    summary["draft_feature_dim"] = int(draft_features.shape[1])
    summary["draft_feature_names"] = [
        "direct_operation_one_hot_4",
        "belief_operation_one_hot_4",
        "belief_to_direct_transition_one_hot_16",
        "direct_equals_belief",
        "direct_is_edit",
        "belief_is_edit",
        "direct_terminal_valid",
        "tool_cost",
    ]
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def augment_direct_draft_features(
    base_feature_root: Path,
    train_branch_root: Path,
    val_branch_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    output_root.mkdir(parents=True)
    train = _augment(
        base_feature_root / "train",
        train_branch_root,
        output_root / "train",
        split="train",
    )
    val = _augment(
        base_feature_root / "val",
        val_branch_root,
        output_root / "val",
        split="val",
    )
    if train["feature_dim"] != val["feature_dim"]:
        raise ValueError("draft-conditioned train and validation dimensions differ")
    summary = {
        "schema_version": "direct-draft-conditioned-gate-features-v1",
        "decision_order": "DIRECT_DRAFT-SELECT-TOOL-BELIEF_UPDATE-COMMIT",
        "train": train,
        "val": val,
        "test_assets_read": False,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_feature_root", type=Path)
    parser.add_argument("train_branch_root", type=Path)
    parser.add_argument("val_branch_root", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    result = augment_direct_draft_features(
        args.base_feature_root,
        args.train_branch_root,
        args.val_branch_root,
        args.output_root,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
