#!/usr/bin/env python3
"""Build conservative gate targets that are beneficial across policy snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.assemble_policy_relative_crossfit_features import (
    _bundle,
    _sha256,
    _write_bundle,
)


def assemble_policy_robust_features(
    onpolicy_root: Path,
    crossfit_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    current_x, current_rows, _ = _bundle(onpolicy_root / "train")
    crossfit_x, crossfit_rows, _ = _bundle(crossfit_root / "train")
    if current_x.shape != crossfit_x.shape or not np.array_equal(current_x, crossfit_x):
        raise ValueError("current-policy and crossfit train feature tensors differ")
    crossfit_by_id = {str(row["example_id"]): row for row in crossfit_rows}
    if len(crossfit_by_id) != len(crossfit_rows):
        raise ValueError("crossfit features contain duplicate example IDs")
    robust_rows = []
    for current in current_rows:
        example_id = str(current["example_id"])
        if example_id not in crossfit_by_id:
            raise ValueError("current-policy and crossfit feature IDs differ")
        crossfit = crossfit_by_id[example_id]
        if str(current["task_id"]) != str(crossfit["task_id"]):
            raise ValueError("current-policy and crossfit task IDs differ")
        current_advantage = float(current["policy_relative_advantage"])
        crossfit_advantage = float(crossfit["policy_relative_advantage"])
        robust_advantage = min(current_advantage, crossfit_advantage)
        robust_rows.append(
            {
                **current,
                "oracle_use_tool": robust_advantage > 0.0,
                "policy_relative_use_tool": robust_advantage > 0.0,
                "consensus_mean_utility_gain": robust_advantage,
                "policy_robust_advantage": robust_advantage,
                "current_policy_advantage": current_advantage,
                "crossfit_policy_advantage": crossfit_advantage,
            }
        )
    if len(robust_rows) != len(crossfit_rows):
        raise ValueError("current-policy and crossfit feature IDs differ")

    val_x, val_rows, _ = _bundle(onpolicy_root / "val")
    output_root.mkdir(parents=True)
    train_summary = _write_bundle(
        output_root / "train",
        current_x,
        robust_rows,
        split="train",
        sources=[
            {
                "onpolicy_summary_sha256": _sha256(onpolicy_root / "train" / "summary.json"),
                "crossfit_summary_sha256": _sha256(crossfit_root / "train" / "summary.json"),
            }
        ],
    )
    train_summary["crossfit_protocol"] = "positive-only-if-beneficial-under-both-policy-snapshots"
    train_summary["current_policy_positive_count"] = sum(
        float(row["current_policy_advantage"]) > 0.0 for row in robust_rows
    )
    train_summary["crossfit_policy_positive_count"] = sum(
        float(row["crossfit_policy_advantage"]) > 0.0 for row in robust_rows
    )
    (output_root / "train" / "summary.json").write_text(
        json.dumps(train_summary, indent=2) + "\n", encoding="utf-8"
    )
    val_summary = _write_bundle(
        output_root / "val",
        val_x,
        val_rows,
        split="val",
        sources=[
            {"onpolicy_summary_sha256": _sha256(onpolicy_root / "val" / "summary.json")}
        ],
    )
    summary = {
        "schema_version": "policy-robust-gate-feature-assembly-v1",
        "train_target": "minimum-realized-advantage-across-crossfit-and-final-policy",
        "validation_target": "final-policy-realized-advantage",
        "train": train_summary,
        "val": val_summary,
        "test_assets_read": False,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("onpolicy_root", type=Path)
    parser.add_argument("crossfit_root", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    result = assemble_policy_robust_features(
        args.onpolicy_root, args.crossfit_root, args.output_root
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
