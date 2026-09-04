#!/usr/bin/env python3
"""Join current-policy realized advantages with the same adapter's hidden states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.assemble_policy_relative_crossfit_features import (
    _bundle,
    _join,
    _jsonl,
    _sha256,
    _write_bundle,
)


def assemble_onpolicy_features(
    train_branch_root: Path,
    train_feature_root: Path,
    val_branch_root: Path,
    val_feature_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    train_x, train_feature_rows, train_feature_summary = _bundle(train_feature_root)
    train_x, train_records = _join(
        train_x,
        train_feature_rows,
        _jsonl(train_branch_root / "traces.jsonl"),
        split="train",
        fold=None,
        assignment=None,
    )
    val_x, val_feature_rows, val_feature_summary = _bundle(val_feature_root)
    val_x, val_records = _join(
        val_x,
        val_feature_rows,
        _jsonl(val_branch_root / "traces.jsonl"),
        split="val",
        fold=None,
        assignment=None,
    )
    if train_x.shape[1] != val_x.shape[1]:
        raise ValueError("on-policy train and validation feature dimensions differ")
    if train_feature_summary["adapter"] != val_feature_summary["adapter"]:
        raise ValueError("train and validation features use different deployment adapters")

    output_root.mkdir(parents=True)
    train_summary = _write_bundle(
        output_root / "train",
        train_x,
        train_records,
        split="train",
        sources=[
            {
                "feature_summary_sha256": _sha256(train_feature_root / "summary.json"),
                "branch_summary_sha256": _sha256(train_branch_root / "summary.json"),
                "adapter": train_feature_summary["adapter"],
            }
        ],
    )
    train_summary["crossfit_protocol"] = "current-deployment-policy-on-train-environments"
    (output_root / "train" / "summary.json").write_text(
        json.dumps(train_summary, indent=2) + "\n", encoding="utf-8"
    )
    val_summary = _write_bundle(
        output_root / "val",
        val_x,
        val_records,
        split="val",
        sources=[
            {
                "feature_summary_sha256": _sha256(val_feature_root / "summary.json"),
                "branch_summary_sha256": _sha256(val_branch_root / "summary.json"),
                "adapter": val_feature_summary["adapter"],
            }
        ],
    )
    summary = {
        "schema_version": "policy-relative-onpolicy-feature-assembly-v1",
        "representation": "full-deployment-adapter-state",
        "label_source": "same-full-adapter-realized-advantage",
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
    parser.add_argument("train_branch_root", type=Path)
    parser.add_argument("train_feature_root", type=Path)
    parser.add_argument("val_branch_root", type=Path)
    parser.add_argument("val_feature_root", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    result = assemble_onpolicy_features(
        args.train_branch_root,
        args.train_feature_root,
        args.val_branch_root,
        args.val_feature_root,
        args.output_root,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
