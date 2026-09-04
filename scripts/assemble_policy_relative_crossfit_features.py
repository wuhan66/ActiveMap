#!/usr/bin/env python3
"""Assemble cross-fitted VLM states and policy-relative acquisition labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def _bundle(root: Path) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    features = np.load(root / "features.npy")
    rows = _jsonl(root / "records.jsonl")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if len(features) != len(rows):
        raise ValueError(f"feature/record count mismatch: {root}")
    if summary.get("test_assets_read") is not False:
        raise ValueError(f"feature bundle violates frozen-test protocol: {root}")
    return features, rows, summary


def _join(
    features: np.ndarray,
    feature_rows: list[dict[str, Any]],
    branch_rows: list[dict[str, Any]],
    *,
    split: str,
    fold: int | None,
    assignment: dict[str, int] | None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    branches = {str(row["example_id"]): row for row in branch_rows}
    if len(branches) != len(branch_rows):
        raise ValueError("duplicate policy-relative branch example IDs")
    feature_ids = [str(row["example_id"]) for row in feature_rows]
    if set(feature_ids) != set(branches):
        raise ValueError("feature and policy-relative branch IDs differ")
    records = []
    for row in feature_rows:
        example_id = str(row["example_id"])
        branch = branches[example_id]
        task_id = str(row["task_id"])
        if row["split"] != split or branch["split"] != split:
            raise ValueError("joined bundle contains the wrong split")
        if str(branch["task_id"]) != task_id:
            raise ValueError("feature and branch task IDs differ")
        if assignment is not None and assignment.get(task_id) != fold:
            raise ValueError("train feature was not produced by its assigned outer fold")
        advantage = float(branch["policy_relative_advantage"])
        target = bool(branch["policy_relative_use_tool"])
        if target != (advantage > 0.0):
            raise ValueError("policy-relative label and advantage disagree")
        records.append(
            {
                "example_id": example_id,
                "task_id": task_id,
                "split": split,
                "oracle_use_tool": target,
                "policy_relative_use_tool": target,
                "policy_relative_advantage": advantage,
                "static_oracle_use_tool": bool(branch["static_use_tool"]),
                "consensus_mean_utility_gain": advantage,
                "crossfit_fold": fold,
                "direct_terminal_valid": bool(branch["direct_terminal_valid"]),
                "post_terminal_valid": bool(branch["post_terminal_valid"]),
            }
        )
    return features, records


def _write_bundle(
    root: Path,
    features: np.ndarray,
    records: list[dict[str, Any]],
    *,
    split: str,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    root.mkdir(parents=True)
    np.save(root / "features.npy", features.astype(np.float16))
    with (root / "records.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {
        "schema_version": "policy-relative-crossfit-gate-features-v1",
        "split": split,
        "sample_count": len(records),
        "task_count": len({str(row["task_id"]) for row in records}),
        "positive_count": sum(bool(row["oracle_use_tool"]) for row in records),
        "positive_rate": sum(bool(row["oracle_use_tool"]) for row in records) / len(records),
        "static_positive_count": sum(bool(row["static_oracle_use_tool"]) for row in records),
        "feature_dim": int(features.shape[1]),
        "feature_dtype": "float16",
        "label_protocol": "realized-post-tool-minus-direct-utility-positive",
        "utility_metadata": "policy-relative-realized-advantage",
        "crossfit_protocol": (
            "task-disjoint-outer-fold-adapter" if split == "train" else "held-out-validation"
        ),
        "assistant_tokens_seen": False,
        "sources": sources,
        "test_assets_read": False,
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def assemble(
    folds_root: Path,
    branch_root: Path,
    feature_root: Path,
    val_branch_root: Path,
    val_feature_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    fold_summary_path = folds_root / "summary.json"
    fold_summary = json.loads(fold_summary_path.read_text(encoding="utf-8"))
    if fold_summary.get("test_assets_read") is not False:
        raise ValueError("fold manifest violates the frozen-test protocol")
    assignments = {
        str(row["task_id"]): int(row["outer_fold"])
        for row in _jsonl(folds_root / "outer_task_assignments.jsonl")
    }
    train_blocks = []
    train_records = []
    train_sources = []
    feature_dim = None
    for spec in sorted(fold_summary["folds"], key=lambda row: int(row["fold"])):
        fold = int(spec["fold"])
        fold_features_root = feature_root / f"fold{fold}"
        fold_branches_root = branch_root / f"fold{fold}"
        features, feature_rows, feature_summary = _bundle(fold_features_root)
        branch_rows = _jsonl(fold_branches_root / "traces.jsonl")
        joined_features, joined_rows = _join(
            features,
            feature_rows,
            branch_rows,
            split="train",
            fold=fold,
            assignment=assignments,
        )
        if feature_dim is None:
            feature_dim = int(joined_features.shape[1])
        elif joined_features.shape[1] != feature_dim:
            raise ValueError("cross-fit feature dimensions differ")
        train_blocks.append(joined_features)
        train_records.extend(joined_rows)
        train_sources.append(
            {
                "fold": fold,
                "feature_summary": str((fold_features_root / "summary.json").resolve()),
                "feature_summary_sha256": _sha256(fold_features_root / "summary.json"),
                "branch_summary": str((fold_branches_root / "summary.json").resolve()),
                "branch_summary_sha256": _sha256(fold_branches_root / "summary.json"),
                "adapter": feature_summary["adapter"],
            }
        )
    if len(train_records) != int(fold_summary["source_examples"]):
        raise ValueError("cross-fit train bundle does not cover every source example")
    if len({row["example_id"] for row in train_records}) != len(train_records):
        raise ValueError("cross-fit train bundle contains duplicate examples")

    val_features, val_feature_rows, val_feature_summary = _bundle(val_feature_root)
    val_branch_rows = _jsonl(val_branch_root / "traces.jsonl")
    val_features, val_records = _join(
        val_features,
        val_feature_rows,
        val_branch_rows,
        split="val",
        fold=None,
        assignment=None,
    )
    if val_features.shape[1] != feature_dim:
        raise ValueError("train and validation feature dimensions differ")
    output_root.mkdir(parents=True)
    train_summary = _write_bundle(
        output_root / "train",
        np.concatenate(train_blocks, axis=0),
        train_records,
        split="train",
        sources=train_sources,
    )
    val_summary = _write_bundle(
        output_root / "val",
        val_features,
        val_records,
        split="val",
        sources=[
            {
                "feature_summary": str((val_feature_root / "summary.json").resolve()),
                "feature_summary_sha256": _sha256(val_feature_root / "summary.json"),
                "branch_summary": str((val_branch_root / "summary.json").resolve()),
                "branch_summary_sha256": _sha256(val_branch_root / "summary.json"),
                "adapter": val_feature_summary["adapter"],
            }
        ],
    )
    summary = {
        "schema_version": "policy-relative-crossfit-feature-assembly-v1",
        "fold_summary_sha256": _sha256(fold_summary_path),
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
    parser.add_argument("folds_root", type=Path)
    parser.add_argument("branch_root", type=Path)
    parser.add_argument("feature_root", type=Path)
    parser.add_argument("val_branch_root", type=Path)
    parser.add_argument("val_feature_root", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            assemble(
                args.folds_root,
                args.branch_root,
                args.feature_root,
                args.val_branch_root,
                args.val_feature_root,
                args.output_root,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
