"""Leakage-safe group-level dataset splits."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd


def _split_counts(group_count: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    if group_count < 3:
        raise ValueError("at least three groups are required for train/val/test splits")
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1:
        raise ValueError("train_ratio and val_ratio must be between zero and one")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than one")

    train_count = max(1, int(group_count * train_ratio))
    val_count = max(1, int(group_count * val_ratio))
    test_count = group_count - train_count - val_count
    if test_count < 1:
        train_count -= 1
        test_count += 1
    return train_count, val_count, test_count


def assign_group_splits(
    frame: pd.DataFrame,
    *,
    group_column: str = "aoi_id",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 20260710,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    if group_column not in frame.columns:
        raise ValueError(f"missing group column: {group_column}")
    groups = sorted(frame[group_column].dropna().astype(str).unique().tolist())
    random.Random(seed).shuffle(groups)
    train_count, val_count, _ = _split_counts(len(groups), train_ratio, val_ratio)

    split_groups = {
        "train": sorted(groups[:train_count]),
        "val": sorted(groups[train_count : train_count + val_count]),
        "test": sorted(groups[train_count + val_count :]),
    }
    group_to_split = {
        group: split_name
        for split_name, split_members in split_groups.items()
        for group in split_members
    }
    result = frame.copy()
    result["split"] = result[group_column].astype(str).map(group_to_split)
    if result["split"].isna().any():
        raise ValueError("some rows could not be assigned to a split")
    assert_no_group_leakage(result, group_column=group_column)
    return result, split_groups


def assert_no_group_leakage(frame: pd.DataFrame, *, group_column: str = "aoi_id") -> None:
    required = {group_column, "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing split columns: {sorted(missing)}")
    split_counts = frame.groupby(group_column, dropna=False)["split"].nunique()
    leaked = split_counts[split_counts > 1]
    if not leaked.empty:
        raise ValueError(f"groups occur in multiple splits: {leaked.index.tolist()}")


def write_split_files(split_groups: dict[str, list[str]], output_dir: Path, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, groups in split_groups.items():
        content = "\n".join(groups) + "\n"
        (output_dir / f"sn7_v1_{split_name}.txt").write_text(content, encoding="utf-8")
    metadata = {"seed": seed, "groups": split_groups}
    (output_dir / "sn7_v1_splits.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
