"""Portable manifest input and output."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd


def select_pilot_groups(
    frame: pd.DataFrame,
    *,
    groups_per_split: int = 1,
    min_observations: int = 12,
    seed: int = 20260710,
    require_udm: bool = False,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    required = {"aoi_id", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"manifest is missing pilot columns: {sorted(missing)}")
    if groups_per_split < 1 or min_observations < 1:
        raise ValueError("groups_per_split and min_observations must be positive")
    rng = random.Random(seed)
    selected: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        split_frame = frame[frame["split"] == split]
        counts = split_frame.groupby("aoi_id").size()
        eligible_ids = set(counts[counts >= min_observations].index.astype(str))
        if require_udm:
            if "has_udm" not in frame.columns:
                raise ValueError("require_udm needs a has_udm manifest column")
            with_udm = split_frame.groupby("aoi_id")["has_udm"].any()
            eligible_ids &= set(with_udm[with_udm].index.astype(str))
        eligible = sorted(eligible_ids)
        if len(eligible) < groups_per_split:
            raise ValueError(
                f"split {split} has {len(eligible)} AOIs with at least "
                f"{min_observations} observations; requested {groups_per_split}"
            )
        rng.shuffle(eligible)
        selected[split] = sorted(eligible[:groups_per_split])
    selected_ids = {aoi_id for values in selected.values() for aoi_id in values}
    pilot = frame[frame["aoi_id"].astype(str).isin(selected_ids)].copy()
    pilot = pilot.sort_values(["split", "aoi_id", "timestamp"], ignore_index=True)
    return pilot, selected


def read_manifest(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"manifest must be .parquet or .csv, got: {path}")


def write_manifest(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame.to_parquet(path, index=False)
        return
    if suffix == ".csv":
        frame.to_csv(path, index=False)
        return
    raise ValueError(f"manifest must be .parquet or .csv, got: {path}")
