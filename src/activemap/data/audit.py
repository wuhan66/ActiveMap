"""Fast manifest checks that fail before expensive preprocessing begins."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from activemap.data.splits import assert_no_group_leakage

REQUIRED_COLUMNS = {"aoi_id", "timestamp", "image_path", "has_label", "has_udm"}


def audit_manifest(frame: pd.DataFrame, *, check_paths: bool = True) -> list[str]:
    issues: list[str] = []
    missing_columns = REQUIRED_COLUMNS - set(frame.columns)
    if missing_columns:
        return [f"missing required columns: {sorted(missing_columns)}"]

    duplicates = frame.duplicated(["aoi_id", "timestamp"], keep=False)
    if duplicates.any():
        keys = frame.loc[duplicates, ["aoi_id", "timestamp"]].drop_duplicates()
        issues.append(f"duplicate AOI/timestamp keys: {keys.to_dict(orient='records')}")

    invalid_timestamps = ~frame["timestamp"].astype(str).str.fullmatch(r"20\d{2}_(0[1-9]|1[0-2])")
    if invalid_timestamps.any():
        issues.append(f"invalid timestamps: {frame.loc[invalid_timestamps, 'timestamp'].tolist()}")

    if "split" in frame.columns:
        try:
            assert_no_group_leakage(frame)
        except ValueError as exc:
            issues.append(str(exc))

    if check_paths:
        for column in ("image_path", "label_path", "udm_path"):
            if column not in frame.columns:
                continue
            missing = [value for value in frame[column].dropna() if not Path(str(value)).is_file()]
            if missing:
                preview = missing[:5]
                issues.append(f"{column} has {len(missing)} missing files; examples: {preview}")
    return issues
