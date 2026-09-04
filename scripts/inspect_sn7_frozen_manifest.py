from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _month_index(value: str) -> int:
    year, month = value.split("_")
    return int(year) * 12 + int(month)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()

    frame = pd.read_parquet(args.manifest)
    print(f"rows={len(frame)}")
    print(f"columns={','.join(frame.columns)}")
    print(f"aois={frame['aoi_id'].nunique()}")
    print(frame.groupby("aoi_id").size().to_string())
    adjacent_pairs = 0
    for _, group in frame.groupby("aoi_id"):
        timestamps = sorted(str(value) for value in group["timestamp"])
        adjacent_pairs += sum(
            _month_index(new) - _month_index(old) == 1
            for old, new in zip(timestamps, timestamps[1:])
        )
    print(f"adjacent_pairs={adjacent_pairs}")


if __name__ == "__main__":
    main()
