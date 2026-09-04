#!/usr/bin/env python3
"""Merge disjoint frozen-feature shards with schema and ID validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("shards", nargs="+", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    summaries = [
        json.loads((root / "summary.json").read_text(encoding="utf-8"))
        for root in args.shards
    ]
    expected_indices = set(range(len(args.shards)))
    observed_indices = {int(row["shard"]["index"]) for row in summaries}
    if observed_indices != expected_indices or {
        int(row["shard"]["count"]) for row in summaries
    } != {len(args.shards)}:
        raise ValueError("feature shards are incomplete or duplicated")
    compatibility = {
        (
            row["schema_version"],
            row["feature_dim"],
            row["pooling"],
            row["utility_metadata"],
            row["source"]["sha256"],
            json.dumps(row.get("model"), sort_keys=True),
            json.dumps(row.get("adapter"), sort_keys=True),
            row["test_assets_read"],
        )
        for row in summaries
    }
    if len(compatibility) != 1:
        raise ValueError("feature shard summaries are incompatible")

    features = np.concatenate(
        [np.load(root / "features.npy") for root in args.shards], axis=0
    )
    records = []
    for root in args.shards:
        records.extend(
            json.loads(line)
            for line in (root / "records.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    ids = [str(row["example_id"]) for row in records]
    if len(records) != len(features) or len(ids) != len(set(ids)):
        raise ValueError("merged features are misaligned or duplicate IDs")

    args.output_dir.mkdir(parents=True)
    np.save(args.output_dir / "features.npy", features)
    with (args.output_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    template = summaries[0]
    summary = {
        **{key: value for key, value in template.items() if key != "shard"},
        "sample_count": len(records),
        "task_count": len({str(row["task_id"]) for row in records}),
        "positive_count": sum(bool(row["oracle_use_tool"]) for row in records),
        "positive_rate": sum(bool(row["oracle_use_tool"]) for row in records)
        / len(records),
        "merged_shards": len(args.shards),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
