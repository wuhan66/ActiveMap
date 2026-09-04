#!/usr/bin/env python3
"""Merge immutable ArgoTweak feature shards in episode order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    import numpy as np

    rows = []
    arrays = []
    for path in args.inputs:
        part_rows = [
            json.loads(line)
            for line in (path / "records.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        part_features = np.load(path / "features.npy")
        if len(part_rows) != len(part_features):
            raise ValueError(f"record mismatch in {path}")
        rows.extend(part_rows)
        arrays.append(part_features)
    order = sorted(
        range(len(rows)),
        key=lambda index: (rows[index]["episode_id"], rows[index]["timestamp"]),
    )
    features = np.concatenate(arrays, axis=0)[order]
    rows = [rows[index] for index in order]
    args.output_dir.mkdir(parents=True)
    np.save(args.output_dir / "features.npy", features)
    (args.output_dir / "records.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "shards": len(args.inputs),
                "records": len(rows),
                "feature_dim": int(features.shape[1]),
                "test_assets_read": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
