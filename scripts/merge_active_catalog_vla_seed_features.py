#!/usr/bin/env python3
"""Pool frozen VLA features across adapter seeds with provenance checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_source(value: str) -> tuple[int, Path]:
    seed, separator, path = value.partition("=")
    if not separator or not seed or not path:
        raise argparse.ArgumentTypeError("source must be SEED=PATH")
    return int(seed), Path(path)


def merge_sources(
    output_dir: Path, sources: list[tuple[int, Path]]
) -> dict[str, Any]:
    roots = dict(sources)
    if len(roots) != len(sources) or len(roots) < 2:
        raise ValueError("at least two unique model seeds are required")
    summaries = {
        seed: json.loads((root / "summary.json").read_text(encoding="utf-8"))
        for seed, root in sorted(roots.items())
    }
    compatibility = {
        (
            row["schema_version"],
            row["feature_dim"],
            row["pooling"],
            row["utility_metadata"],
            row["source"]["sha256"],
            row["test_assets_read"],
        )
        for row in summaries.values()
    }
    if len(compatibility) != 1:
        raise ValueError("seed feature summaries are incompatible")

    feature_blocks = []
    records = []
    for seed, root in sorted(roots.items()):
        features = np.load(root / "features.npy")
        rows = [
            json.loads(line)
            for line in (root / "records.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        if len(features) != len(rows):
            raise ValueError(f"feature alignment failed for seed {seed}")
        feature_blocks.append(features)
        records.extend(
            {
                **row,
                "example_id": f"{seed}:{row['example_id']}",
                "model_seed": seed,
            }
            for row in rows
        )
    features = np.concatenate(feature_blocks, axis=0)
    ids = [str(row["example_id"]) for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("cross-seed example IDs are not unique")

    output_dir.mkdir(parents=True)
    np.save(output_dir / "features.npy", features)
    with (output_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    template = next(iter(summaries.values()))
    summary = {
        "schema_version": "active-catalog-vla-cross-seed-features-v1",
        "sample_count": len(records),
        "task_count": len({str(row["task_id"]) for row in records}),
        "positive_count": sum(bool(row["oracle_use_tool"]) for row in records),
        "positive_rate": sum(bool(row["oracle_use_tool"]) for row in records)
        / len(records),
        "feature_dim": int(features.shape[1]),
        "feature_dtype": str(features.dtype),
        "pooling": template["pooling"],
        "utility_metadata": template["utility_metadata"],
        "source": template["source"],
        "model_seeds": sorted(roots),
        "seed_sources": {
            str(seed): {
                "path": str(roots[seed].resolve()),
                "model": summaries[seed].get("model"),
                "adapter": summaries[seed].get("adapter"),
            }
            for seed in sorted(roots)
        },
        "grouping_contract": (
            "task_id is shared across adapter seeds so one task remains in one OOF fold"
        ),
        "test_assets_read": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source", action="append", type=parse_source, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    print(json.dumps(merge_sources(args.output_dir, args.source), indent=2))


if __name__ == "__main__":
    main()
