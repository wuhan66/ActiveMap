#!/usr/bin/env python3
"""Combine a source-region training pool with an external validation region."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _paths(root: Path, pattern: str) -> list[Path]:
    paths = sorted(root.glob(pattern))
    if len(paths) != 3:
        raise ValueError(f"expected three seed files below {root}, found {len(paths)}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("germany_root", type=Path)
    parser.add_argument("external_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--germany-pattern", default="changeformer_weight5_seed*/per_candidate.jsonl"
    )
    parser.add_argument("--external-pattern", default="seed*/per_candidate.jsonl")
    args = parser.parse_args()

    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    germany_paths = _paths(args.germany_root, args.germany_pattern)
    external_paths = _paths(args.external_root, args.external_pattern)
    args.output_root.mkdir(parents=True)

    summaries = []
    for index, (germany_path, external_path) in enumerate(
        zip(germany_paths, external_paths), start=1
    ):
        germany_train = [row for row in _read_jsonl(germany_path) if row["split"] == "train"]
        external_val = _read_jsonl(external_path)
        if not germany_train:
            raise ValueError(f"no Germany train rows in {germany_path}")
        if not external_val or any(row["split"] != "val" for row in external_val):
            raise ValueError(f"external rows must be validation-only: {external_path}")
        train_ids = {str(row["sample_id"]) for row in germany_train}
        val_ids = {str(row["sample_id"]) for row in external_val}
        overlap = train_ids & val_ids
        if overlap:
            raise ValueError(f"source/external sample overlap: {sorted(overlap)[:3]}")
        output = args.output_root / f"seed{index:02d}_per_candidate.jsonl"
        with output.open("x", encoding="utf-8") as handle:
            for row in germany_train + external_val:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        summaries.append(
            {
                "seed_index": index,
                "germany_train_candidates": len(germany_train),
                "external_val_candidates": len(external_val),
                "germany_train_episodes": len(train_ids),
                "external_val_episodes": len(val_ids),
                "path": str(output),
            }
        )

    result = {
        "schema_version": "activemap-spacenet8-external-validation-combine-v1",
        "source_region": "germany",
        "external_region": "louisiana-east",
        "source_split": "train",
        "external_split": "val",
        "seed_count": len(summaries),
        "seeds": summaries,
        "test_assets_read": False,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
