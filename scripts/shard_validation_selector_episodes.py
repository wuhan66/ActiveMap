#!/usr/bin/env python3
"""Create immutable validation-only episode shards for parallel oracle inference.

The source manifest may contain both train and validation records.  A candidate
headroom gate consumes validation states only, so this utility copies the exact
validation JSONL records into deterministic shards without touching any assets.
It writes a receipt that binds every shard to the source manifest hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shard_validation_episodes(
    episodes_path: Path,
    output_dir: Path,
    *,
    shard_count: int,
) -> dict[str, Any]:
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite shard directory: {output_dir}")
    partial_dir = output_dir.with_name(f"{output_dir.name}.partial")
    if partial_dir.exists():
        raise FileExistsError(f"refusing to overwrite partial shard directory: {partial_dir}")

    partial_dir.mkdir(parents=True)
    shard_paths = [
        partial_dir / f"episodes_val_shard_{index:02d}.jsonl" for index in range(shard_count)
    ]
    handles = [path.open("w", encoding="utf-8") for path in shard_paths]
    total_rows = 0
    train_rows = 0
    validation_rows = 0
    test_rows = 0
    shard_rows = [0] * shard_count
    seen_episode_ids: set[str] = set()
    try:
        with episodes_path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                total_rows += 1
                try:
                    payload = json.loads(line)
                    split = str(payload["split"])
                    episode_id = str(payload["episode_id"])
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise ValueError(f"invalid episode record at line {line_number}") from exc
                if not episode_id:
                    raise ValueError(f"empty episode_id at line {line_number}")
                if split == "train":
                    train_rows += 1
                    continue
                if split == "test":
                    test_rows += 1
                    continue
                if split != "val":
                    raise ValueError(f"unexpected split {split!r} at line {line_number}")
                if episode_id in seen_episode_ids:
                    raise ValueError(f"duplicate validation episode_id {episode_id!r}")
                seen_episode_ids.add(episode_id)
                shard_index = validation_rows % shard_count
                handles[shard_index].write(line if line.endswith("\n") else line + "\n")
                validation_rows += 1
                shard_rows[shard_index] += 1
    finally:
        for handle in handles:
            handle.close()

    if test_rows:
        raise PermissionError(
            "validation headroom sharding refuses a source manifest with test rows"
        )
    if not validation_rows:
        raise ValueError("source manifest contains no validation episodes")

    receipt = {
        "schema": "activemap-validation-selector-episode-shards-v1",
        "source_episodes": str(episodes_path.resolve()),
        "source_episodes_sha256": sha256_file(episodes_path),
        "source_rows": {
            "total": total_rows,
            "train": train_rows,
            "val": validation_rows,
            "test": test_rows,
        },
        "shard_count": shard_count,
        "shards": [
            {
                "index": index,
                "path": str((output_dir / shard_paths[index].name).resolve()),
                "rows": shard_rows[index],
                "sha256": sha256_file(shard_paths[index]),
            }
            for index in range(shard_count)
        ],
        "test_assets_read": False,
    }
    (partial_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(partial_dir, output_dir)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--shards", type=int, required=True)
    args = parser.parse_args()
    if not args.episodes.is_file():
        raise FileNotFoundError(args.episodes)
    result = shard_validation_episodes(
        args.episodes, args.output_dir, shard_count=args.shards
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
