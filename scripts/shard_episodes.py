#!/usr/bin/env python3
"""Partition train/validation episodes into deterministic, restartable shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from activemap.models import EpisodeRecord


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shard_index(episode_id: str, *, seed: int, num_shards: int) -> int:
    digest = hashlib.sha256(f"{seed}|{episode_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % num_shards


def shard_episodes(
    source: Path, output_root: Path, *, num_shards: int, seed: int
) -> dict[str, Any]:
    if num_shards <= 1:
        raise ValueError("num-shards must be greater than one")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    output_root.mkdir(parents=True)
    paths = [output_root / f"shard-{index:02d}" / "episodes.jsonl" for index in range(num_shards)]
    for path in paths:
        path.parent.mkdir(parents=True)
    handles = [path.open("x", encoding="utf-8") for path in paths]
    counts = [Counter() for _ in range(num_shards)]
    aois: list[dict[str, set[str]]] = [defaultdict(set) for _ in range(num_shards)]
    seen: set[str] = set()
    try:
        with source.open(encoding="utf-8") as source_handle:
            for line_number, line in enumerate(source_handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if payload.get("split") not in {"train", "val"}:
                    continue
                try:
                    row = EpisodeRecord.model_validate(payload)
                except Exception as exc:
                    raise ValueError(f"invalid episode at line {line_number}: {exc}") from exc
                if row.episode_id in seen:
                    raise ValueError(f"duplicate episode ID: {row.episode_id}")
                seen.add(row.episode_id)
                index = shard_index(row.episode_id, seed=seed, num_shards=num_shards)
                handles[index].write(row.model_dump_json() + "\n")
                counts[index]["episodes"] += 1
                counts[index][f"split:{row.split}"] += 1
                counts[index][f"operation:{row.gt_edit.op.value}"] += 1
                aois[index][row.split].add(row.aoi_id or "__missing_aoi__")
    finally:
        for handle in handles:
            handle.close()

    if any(counter["episodes"] == 0 for counter in counts):
        raise ValueError("deterministic partition produced an empty shard")
    shards = []
    for index, path in enumerate(paths):
        digest = _sha256(path)
        shards.append(
            {
                "index": index,
                "path": str(path.resolve()),
                "sha256": digest,
                "counts": dict(sorted(counts[index].items())),
                "aoi_counts": {
                    split: len(values) for split, values in sorted(aois[index].items())
                },
            }
        )
    summary = {
        "schema_version": "episode-shards-v1",
        "source": str(source.resolve()),
        "source_sha256": _sha256(source),
        "num_shards": num_shards,
        "seed": seed,
        "episode_count": len(seen),
        "shards": shards,
        "test_assets_read": False,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes_jsonl", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            shard_episodes(
                args.episodes_jsonl,
                args.output_root,
                num_shards=args.num_shards,
                seed=args.seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
