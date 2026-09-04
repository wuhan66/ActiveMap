#!/usr/bin/env python3
"""Merge deterministic train/validation episode shards without reopening test records."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from activemap.models import EpisodeRecord
from scripts.shard_episodes import shard_index


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def merge_episode_shards(shard_root: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    summary_path = shard_root / "summary.json"
    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if source_summary.get("test_assets_read") is not False:
        raise ValueError("source shards do not prove test isolation")
    num_shards = int(source_summary["num_shards"])
    seed = int(source_summary["seed"])
    paths = [shard_root / f"shard-{index:02d}" / "episodes.jsonl" for index in range(num_shards)]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("episode shards are incomplete")

    partial = output.with_suffix(output.suffix + ".partial")
    if partial.exists():
        raise FileExistsError(f"refusing to overwrite partial merge: {partial}")
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    sources = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("x", encoding="utf-8") as target:
        for index, path in enumerate(paths):
            sources.append({"path": str(path.resolve()), "sha256": _sha256(path)})
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        row = EpisodeRecord.model_validate_json(line)
                    except Exception as exc:
                        raise ValueError(f"invalid episode at {path}:{line_number}: {exc}") from exc
                    if row.split not in {"train", "val"}:
                        raise ValueError(f"forbidden episode split: {row.split}")
                    if shard_index(row.episode_id, seed=seed, num_shards=num_shards) != index:
                        raise ValueError(
                            "episode is in the wrong deterministic shard: "
                            f"{row.episode_id}"
                        )
                    if row.episode_id in seen:
                        raise ValueError(f"duplicate episode ID: {row.episode_id}")
                    seen.add(row.episode_id)
                    counts[f"split:{row.split}"] += 1
                    counts[f"operation:{row.gt_edit.op.value}"] += 1
                    target.write(row.model_dump_json() + "\n")
    if len(seen) != int(source_summary["episode_count"]):
        raise ValueError("merged episode count disagrees with shard summary")
    partial.replace(output)
    report = {
        "schema_version": "train-validation-episode-merge-v1",
        "episode_count": len(seen),
        "counts": dict(sorted(counts.items())),
        "sources": sources,
        "output": str(output.resolve()),
        "output_sha256": _sha256(output),
        "test_assets_read": False,
    }
    output.with_suffix(".merge_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shard_root", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    args = parser.parse_args()
    print(json.dumps(merge_episode_shards(args.shard_root, args.output_jsonl), indent=2))


if __name__ == "__main__":
    main()
