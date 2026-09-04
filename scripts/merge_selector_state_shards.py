#!/usr/bin/env python3
"""Merge audited selector-state shards with duplicate and split checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from activemap.selector_records import SelectorSample


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def merge_shards(shard_root: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    partial = output.with_suffix(output.suffix + ".partial")
    if partial.exists():
        raise FileExistsError(f"refusing to overwrite partial merge: {partial}")
    inputs = sorted(shard_root.glob("shard-*/selector_states.jsonl"))
    if not inputs:
        raise FileNotFoundError(f"no selector state shards in {shard_root}")
    summary_path = shard_root / "summary.json"
    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if len(inputs) != int(source_summary["num_shards"]):
        raise ValueError("selector-state shards are incomplete")

    seen: set[str] = set()
    source_episodes: set[str] = set()
    counts: Counter[str] = Counter()
    sources = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("x", encoding="utf-8") as target:
        for path in inputs:
            sources.append({"path": str(path.resolve()), "sha256": _sha256(path)})
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        row = SelectorSample.model_validate_json(line)
                    except Exception as exc:
                        raise ValueError(f"invalid state at {path}:{line_number}: {exc}") from exc
                    if row.split not in {"train", "val"}:
                        raise ValueError(f"forbidden split in selector shard: {row.split}")
                    if row.sample_id in seen:
                        raise ValueError(f"duplicate selector state: {row.sample_id}")
                    seen.add(row.sample_id)
                    source_episodes.add(str(row.metadata["source_episode"]))
                    action = (
                        "STOP"
                        if row.target_index(allow_stop=True) == len(row.evidence_ids)
                        else "ACQUIRE"
                    )
                    counts[f"split:{row.split}"] += 1
                    counts[f"target:{action}"] += 1
                    counts[f"gt:{row.metadata.get('gt_edit', 'UNKNOWN')}:{action}"] += 1
                    target.write(row.model_dump_json() + "\n")
    if len(source_episodes) != int(source_summary["episode_count"]):
        raise ValueError("merged states do not cover every source episode")
    partial.replace(output)
    summary = {
        "schema_version": "selector-state-shard-merge-v1",
        "source_episode_count": len(source_episodes),
        "state_count": len(seen),
        "counts": dict(sorted(counts.items())),
        "sources": sources,
        "output": str(output.resolve()),
        "output_sha256": _sha256(output),
        "test_assets_read": False,
    }
    output.with_suffix(".merge_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shard_root", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    args = parser.parse_args()
    print(json.dumps(merge_shards(args.shard_root, args.output_jsonl), indent=2))


if __name__ == "__main__":
    main()
