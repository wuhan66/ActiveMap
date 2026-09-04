#!/usr/bin/env python3
"""Merge complete deterministic shards from policy-relative branch caching."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.cache_policy_relative_vlm_branches import policy_relative_metrics


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def interleave_shards(shards: list[list[dict[str, Any]]], total: int) -> list[dict[str, Any]]:
    if not shards:
        raise ValueError("no shards supplied")
    rows = []
    for index in range(total):
        shard_index = index % len(shards)
        offset = index // len(shards)
        if offset >= len(shards[shard_index]):
            raise ValueError("shards do not cover the declared pair count")
        rows.append(shards[shard_index][offset])
    if sum(len(shard) for shard in shards) != total:
        raise ValueError("shards contain rows beyond the declared pair count")
    ids = [str(row["example_id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("merged shards contain duplicate example IDs")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("shard_dir", nargs="+", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    summaries = [_json(root / "summary.json") for root in args.shard_dir]
    if any(summary.get("test_assets_read") is not False for summary in summaries):
        raise ValueError("a branch shard violates the frozen-test protocol")
    num_shards = int(summaries[0]["num_shards"])
    if len(summaries) != num_shards:
        raise ValueError("all declared shards must be supplied")
    by_index = {int(summary["shard_index"]): index for index, summary in enumerate(summaries)}
    if set(by_index) != set(range(num_shards)):
        raise ValueError("shard indices are incomplete or duplicated")
    ordered_summaries = [summaries[by_index[index]] for index in range(num_shards)]
    ordered_roots = [args.shard_dir[by_index[index]] for index in range(num_shards)]
    compatible = (
        "model",
        "adapter",
        "rollout_jsonl",
        "rollout_sha256",
        "seed",
        "max_new_tokens",
        "source_pair_count",
        "limited_pair_count",
        "num_shards",
    )
    for key in compatible:
        if len({json.dumps(summary[key], sort_keys=True) for summary in ordered_summaries}) != 1:
            raise ValueError(f"branch shards disagree on {key}")
    total = int(ordered_summaries[0]["limited_pair_count"])
    rows = interleave_shards([_jsonl(root / "traces.jsonl") for root in ordered_roots], total)

    args.output_dir.mkdir(parents=True)
    traces_path = args.output_dir / "traces.jsonl"
    with traces_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    first = ordered_summaries[0]
    summary = {
        "schema_version": "policy-relative-vlm-branch-cache-v1",
        **{key: first[key] for key in compatible if key != "num_shards"},
        "pair_count": len(rows),
        "sharding": {
            "num_shards": num_shards,
            "sources": [
                {
                    "path": str(root.resolve()),
                    "summary_sha256": _sha256(root / "summary.json"),
                }
                for root in ordered_roots
            ],
        },
        "metrics": policy_relative_metrics(rows),
        "direct_terminal_valid_rate": sum(row["direct_terminal_valid"] for row in rows)
        / len(rows),
        "post_terminal_valid_rate": sum(row["post_terminal_valid"] for row in rows)
        / len(rows),
        "trace_sha256": _sha256(traces_path),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "test_assets_read": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
