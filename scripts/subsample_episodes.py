#!/usr/bin/env python3
"""Create a deterministic train/validation episode subset without reading test."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from activemap.models import EpisodeRecord

try:
    from scripts.audit_episode_support import audit_episode_support, load_episodes
except ModuleNotFoundError:  # Direct execution adds scripts/, not the project root.
    from audit_episode_support import audit_episode_support, load_episodes


def _rank(seed: int, row: EpisodeRecord) -> str:
    return hashlib.sha256(f"{seed}|{row.episode_id}".encode()).hexdigest()


def select_rows(
    rows: list[EpisodeRecord], *, per_operation: int, seed: int
) -> list[EpisodeRecord]:
    if per_operation <= 0:
        raise ValueError("per-operation must be positive")
    allowed = [row for row in rows if row.split in {"train", "val"}]
    grouped: dict[tuple[str, str], list[EpisodeRecord]] = defaultdict(list)
    for row in allowed:
        grouped[(row.split, row.gt_edit.op.value)].append(row)

    selected: list[EpisodeRecord] = []
    for key in sorted(grouped):
        candidates = grouped[key]
        # Round-robin across AOIs before filling from the deterministic global rank.
        by_aoi: dict[str, list[EpisodeRecord]] = defaultdict(list)
        for row in candidates:
            by_aoi[row.aoi_id or "__missing_aoi__"].append(row)
        for values in by_aoi.values():
            values.sort(key=lambda row: _rank(seed, row))
        aoi_order = sorted(
            by_aoi,
            key=lambda value: hashlib.sha256(f"{seed}|{value}".encode()).hexdigest(),
        )
        bucket: list[EpisodeRecord] = []
        depth = 0
        while len(bucket) < min(per_operation, len(candidates)):
            added = False
            for aoi in aoi_order:
                if depth < len(by_aoi[aoi]):
                    bucket.append(by_aoi[aoi][depth])
                    added = True
                    if len(bucket) == min(per_operation, len(candidates)):
                        break
            if not added:
                break
            depth += 1
        selected.extend(bucket)
    return sorted(selected, key=lambda row: (row.split, row.aoi_id, _rank(seed, row)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes_jsonl", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument("--per-operation", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.output_jsonl.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_jsonl}")
    source = load_episodes(
        args.episodes_jsonl, allowed_splits={"train", "val"}
    )
    selected = select_rows(source, per_operation=args.per_operation, seed=args.seed)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("x", encoding="utf-8") as handle:
        for row in selected:
            handle.write(row.model_dump_json() + "\n")
    audit = audit_episode_support(selected, args.output_jsonl)
    summary = {
        **audit,
        "schema_version": "episode-subset-v1",
        "source": str(args.episodes_jsonl.resolve()),
        "source_train_val_episode_count": len(source),
        "per_operation": args.per_operation,
        "seed": args.seed,
        "selection_counts": dict(
            sorted(Counter(f"{row.split}:{row.gt_edit.op.value}" for row in selected).items())
        ),
        "test_assets_read": False,
    }
    args.output_jsonl.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
