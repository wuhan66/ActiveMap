#!/usr/bin/env python3
"""Deterministically sample AOI-balanced step-0 states for on-policy rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from activemap.selector_records import SelectorSample


def sample_states(source: Path, output: Path, *, split: str, count: int, seed: int) -> dict:
    if split not in {"train", "val"} or count < 1:
        raise ValueError("invalid online state sample settings")
    groups = defaultdict(list)
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = SelectorSample.model_validate_json(line)
        if row.split == split and int(row.metadata.get("oracle_step", -1)) == 0:
            groups[str(row.metadata["aoi_id"])].append(row)
    if not groups:
        raise ValueError(f"no {split} step-0 states")
    available = sum(len(rows) for rows in groups.values())
    rng = random.Random(seed)
    for rows in groups.values():
        rng.shuffle(rows)
    selected = []
    group_ids = sorted(groups)
    while len(selected) < count:
        progressed = False
        for aoi in group_ids:
            if groups[aoi]:
                selected.append(groups[aoi].pop())
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
    if len(selected) != min(count, available):
        raise RuntimeError("online sampling terminated unexpectedly")
    identities = {
        (str(row.metadata["source_episode"]), float(row.metadata["budget"]))
        for row in selected
    }
    if len(identities) != len(selected):
        raise ValueError("sampled online states contain duplicate episode-budget keys")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(row.model_dump_json() + "\n" for row in selected), encoding="utf-8")
    summary = {
        "schema_version": "active-catalog-online-state-sample-v1",
        "split": split,
        "requested": count,
        "selected": len(selected),
        "aoi_count": len({str(row.metadata["aoi_id"]) for row in selected}),
        "seed": seed,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "test_assets_read": False,
    }
    output.with_suffix(output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(sample_states(
        args.source, args.output, split=args.split, count=args.count, seed=args.seed
    ), indent=2))


if __name__ == "__main__":
    main()
