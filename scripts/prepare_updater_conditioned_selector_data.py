#!/usr/bin/env python3
"""Create a group-disjoint train/dev selector file from a C5 train state cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _group_is_dev(group: str, seed: int, fraction: float) -> bool:
    digest = hashlib.sha256(f"{seed}:{group}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < fraction


def _load(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"empty state cache: {path}")
    ids = [str(row["sample_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("state cache has duplicate sample IDs")
    if any(str(row.get("split")) != "train" for row in rows):
        raise ValueError("input must contain train-only state records")
    if any("source_episode" not in row.get("metadata", {}) for row in rows):
        raise ValueError("state records must include metadata.source_episode")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("states", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    args = parser.parse_args()
    if not 0.0 < args.dev_fraction < 0.5:
        raise ValueError("dev fraction must be in (0, 0.5)")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    rows = _load(args.states)
    output_rows: list[dict[str, Any]] = []
    groups: dict[str, str] = {}
    for row in rows:
        group = str(row["metadata"]["source_episode"])
        split = "val" if _group_is_dev(group, args.seed, args.dev_fraction) else "train"
        groups[group] = split
        output_rows.append({**row, "split": split})
    counts = {split: sum(value == split for value in groups.values()) for split in ("train", "val")}
    if not all(counts.values()):
        raise ValueError("group split produced an empty train or dev partition")

    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "selector_data.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {
        "schema_version": "updater-conditioned-selector-split-v1",
        "states": str(args.states.resolve()),
        "output": str(output.resolve()),
        "seed": args.seed,
        "dev_fraction": args.dev_fraction,
        "source_episode_counts": counts,
        "sample_counts": {
            split: sum(str(row["split"]) == split for row in output_rows)
            for split in ("train", "val")
        },
        "test_assets_read": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
