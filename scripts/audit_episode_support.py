#!/usr/bin/env python3
"""Audit task, geography, edit, and evidence support in episode JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def load_episodes(
    path: Path, *, allowed_splits: set[str] | None = None
) -> list[EpisodeRecord]:
    rows: list[EpisodeRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                if allowed_splits is not None and payload.get("split") not in allowed_splits:
                    continue
                rows.append(EpisodeRecord.model_validate(payload))
            except Exception as exc:
                raise ValueError(f"invalid episode at {path}:{line_number}: {exc}") from exc
    if not rows:
        raise ValueError(f"empty episode file: {path}")
    return rows


def audit_episode_support(rows: list[EpisodeRecord], source: Path) -> dict[str, Any]:
    episode_ids = [row.episode_id for row in rows]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("episode IDs are not unique")
    splits = {row.split for row in rows}
    if not splits <= {"train", "val", "test"}:
        raise ValueError(f"unsupported splits: {sorted(splits)}")

    split_counts: Counter[str] = Counter()
    operation_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    aoi_by_split: dict[str, set[str]] = defaultdict(set)
    missing_aoi_count = 0
    evidence_counts: list[int] = []
    for row in rows:
        split_counts[row.split] += 1
        operation_counts[f"{row.split}:{row.gt_edit.op.value}"] += 1
        source_counts[row.source_dataset] += 1
        if row.aoi_id is None:
            missing_aoi_count += 1
        aoi_by_split[row.split].add(row.aoi_id or "__missing_aoi__")
        evidence_counts.append(len(row.evidence_catalog))

    split_names = sorted(splits)
    aoi_overlap = {
        f"{left}:{right}": len(aoi_by_split[left] & aoi_by_split[right])
        for index, left in enumerate(split_names)
        for right in split_names[index + 1 :]
    }
    sorted_evidence = sorted(evidence_counts)
    percentile_index = min(
        len(sorted_evidence) - 1, math.floor(0.95 * len(sorted_evidence))
    )
    return {
        "schema_version": "episode-support-audit-v1",
        "source": str(source.resolve()),
        "source_sha256": _sha256(source),
        "episode_count": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "operation_counts": dict(sorted(operation_counts.items())),
        "source_dataset_counts": dict(sorted(source_counts.items())),
        "aoi_counts": {
            split: len(values) for split, values in sorted(aoi_by_split.items())
        },
        "aoi_overlap": aoi_overlap,
        "missing_aoi_count": missing_aoi_count,
        "evidence_per_episode": {
            "minimum": min(evidence_counts),
            "mean": math.fsum(evidence_counts) / len(evidence_counts),
            "p95": sorted_evidence[percentile_index],
            "maximum": max(evidence_counts),
        },
        "test_assets_read": "test" in splits,
        "passed": missing_aoi_count == 0 and all(
            value == 0 for value in aoi_overlap.values()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes_jsonl", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--splits", default="train,val")
    args = parser.parse_args()
    splits = {value.strip() for value in args.splits.split(",") if value.strip()}
    if not splits or not splits <= {"train", "val", "test"}:
        raise ValueError("--splits must contain train, val, or test")
    summary = audit_episode_support(
        load_episodes(args.episodes_jsonl, allowed_splits=splits), args.episodes_jsonl
    )
    summary["requested_splits"] = sorted(splits)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
