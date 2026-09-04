#!/usr/bin/env python3
"""Build auditable train/validation re-entry states for recurrent policy smoke tests.

Natural no-change examples often terminate at selector step zero.  The output
keeps native post-acquisition states and adds a clearly labelled, affordable
counterfactual post-acquisition state for each eligible KEEP example.  It must
not be used for frozen test evaluation or reported as a primary benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from activemap.models import EditOperation
from activemap.training.data import load_selector_samples
from scripts.build_post_acquisition_mixed_sft import (
    counterfactual_keep_rollout_sample,
)


PROTOCOL = "post-acquisition-reentry-state-support-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_states(source: Path, *, splits: set[str]) -> tuple[list[object], dict[str, object]]:
    records = []
    native_counts: Counter[str] = Counter()
    synthetic_counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    for sample in load_selector_samples(source):
        if sample.split == "test":
            skipped["test_excluded"] += 1
            continue
        if sample.split not in splits:
            skipped["split_excluded"] += 1
            continue
        step = int(sample.metadata.get("oracle_step", -1))
        if step == 1:
            records.append(sample)
            native_counts[sample.split] += 1
        if step != 0:
            continue
        target = EditOperation(str(sample.metadata.get("gt_edit", sample.edit_type.value)))
        if target != EditOperation.KEEP:
            continue
        try:
            records.append(counterfactual_keep_rollout_sample(sample))
            synthetic_counts[sample.split] += 1
        except ValueError as exc:
            skipped[str(exc).split(":", 1)[0]] += 1
    if not records:
        raise ValueError("no re-entry states were produced")
    if not any(synthetic_counts.values()):
        raise ValueError("no eligible counterfactual KEEP states were produced")
    records.sort(key=lambda item: item.sample_id)
    duplicate_ids = [
        sample_id
        for sample_id, count in Counter(item.sample_id for item in records).items()
        if count > 1
    ]
    if duplicate_ids:
        raise ValueError(f"duplicate re-entry sample IDs: {duplicate_ids[:3]}")
    summary = {
        "schema_version": PROTOCOL,
        "source": str(source.resolve()),
        "source_sha256": _sha256(source),
        "included_splits": sorted(splits),
        "native_step_one_counts": dict(sorted(native_counts.items())),
        "counterfactual_keep_counts": dict(sorted(synthetic_counts.items())),
        "skipped_counts": dict(sorted(skipped.items())),
        "output_count": len(records),
        "test_assets_read": False,
        "intended_use": "train_val_recurrent_exploration_readiness_only",
        "forbidden_use": "frozen_test_or_primary_paper_benchmark",
    }
    return records, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--splits", default="train,val")
    args = parser.parse_args()
    splits = {part.strip() for part in args.splits.split(",") if part.strip()}
    if not splits or not splits <= {"train", "val"}:
        raise ValueError("--splits must be a non-empty subset of train,val")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    records, summary = build_states(args.source, splits=splits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
