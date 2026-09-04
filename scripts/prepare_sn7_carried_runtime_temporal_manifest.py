"""Merge temporal carried-state replay with group-disjoint canonical training."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from activemap.updater_records import UpdaterSample, load_updater_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("canonical_manifest", type=Path)
    parser.add_argument("replay_manifest", type=Path)
    parser.add_argument("output_manifest", type=Path)
    return parser.parse_args()


def _group_key(sample: UpdaterSample) -> tuple[str, str]:
    if sample.aoi_id is None or sample.object_id is None:
        raise ValueError(f"sample {sample.sample_id} lacks an AOI or object identifier")
    return sample.aoi_id, sample.object_id


def _count(samples: list[UpdaterSample], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(getattr(sample, field)) for sample in samples).items()))


def prepare_manifest(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_manifest.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_manifest}")
    canonical_train = load_updater_samples(args.canonical_manifest, split="train")
    replay_samples = load_updater_samples(args.replay_manifest)
    replay_train = [sample for sample in replay_samples if sample.split == "train"]
    replay_val = [sample for sample in replay_samples if sample.split == "val"]
    if not replay_train or not replay_val:
        raise ValueError("temporal carried replay requires train and validation samples")
    if any(
        sample.dataset_name != "sn7_carried_runtime_temporal"
        or sample.source_metadata.get("replay_policy") != "carried-runtime-temporal-v2"
        or sample.source_metadata.get("target_alignment") != "current_anchor_only"
        for sample in replay_samples
    ):
        raise ValueError("replay manifest violates the temporal-v2 contract")
    if any(sample.prior_image_path is None for sample in [*canonical_train, *replay_samples]):
        raise ValueError("every temporal sample must provide prior_image_path")

    replay_train_groups = {_group_key(sample) for sample in replay_train}
    replay_val_groups = {_group_key(sample) for sample in replay_val}
    if replay_train_groups & replay_val_groups:
        raise ValueError("temporal replay train and validation groups overlap")
    retained_canonical = [
        sample for sample in canonical_train if _group_key(sample) not in replay_val_groups
    ]
    train_groups = {_group_key(sample) for sample in retained_canonical}
    train_groups.update(replay_train_groups)
    if train_groups & replay_val_groups:
        raise ValueError("training samples overlap temporal replay validation groups")

    merged = [*retained_canonical, *replay_train, *replay_val]
    sample_ids = [sample.sample_id for sample in merged]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("duplicate sample IDs across canonical and replay manifests")
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        "".join(sample.model_dump_json() + "\n" for sample in merged), encoding="utf-8"
    )
    summary = {
        "schema_version": "sn7-carried-runtime-temporal-manifest-v2",
        "canonical_manifest": str(args.canonical_manifest.resolve()),
        "replay_manifest": str(args.replay_manifest.resolve()),
        "output_manifest": str(args.output_manifest.resolve()),
        "test_assets_read": False,
        "temporal_pair_input": True,
        "target_alignment": "current_anchor_only",
        "canonical_train_total": len(canonical_train),
        "canonical_train_retained": len(retained_canonical),
        "canonical_train_excluded_for_replay_validation": (
            len(canonical_train) - len(retained_canonical)
        ),
        "replay_train": len(replay_train),
        "replay_val": len(replay_val),
        "combined_split_counts": _count(merged, "split"),
        "combined_dataset_counts": _count(merged, "dataset_name"),
        "combined_operation_counts": _count(merged, "edit_type"),
        "train_validation_group_overlap": 0,
    }
    args.output_manifest.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    print(json.dumps(prepare_manifest(parse_args()), indent=2))


if __name__ == "__main__":
    main()
