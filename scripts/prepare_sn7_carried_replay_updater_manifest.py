"""Build a group-disjoint updater manifest for carried-map replay repair."""

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
        raise ValueError("carried replay requires non-empty train and validation partitions")
    if any(sample.dataset_name != "sn7_carried_replay" for sample in replay_samples):
        raise ValueError("replay manifest contains an unexpected dataset name")
    if any(
        sample.source_metadata.get("replay_policy") != "executed_noop_carry"
        for sample in replay_samples
    ):
        raise ValueError("replay manifest does not use executed-history carry supervision")

    replay_train_groups = {_group_key(sample) for sample in replay_train}
    replay_val_groups = {_group_key(sample) for sample in replay_val}
    if replay_train_groups & replay_val_groups:
        raise ValueError("carried replay train and validation groups overlap")
    retained_canonical_train = [
        sample for sample in canonical_train if _group_key(sample) not in replay_val_groups
    ]
    if not retained_canonical_train:
        raise ValueError("replay validation groups removed every canonical training sample")

    merged = [*retained_canonical_train, *replay_train, *replay_val]
    sample_ids = [sample.sample_id for sample in merged]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("duplicate sample IDs across canonical and replay manifests")
    train_groups = {_group_key(sample) for sample in retained_canonical_train}
    train_groups.update(replay_train_groups)
    if train_groups & replay_val_groups:
        raise ValueError("training samples overlap replay validation groups")

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        "".join(sample.model_dump_json() + "\n" for sample in merged), encoding="utf-8"
    )
    summary = {
        "schema_version": "sn7-carried-replay-updater-manifest-v1",
        "canonical_manifest": str(args.canonical_manifest.resolve()),
        "replay_manifest": str(args.replay_manifest.resolve()),
        "output_manifest": str(args.output_manifest.resolve()),
        "test_assets_read": False,
        "canonical_train_total": len(canonical_train),
        "canonical_train_retained": len(retained_canonical_train),
        "canonical_train_excluded_for_replay_validation": (
            len(canonical_train) - len(retained_canonical_train)
        ),
        "replay_train": len(replay_train),
        "replay_val": len(replay_val),
        "combined_split_counts": _count(merged, "split"),
        "combined_dataset_counts": _count(merged, "dataset_name"),
        "combined_operation_counts": _count(merged, "edit_type"),
        "replay_train_group_count": len(replay_train_groups),
        "replay_val_group_count": len(replay_val_groups),
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
