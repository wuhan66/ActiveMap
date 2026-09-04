"""Materialize train-internal updater samples from true carried-map replay."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from activemap.oracle.updater_counterfactual import _month_index, _read_candidate, load_episodes
from activemap.training.updater_data import _residual_geometry_delta, _residual_operation
from activemap.updater_records import UpdaterSample
from scripts.evaluate_online_persistent_map_maintenance import (
    _episode_geometry,
    _remap_episode_assets,
    _same_state,
    build_contiguous_chains,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--image-channels", type=int, default=3)
    parser.add_argument("--minimum-chain-length", type=int, default=2)
    parser.add_argument("--continuity-tolerance", type=float, default=1e-6)
    parser.add_argument("--validation-chain-index", type=int, default=20)
    parser.add_argument("--max-chains", type=int)
    parser.add_argument("--max-evidence-per-step", type=int, default=16)
    parser.add_argument("--asset-root-map", action="append", default=[])
    return parser.parse_args()


def _parse_asset_root_maps(values: list[str]) -> tuple[tuple[Path, Path], ...]:
    mappings: list[tuple[Path, Path]] = []
    for value in values:
        source, separator, destination = value.partition("=")
        if not separator or not source or not destination:
            raise ValueError("asset root mappings must use SOURCE=DESTINATION")
        mappings.append((Path(source), Path(destination)))
    return tuple(mappings)


def _observable_items(episode: Any, *, limit: int) -> list[Any]:
    if limit < 1:
        raise ValueError("max_evidence_per_step must be positive")
    anchor_timestamp = episode.anchor_timestamp or episode.evidence_catalog[-1].timestamp
    anchor_month = _month_index(anchor_timestamp)
    visible = [
        item for item in episode.evidence_catalog if _month_index(item.timestamp) <= anchor_month
    ]
    if not visible:
        raise ValueError(f"episode {episode.episode_id} has no causally visible evidence")
    direct = next(
        (item for item in visible if item.timestamp == anchor_timestamp),
        visible[-1],
    )
    remaining = [item for item in visible if item.evidence_id != direct.evidence_id]
    remaining.sort(key=lambda item: (str(item.timestamp), item.evidence_id), reverse=True)
    return [direct, *remaining[: limit - 1]]


def _write_array(path: Path, value: np.ndarray, *, image: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image:
        encoded = np.rint(np.clip(value, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        encoded = np.rint(np.clip(value, 0.0, 1.0)).astype(np.uint8)
    np.save(path, encoded)


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.image_size < 16 or args.image_channels < 1:
        raise ValueError("image size and channels must be positive")
    episodes = _remap_episode_assets(
        [episode for episode in load_episodes(args.episodes) if episode.split == "train"],
        _parse_asset_root_maps(args.asset_root_map),
    )
    chains = build_contiguous_chains(
        episodes,
        minimum_length=args.minimum_chain_length,
        continuity_tolerance=args.continuity_tolerance,
    )
    if args.max_chains is not None:
        chains = chains[: args.max_chains]
    if len(chains) <= args.validation_chain_index:
        raise ValueError("not enough chronological chains for a held-out validation partition")

    manifest = args.output_dir / "updater_samples.jsonl"
    data_dir = args.output_dir / "data"
    records: list[UpdaterSample] = []
    split_counts: Counter[str] = Counter()
    operation_counts: Counter[str] = Counter()
    canonical_match_counts: Counter[str] = Counter()
    for chain_index, chain in enumerate(chains):
        split = "train" if chain_index < args.validation_chain_index else "val"
        carried_prior = _episode_geometry(chain[0].prior_geometry)
        for step, episode in enumerate(chain):
            target_geometry = _episode_geometry(episode.target_geometry)
            canonical_prior = _episode_geometry(episode.prior_geometry)
            carried_matches_canonical = _same_state(
                carried_prior, canonical_prior, args.continuity_tolerance
            )
            canonical_match_counts[
                "matches" if carried_matches_canonical else "differs"
            ] += 1
            for evidence_index, item in enumerate(
                _observable_items(episode, limit=args.max_evidence_per_step)
            ):
                image, prior, target, valid, _ = _read_candidate(
                    item,
                    prior_geometry=carried_prior,
                    target_geometry=target_geometry,
                    image_size=args.image_size,
                    image_channels=args.image_channels,
                    road_width_source_pixels=(
                        float(episode.metadata["road_width_source_pixels"])
                        if episode.metadata.get("road_width_source_pixels") is not None
                        else None
                    ),
                )
                operation = _residual_operation(prior, target, valid)
                sample_id = (
                    f"{episode.episode_id}__c{chain_index:05d}__s{step:04d}"
                    f"__e{evidence_index:02d}"
                )
                prefix = data_dir / split / sample_id
                image_path = prefix.with_name(f"{prefix.name}_image.npy")
                prior_path = prefix.with_name(f"{prefix.name}_prior.npy")
                target_path = prefix.with_name(f"{prefix.name}_target.npy")
                valid_path = prefix.with_name(f"{prefix.name}_valid.npy")
                _write_array(image_path, image, image=True)
                _write_array(prior_path, prior, image=False)
                _write_array(target_path, target, image=False)
                _write_array(valid_path, valid, image=False)
                records.append(
                    UpdaterSample(
                        sample_id=sample_id,
                        aoi_id=str(episode.aoi_id),
                        split=split,
                        image_path=image_path.relative_to(args.output_dir).as_posix(),
                        prior_mask_path=prior_path.relative_to(args.output_dir).as_posix(),
                        target_mask_path=target_path.relative_to(args.output_dir).as_posix(),
                        valid_mask_path=valid_path.relative_to(args.output_dir).as_posix(),
                        edit_type=operation,
                        geometry_delta=_residual_geometry_delta(prior, target).tolist(),
                        object_id=str(episode.hypothesis.object_id),
                        clear_fraction=float(np.mean(valid >= 0.5)),
                        quality_source="sn7_train_internal_carried_replay_v1",
                        dataset_name="sn7_carried_replay",
                        supervision_type="real_temporal",
                        source_metadata={
                            "chain_index": chain_index,
                            "step": step,
                            "evidence_id": item.evidence_id,
                            "anchor_timestamp": str(episode.anchor_timestamp),
                            "carried_matches_canonical": carried_matches_canonical,
                            "replay_policy": "executed_noop_carry",
                        },
                    )
                )
                split_counts[split] += 1
                operation_counts[f"{split}:{operation.value}"] += 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "".join(record.model_dump_json() + "\n" for record in records), encoding="utf-8"
    )
    summary = {
        "schema_version": "sn7-carried-replay-updater-dataset-v1",
        "episodes": str(args.episodes.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "test_assets_read": False,
        "chain_count": len(chains),
        "validation_chain_index": args.validation_chain_index,
        "max_evidence_per_step": args.max_evidence_per_step,
        "sample_counts": dict(sorted(split_counts.items())),
        "residual_operation_counts": dict(sorted(operation_counts.items())),
        "carried_vs_canonical_state_counts": dict(sorted(canonical_match_counts.items())),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    summary = build_dataset(parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
