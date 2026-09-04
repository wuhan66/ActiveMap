"""Build aligned temporal updater samples from train-internal carried states."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from activemap.models import EditOperation, EpisodeRecord
from activemap.updater_records import UpdaterSample
from scripts.evaluate_online_persistent_map_maintenance import (
    _episode_geometry,
    _remap_episode_assets,
    _same_state,
    build_contiguous_chains,
)

STATE_POLICIES = ("teacher_corrected", "one_step_defer", "defer_from_start")


def load_episodes(path: Path) -> list[EpisodeRecord]:
    return [
        EpisodeRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _residual_operation(
    prior: np.ndarray, target: np.ndarray, valid: np.ndarray
) -> EditOperation:
    prior_foreground = (prior >= 0.5) & (valid >= 0.5)
    target_foreground = (target >= 0.5) & (valid >= 0.5)
    adds = bool(np.any(target_foreground & ~prior_foreground))
    removes = bool(np.any(prior_foreground & ~target_foreground))
    if adds and removes:
        return EditOperation.RESHAPE
    if adds:
        return EditOperation.ADD
    if removes:
        return EditOperation.DELETE
    return EditOperation.KEEP


def _mask_bounds(mask: np.ndarray) -> np.ndarray:
    foreground = np.argwhere(mask >= 0.5)
    if foreground.size == 0:
        return np.zeros(4, dtype=np.float32)
    height, width = mask.shape
    y1, x1 = foreground.min(axis=0)
    y2, x2 = foreground.max(axis=0) + 1
    return np.asarray(
        (x1 / width, y1 / height, x2 / width, y2 / height), dtype=np.float32
    )


def _residual_geometry_delta(prior: np.ndarray, target: np.ndarray) -> np.ndarray:
    target_bounds = _mask_bounds(target)
    prior_bounds = _mask_bounds(prior)
    return np.concatenate((target_bounds, target_bounds - prior_bounds)).astype(np.float32)


def _read_paired_candidate(item: Any, **kwargs: Any) -> tuple[Any, ...]:
    from activemap.oracle.updater_counterfactual import _read_candidate

    return _read_candidate(item, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--minimum-chain-length", type=int, default=2)
    parser.add_argument("--continuity-tolerance", type=float, default=1e-6)
    parser.add_argument("--validation-chain-index", type=int, default=20)
    parser.add_argument("--max-chains", type=int)
    parser.add_argument(
        "--state-policy",
        action="append",
        choices=STATE_POLICIES,
        dest="state_policies",
    )
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


def _anchor(episode: Any) -> Any:
    timestamp = episode.anchor_timestamp or episode.evidence_catalog[-1].timestamp
    return next(
        (item for item in episode.evidence_catalog if item.timestamp == timestamp),
        episode.evidence_catalog[-1],
    )


def _temporal_item(chain: list[Any], step: int) -> tuple[Any | None, str]:
    current = _anchor(chain[step])
    if current.prior_image_path is not None:
        return current, "episode_metadata"
    if step == 0:
        return None, "missing_chain_predecessor"
    previous = _anchor(chain[step - 1])
    return (
        current.model_copy(
            update={
                "prior_image_path": previous.image_path,
                "prior_udm_path": previous.udm_path,
                "prior_timestamp": previous.timestamp,
            }
        ),
        "previous_chain_anchor",
    )


def _state_variants(chain: list[Any], step: int) -> list[tuple[Any, tuple[str, ...]]]:
    candidates = {
        "teacher_corrected": _episode_geometry(chain[step].prior_geometry),
        "one_step_defer": _episode_geometry(
            chain[step - 1].prior_geometry if step > 0 else chain[0].prior_geometry
        ),
        "defer_from_start": _episode_geometry(chain[0].prior_geometry),
    }
    variants: list[tuple[Any, list[str]]] = []
    for policy, geometry in candidates.items():
        matched = next(
            (
                entry
                for entry in variants
                if _same_state(entry[0], geometry, tolerance=1e-9)
            ),
            None,
        )
        if matched is None:
            variants.append((geometry, [policy]))
        else:
            matched[1].append(policy)
    return [(geometry, tuple(policies)) for geometry, policies in variants]


def _write_array(path: Path, value: np.ndarray, *, image: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        np.rint(np.clip(value, 0.0, 1.0) * 255.0).astype(np.uint8)
        if image
        else np.rint(np.clip(value, 0.0, 1.0)).astype(np.uint8)
    )
    np.save(path, encoded)


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.image_size < 16:
        raise ValueError("image size must be at least 16")
    policies = tuple(args.state_policies or STATE_POLICIES)
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
        raise ValueError("not enough chronological chains for held-out validation")

    records: list[UpdaterSample] = []
    split_counts: Counter[str] = Counter()
    operation_counts: Counter[str] = Counter()
    policy_counts: Counter[str] = Counter()
    pair_source_counts: Counter[str] = Counter()
    missing_pair_count = 0
    data_dir = args.output_dir / "data"
    for chain_index, chain in enumerate(chains):
        split = "train" if chain_index < args.validation_chain_index else "val"
        for step, episode in enumerate(chain):
            item, pair_source = _temporal_item(chain, step)
            if item is None:
                missing_pair_count += 1
                continue
            pair_source_counts[f"{split}:{pair_source}"] += 1
            target_geometry = _episode_geometry(episode.target_geometry)
            for variant_index, (carried_prior, sources) in enumerate(
                _state_variants(chain, step)
            ):
                selected_sources = tuple(source for source in sources if source in policies)
                if not selected_sources:
                    continue
                image_pair, prior, target, valid, _ = _read_paired_candidate(
                    item,
                    prior_geometry=carried_prior,
                    target_geometry=target_geometry,
                    image_size=args.image_size,
                    image_channels=6,
                    temporal_pair_input=True,
                    road_width_source_pixels=(
                        float(episode.metadata["road_width_source_pixels"])
                        if episode.metadata.get("road_width_source_pixels") is not None
                        else None
                    ),
                )
                if image_pair.shape[0] != 6:
                    raise ValueError(f"expected paired RGB input for {episode.episode_id}")
                operation = _residual_operation(prior, target, valid)
                sample_id = (
                    f"{episode.episode_id}__c{chain_index:05d}__s{step:04d}"
                    f"__v{variant_index:02d}"
                )
                prefix = data_dir / split / sample_id
                paths = {
                    "image": prefix.with_name(f"{prefix.name}_image.npy"),
                    "prior_image": prefix.with_name(f"{prefix.name}_prior_image.npy"),
                    "prior": prefix.with_name(f"{prefix.name}_prior.npy"),
                    "target": prefix.with_name(f"{prefix.name}_target.npy"),
                    "valid": prefix.with_name(f"{prefix.name}_valid.npy"),
                }
                _write_array(paths["prior_image"], image_pair[:3], image=True)
                _write_array(paths["image"], image_pair[3:], image=True)
                _write_array(paths["prior"], prior, image=False)
                _write_array(paths["target"], target, image=False)
                _write_array(paths["valid"], valid, image=False)
                records.append(
                    UpdaterSample(
                        sample_id=sample_id,
                        aoi_id=str(episode.aoi_id),
                        split=split,
                        image_path=paths["image"].relative_to(args.output_dir).as_posix(),
                        prior_image_path=(
                            paths["prior_image"].relative_to(args.output_dir).as_posix()
                        ),
                        prior_mask_path=paths["prior"].relative_to(args.output_dir).as_posix(),
                        target_mask_path=paths["target"].relative_to(args.output_dir).as_posix(),
                        valid_mask_path=paths["valid"].relative_to(args.output_dir).as_posix(),
                        edit_type=operation,
                        geometry_delta=_residual_geometry_delta(prior, target).tolist(),
                        object_id=str(episode.hypothesis.object_id),
                        clear_fraction=float(np.mean(valid >= 0.5)),
                        quality_source="sn7_train_internal_carried_runtime_temporal_v2",
                        dataset_name="sn7_carried_runtime_temporal",
                        supervision_type="real_temporal",
                        source_metadata={
                            "chain_index": chain_index,
                            "step": step,
                            "evidence_id": item.evidence_id,
                            "anchor_timestamp": str(episode.anchor_timestamp),
                            "prior_timestamp": item.prior_timestamp,
                            "temporal_pair_source": pair_source,
                            "state_policies": list(selected_sources),
                            "replay_policy": "carried-runtime-temporal-v2",
                            "target_alignment": "current_anchor_only",
                        },
                    )
                )
                split_counts[split] += 1
                operation_counts[f"{split}:{operation.value}"] += 1
                for source in selected_sources:
                    policy_counts[f"{split}:{source}"] += 1

    if not records:
        raise ValueError("no paired temporal carried-state samples were generated")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "updater_samples.jsonl"
    manifest.write_text(
        "".join(record.model_dump_json() + "\n" for record in records),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "sn7-carried-runtime-temporal-dataset-v2",
        "episodes": str(args.episodes.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "test_assets_read": False,
        "chain_count": len(chains),
        "validation_chain_index": args.validation_chain_index,
        "state_policies": list(policies),
        "sample_counts": dict(sorted(split_counts.items())),
        "residual_operation_counts": dict(sorted(operation_counts.items())),
        "state_policy_counts": dict(sorted(policy_counts.items())),
        "temporal_pair_source_counts": dict(sorted(pair_source_counts.items())),
        "missing_temporal_pair_episodes": missing_pair_count,
        "target_alignment": "current_anchor_only",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    print(json.dumps(build_dataset(parse_args()), indent=2))


if __name__ == "__main__":
    main()
