#!/usr/bin/env python3
"""Restrict Step-0 selector supervision to deployment-compatible raster grids."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from functools import cache
from pathlib import Path
from typing import Any

import rasterio
from affine import Affine
from PIL import Image
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

from activemap.models import EpisodeRecord
from activemap.selector_records import SelectorSample

SUPPORT_CONTRACT = "same-runtime-grid-v1"


def parse_asset_map(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("asset maps must use SOURCE=DESTINATION")
    source, destination = value.split("=", 1)
    if not source or not destination:
        raise argparse.ArgumentTypeError("asset maps require non-empty paths")
    return source.rstrip("/"), destination.rstrip("/")


def remap_path(path: str, mappings: tuple[tuple[str, str], ...]) -> Path:
    for source, destination in mappings:
        if path == source or path.startswith(source + "/"):
            return Path(destination + path[len(source) :])
    return Path(path)


@cache
def raster_source(path: str, jpeg: bool) -> tuple[Any, int, int]:
    source = Path(path)
    if jpeg:
        with Image.open(source) as image:
            width, height = image.size
        return None, width, height
    with rasterio.open(source) as dataset:
        return dataset.transform, dataset.width, dataset.height


def grid_key(
    item: Any,
    *,
    image_size: int,
    asset_maps: tuple[tuple[str, str], ...],
) -> tuple[tuple[int, int], tuple[float, ...]]:
    path = remap_path(item.image_path, asset_maps)
    jpeg = path.suffix.lower() in {".jpg", ".jpeg"} and item.udm_path is None
    base_transform, width, height = raster_source(str(path), jpeg)
    x_min, y_min, x_max, y_max = item.region
    window = Window(x_min, y_min, x_max - x_min, y_max - y_min)
    if jpeg:
        transform = Affine.translation(x_min, y_min) * Affine.scale(
            float(window.width) / image_size,
            float(window.height) / image_size,
        )
    else:
        transform = window_transform(window, base_transform) * Affine.scale(
            float(window.width) / image_size,
            float(window.height) / image_size,
        )
    return (width, height), tuple(float(value) for value in transform)


def load_episode_index(
    paths: list[Path],
) -> tuple[dict[str, EpisodeRecord], list[dict[str, Any]]]:
    episodes: dict[str, EpisodeRecord] = {}
    receipts = []
    for path in paths:
        digest = hashlib.sha256()
        count = 0
        with path.open("rb") as handle:
            for line_number, line in enumerate(handle, start=1):
                digest.update(line)
                if not line.strip():
                    continue
                try:
                    episode = EpisodeRecord.model_validate_json(line)
                except Exception as exc:
                    raise ValueError(f"invalid episode at {path}:{line_number}") from exc
                if episode.split == "test":
                    raise ValueError("runtime-grid training support forbids test episodes")
                if episode.episode_id in episodes:
                    raise ValueError(f"duplicate episode id: {episode.episode_id}")
                episodes[episode.episode_id] = episode
                count += 1
        receipts.append(
            {
                "path": str(path.resolve()),
                "sha256": digest.hexdigest(),
                "episode_count": count,
            }
        )
    if not episodes:
        raise ValueError("episode inputs are empty")
    return episodes, receipts


def filter_sample_to_support(
    sample: SelectorSample,
    *,
    eligible_ids: set[str],
    catalog_count: int,
) -> tuple[SelectorSample, dict[str, Any]]:
    indices = [
        index
        for index, evidence_id in enumerate(sample.evidence_ids)
        if evidence_id in eligible_ids
    ]
    if not indices:
        raise ValueError(f"{sample.sample_id} has no affordable same-grid candidates")
    original_target = sample.target_index()
    original_target_id = (
        None
        if original_target == len(sample.evidence_ids)
        else sample.evidence_ids[original_target]
    )
    metadata = dict(sample.metadata)
    metadata["candidate_support_contract"] = {
        "version": SUPPORT_CONTRACT,
        "catalog_candidate_count": catalog_count,
        "eligible_catalog_candidate_count": len(eligible_ids),
        "excluded_catalog_candidate_count": catalog_count - len(eligible_ids) - 1,
        "initial_evidence_id": metadata["initial_evidence_id"],
    }
    filtered = sample.model_copy(
        update={
            "evidence_ids": [sample.evidence_ids[index] for index in indices],
            "evidence_features": [sample.evidence_features[index] for index in indices],
            "evidence_costs": [sample.evidence_costs[index] for index in indices],
            "false_edit_risks": [sample.false_edit_risks[index] for index in indices],
            "oracle_utilities": [sample.oracle_utilities[index] for index in indices],
            "metadata": metadata,
        }
    )
    filtered_target = filtered.target_index()
    filtered_target_id = (
        None
        if filtered_target == len(filtered.evidence_ids)
        else filtered.evidence_ids[filtered_target]
    )
    return filtered, {
        "candidate_count_before": len(sample.evidence_ids),
        "candidate_count_after": len(filtered.evidence_ids),
        "target_before": "STOP" if original_target_id is None else "ACQUIRE",
        "target_after": "STOP" if filtered_target_id is None else "ACQUIRE",
        "target_id_preserved": original_target_id == filtered_target_id,
    }


def build_manifest(
    source: Path,
    episode_paths: list[Path],
    output: Path,
    *,
    image_size: int,
    asset_maps: tuple[tuple[str, str], ...],
    budget: float | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    partial = output.with_suffix(output.suffix + ".partial")
    if partial.exists():
        raise FileExistsError(partial)
    episodes, episode_receipts = load_episode_index(episode_paths)
    source_digest = hashlib.sha256()
    output_digest = hashlib.sha256()
    counts: Counter[str] = Counter()
    before_counts: list[int] = []
    after_counts: list[int] = []
    support_cache: dict[tuple[str, str], set[str]] = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as handle, partial.open("xb") as destination:
        for line_number, line in enumerate(handle, start=1):
            source_digest.update(line)
            if not line.strip():
                continue
            try:
                sample = SelectorSample.model_validate_json(line)
            except Exception as exc:
                raise ValueError(f"invalid selector state at line {line_number}") from exc
            if sample.split == "test" or sample.metadata.get("test_assets_read") is True:
                raise ValueError("runtime-grid manifest forbids test records")
            sample_budget = float(sample.metadata["budget"])
            if budget is not None and abs(sample_budget - budget) > 1e-6:
                continue
            source_episode = str(sample.metadata["source_episode"])
            episode = episodes.get(source_episode)
            if episode is None:
                raise ValueError(f"missing episode for {source_episode}")
            initial_id = str(sample.metadata["initial_evidence_id"])
            cache_key = source_episode, initial_id
            eligible_ids = support_cache.get(cache_key)
            if eligible_ids is None:
                catalog = {item.evidence_id: item for item in episode.evidence_catalog}
                if initial_id not in catalog:
                    raise ValueError(f"{source_episode} lacks initial evidence {initial_id}")
                anchor_key = grid_key(
                    catalog[initial_id], image_size=image_size, asset_maps=asset_maps
                )
                eligible_ids = {
                    evidence_id
                    for evidence_id, item in catalog.items()
                    if evidence_id != initial_id
                    and grid_key(item, image_size=image_size, asset_maps=asset_maps)
                    == anchor_key
                }
                support_cache[cache_key] = eligible_ids
            filtered, audit = filter_sample_to_support(
                sample,
                eligible_ids=eligible_ids,
                catalog_count=len(episode.evidence_catalog),
            )
            encoded = (filtered.model_dump_json() + "\n").encode()
            destination.write(encoded)
            output_digest.update(encoded)
            before_counts.append(audit["candidate_count_before"])
            after_counts.append(audit["candidate_count_after"])
            counts[f"split:{filtered.split}"] += 1
            counts[f"target_before:{audit['target_before']}"] += 1
            counts[f"target_after:{audit['target_after']}"] += 1
            counts[f"target_id_preserved:{audit['target_id_preserved']}"] += 1
            counts[f"gt:{filtered.metadata.get('gt_edit', 'UNKNOWN')}:{audit['target_after']}"] += 1
    if not before_counts:
        partial.unlink(missing_ok=True)
        raise ValueError("no selector states match the requested support")
    partial.replace(output)
    result = {
        "schema_version": "activemap-runtime-grid-selector-manifest-v1",
        "support_contract": SUPPORT_CONTRACT,
        "image_size": image_size,
        "budget": budget,
        "source": {"path": str(source.resolve()), "sha256": source_digest.hexdigest()},
        "episodes": episode_receipts,
        "output": {"path": str(output.resolve()), "sha256": output_digest.hexdigest()},
        "record_count": sum(count for key, count in counts.items() if key.startswith("split:")),
        "counts": dict(sorted(counts.items())),
        "candidate_count": {
            "mean_before": sum(before_counts) / len(before_counts),
            "mean_after": sum(after_counts) / len(after_counts),
            "minimum_after": min(after_counts),
            "maximum_after": max(after_counts),
        },
        "unique_episode_anchor_supports": len(support_cache),
        "asset_root_maps": [f"{source}={destination}" for source, destination in asset_maps],
        "test_assets_read": False,
    }
    summary = output.with_suffix(output.suffix + ".summary.json")
    summary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--episodes", action="append", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--budget", type=float)
    parser.add_argument("--asset-root-map", action="append", type=parse_asset_map, default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.image_size <= 0:
        raise ValueError("--image-size must be positive")
    if args.budget is not None and args.budget <= 0:
        raise ValueError("--budget must be positive")
    print(
        json.dumps(
            build_manifest(
                args.source,
                args.episodes,
                args.output,
                image_size=args.image_size,
                asset_maps=tuple(args.asset_root_map),
                budget=args.budget,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
