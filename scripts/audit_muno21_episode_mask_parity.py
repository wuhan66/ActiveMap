#!/usr/bin/env python3
"""Compare MUNO21 episode-rasterized masks with updater training arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from shapely.geometry import shape

from activemap.models import EpisodeRecord
from activemap.oracle.updater_counterfactual import _read_candidate
from activemap.updater_records import load_updater_samples


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    left_mask = left >= 0.5
    right_mask = right >= 0.5
    union = np.sum(left_mask | right_mask)
    return float(np.sum(left_mask & right_mask) / union) if union else 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("updater_manifest", type=Path)
    parser.add_argument("episodes", type=Path)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()

    samples = load_updater_samples(args.updater_manifest, split=args.split)[: args.count]
    sample_by_id = {sample.sample_id: sample for sample in samples}
    episodes: dict[str, EpisodeRecord] = {}
    with args.episodes.open("r", encoding="utf-8") as handle:
        for line in handle:
            episode = EpisodeRecord.model_validate_json(line)
            source_id = episode.episode_id.removesuffix("__temporal")
            if source_id in sample_by_id:
                episodes[source_id] = episode
    rows = []
    for sample in samples:
        episode = episodes[sample.sample_id]
        prior_array = np.load(sample.prior_mask_path)
        target_array = np.load(sample.target_mask_path)
        latest = max(episode.evidence_catalog, key=lambda item: item.timestamp)
        _, rebuilt_prior, rebuilt_target, _, _ = _read_candidate(
            latest,
            prior_geometry=(
                shape(episode.prior_geometry.model_dump(mode="json"))
                if episode.prior_geometry is not None
                else None
            ),
            target_geometry=(
                shape(episode.target_geometry.model_dump(mode="json"))
                if episode.target_geometry is not None
                else None
            ),
            image_size=prior_array.shape[-1],
            image_channels=3,
            road_width_source_pixels=float(episode.metadata["road_width_source_pixels"]),
        )
        rows.append(
            {
                "sample_id": sample.sample_id,
                "prior_iou": _iou(prior_array, rebuilt_prior),
                "target_iou": _iou(target_array, rebuilt_target),
                "prior_disagreement_pixels": int(
                    np.sum((prior_array >= 0.5) != (rebuilt_prior >= 0.5))
                ),
                "target_disagreement_pixels": int(
                    np.sum((target_array >= 0.5) != (rebuilt_target >= 0.5))
                ),
            }
        )
    summary = {
        "split": args.split,
        "sample_count": len(rows),
        "test_assets_read": False,
        "minimum_prior_iou": min(row["prior_iou"] for row in rows),
        "minimum_target_iou": min(row["target_iou"] for row in rows),
        "rows": rows,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
