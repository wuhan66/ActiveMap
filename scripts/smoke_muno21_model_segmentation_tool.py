#!/usr/bin/env python3
"""Run one validation-only smoke for the model-backed segmentation tool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from activemap.geo_tools.model_segmentation import MapConditionedSegmentationTool
from activemap.geo_tools.records import GeoToolCall, GeoToolName
from activemap.models import EpisodeRecord
from activemap.updater_records import load_updater_samples


def _read_episodes(path: Path) -> dict[str, EpisodeRecord]:
    episodes: dict[str, EpisodeRecord] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                episode = EpisodeRecord.model_validate_json(line)
            except Exception as exc:
                raise ValueError(f"invalid episode at {path}:{line_number}") from exc
            if episode.split == "test":
                raise ValueError("smoke input must not contain test episodes")
            episodes[episode.episode_id] = episode
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("updater_manifest", type=Path)
    parser.add_argument("episodes", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--sample-id")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-size", type=int, default=512)
    args = parser.parse_args()
    if args.out_size <= 0:
        raise ValueError("out-size must be positive")

    validation = load_updater_samples(args.updater_manifest, split="val")
    if args.sample_id is not None:
        matches = [sample for sample in validation if sample.sample_id == args.sample_id]
        if len(matches) != 1:
            raise ValueError(f"expected one validation sample_id={args.sample_id!r}")
        sample = matches[0]
    else:
        sample = validation[0]
    episode_id = f"{sample.sample_id}__temporal"
    episode = _read_episodes(args.episodes).get(episode_id)
    if episode is None or episode.split != "val":
        raise ValueError(f"missing validation episode for {sample.sample_id}")
    initial = next(
        (
            item
            for item in episode.evidence_catalog
            if item.timestamp.startswith(str(episode.anchor_timestamp)[:4])
        ),
        episode.evidence_catalog[-1],
    )
    x_min, y_min, x_max, y_max = initial.region

    tool = MapConditionedSegmentationTool.from_updater_checkpoint(
        args.checkpoint,
        args.output_dir / "artifacts",
        device=args.device,
    )
    result = tool.run(
        GeoToolCall(
            call_id=f"smoke-{sample.sample_id}",
            tool=GeoToolName.RASTER_SEGMENT,
            inputs={
                "image_path": initial.image_path,
                "prior_mask_path": sample.prior_mask_path,
            },
            parameters={
                "pixel_window": [x_min, y_min, x_max - x_min, y_max - y_min],
                "out_size": [args.out_size, args.out_size],
                "threshold": 0.5,
            },
        )
    )
    report = {
        "schema_version": "muno21-model-segmentation-smoke-v1",
        "sample_id": sample.sample_id,
        "episode_id": episode.episode_id,
        "split": episode.split,
        "gt_edit": episode.gt_edit.op.value,
        "evidence_id": initial.evidence_id,
        "result": result.model_dump(mode="json"),
        "test_assets_read": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
