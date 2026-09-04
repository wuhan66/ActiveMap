#!/usr/bin/env python3
"""Execute map-relative segmentation for reachable post-acquisition examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from activemap.agent.identifiers import public_evidence_id, public_task_id
from activemap.agent.tool_belief_data import PostAcquisitionToolPairExample
from activemap.agent.tool_features import (
    SEMANTIC_TOOL_FEATURE_NAMES,
    encode_semantic_tool_result,
)
from activemap.geo_tools.model_segmentation import MapConditionedSegmentationTool
from activemap.geo_tools.records import GeoToolCall, GeoToolName
from activemap.models import EpisodeRecord
from activemap.updater_records import UpdaterSample, load_updater_samples


def _read(path: Path, model: type[Any]) -> list[Any]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(model.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"invalid {path}:{line_number}: {exc}") from exc
    if not rows:
        raise ValueError(f"empty input: {path}")
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _index_episodes(rows: list[EpisodeRecord]) -> dict[str, EpisodeRecord]:
    result = {public_task_id(row.episode_id): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("public episode identifier collision")
    return result


def _index_samples(rows: list[UpdaterSample]) -> dict[str, UpdaterSample]:
    result = {public_task_id(f"{row.sample_id}__temporal"): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("public updater-sample identifier collision")
    return result


def default_threshold_grid(
    backend: str, selected_threshold: float
) -> tuple[float, ...] | None:
    if backend == "prior-sam-road":
        return (selected_threshold,)
    if backend == "sam-road":
        return (0.10, 0.20, 0.30, 0.341, 0.40, 0.50, 0.60)
    return None


def parse_asset_root_maps(values: list[str]) -> tuple[tuple[Path, Path], ...]:
    mappings = []
    for value in values:
        if "=" not in value:
            raise ValueError("asset root maps must use SOURCE=TARGET")
        source_text, target_text = value.split("=", 1)
        source, target = Path(source_text), Path(target_text)
        if not source.is_absolute() or not target.is_absolute():
            raise ValueError("asset root maps must contain absolute paths")
        mappings.append((source, target))
    return tuple(mappings)


def remap_asset_path(
    path: str | Path, mappings: tuple[tuple[Path, Path], ...]
) -> Path:
    original = Path(path)
    for source, target in mappings:
        try:
            return target / original.relative_to(source)
        except ValueError:
            continue
    return original


def augment_examples(
    rows: list[PostAcquisitionToolPairExample],
    episodes: dict[str, EpisodeRecord],
    samples: dict[str, UpdaterSample],
    tool: MapConditionedSegmentationTool,
    *,
    threshold: float,
    threshold_grid: tuple[float, ...] | None = None,
    change_thresholds: dict[str, float] | None = None,
    asset_root_maps: tuple[tuple[Path, Path], ...] = (),
) -> tuple[list[PostAcquisitionToolPairExample], dict[str, Any]]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    splits = {row.split for row in rows}
    if len(splits) != 1 or not splits <= {"train", "val"}:
        raise ValueError("semantic augmentation requires one train or val split")
    output = []
    counts: Counter[str] = Counter()
    feature_values = {name: [] for name in SEMANTIC_TOOL_FEATURE_NAMES}
    for row in tqdm(rows, desc=f"semantic tool {next(iter(splits))}", unit="example"):
        if row.semantic_result is not None:
            raise ValueError(f"example already has semantic result: {row.example_id}")
        episode = episodes.get(row.task_id)
        sample = samples.get(row.task_id)
        if episode is None or sample is None:
            raise ValueError(f"missing episode or updater sample for {row.task_id}")
        evidence_by_public = {
            public_evidence_id(item.evidence_id): item for item in episode.evidence_catalog
        }
        evidence = evidence_by_public.get(row.evidence_id)
        if evidence is None:
            raise ValueError(f"missing evidence {row.task_id}/{row.evidence_id}")
        prior_path = remap_asset_path(sample.prior_mask_path, asset_root_maps)
        image_path = remap_asset_path(sample.image_path, asset_root_maps)
        prior = np.asarray(np.load(prior_path)).squeeze()
        if prior.ndim != 2:
            raise ValueError(f"prior mask is not 2D: {sample.prior_mask_path}")
        result = tool.run(
            GeoToolCall(
                call_id=f"semantic-{row.example_id}",
                tool=GeoToolName.RASTER_SEGMENT,
                inputs={
                    "image_path": str(image_path),
                    "prior_mask_path": str(prior_path),
                },
                parameters={
                    "threshold": threshold,
                    **(
                        {
                            "add_threshold": change_thresholds["add"],
                            "remove_threshold": change_thresholds["remove"],
                        }
                        if change_thresholds is not None
                        else {}
                    ),
                    **(
                        {"threshold_grid": list(threshold_grid)}
                        if threshold_grid is not None
                        else {}
                    ),
                },
            )
        )
        if not result.success:
            raise RuntimeError(f"semantic tool failed: {row.example_id}")
        encoded = encode_semantic_tool_result(result)
        for name, value in zip(SEMANTIC_TOOL_FEATURE_NAMES, encoded, strict=True):
            feature_values[name].append(value)
        output.append(
            row.model_copy(
                update={
                    "semantic_result": result,
                    "tool_cost": row.tool_cost + result.cost,
                    "metadata": {
                        **row.metadata,
                        "semantic_tool": type(getattr(tool, "predictor", tool)).__name__,
                        "semantic_threshold": threshold,
                        "semantic_change_thresholds": change_thresholds,
                        "semantic_input": "updater_aligned_patch",
                        "evidence_region": list(evidence.region),
                    },
                }
            )
        )
        counts[f"gt:{row.gt_edit.value}"] += 1
    summary = {
        "schema_version": "post-acquisition-semantic-tool-data-v1",
        "split": next(iter(splits)),
        "example_count": len(output),
        "task_count": len({row.task_id for row in output}),
        "counts": dict(sorted(counts.items())),
        "threshold": threshold,
        "threshold_grid": list(threshold_grid) if threshold_grid is not None else None,
        "change_thresholds": change_thresholds,
        "asset_root_maps": [
            {"source": str(source), "target": str(target)}
            for source, target in asset_root_maps
        ],
        "feature_summary": {
            name: {
                "minimum": min(values),
                "maximum": max(values),
                "mean": float(np.mean(values)),
                "unique_count": len(set(values)),
            }
            for name, values in feature_values.items()
            if values
        },
        "test_assets_read": False,
    }
    return output, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("post_acquisition_jsonl", type=Path)
    parser.add_argument("episodes_jsonl", type=Path)
    parser.add_argument("updater_manifest", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--split", required=True, choices=("train", "val"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--threshold-grid")
    parser.add_argument(
        "--backend",
        choices=("updater", "sam-road", "prior-sam-road"),
        default="updater",
    )
    parser.add_argument("--sam-road-repo", type=Path)
    parser.add_argument("--sam-road-config", type=Path)
    parser.add_argument("--sam-base-checkpoint", type=Path)
    parser.add_argument("--sam-road-source-checkpoint", type=Path)
    parser.add_argument("--change-summary", type=Path)
    parser.add_argument(
        "--change-parameterization",
        choices=("independent", "current_difference"),
        default="independent",
    )
    parser.add_argument(
        "--operation-head",
        choices=("global_stats", "spatial_pyramid"),
        default="global_stats",
    )
    parser.add_argument("--sam-road-commit")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--limit-per-class", type=int)
    parser.add_argument(
        "--asset-root-map",
        action="append",
        default=[],
        metavar="SOURCE=TARGET",
    )
    args = parser.parse_args()
    asset_root_maps = parse_asset_root_maps(args.asset_root_map)
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    if args.limit_per_class is not None and args.limit_per_class <= 0:
        raise ValueError("limit-per-class must be positive")
    if args.limit is not None and args.limit_per_class is not None:
        raise ValueError("limit and limit-per-class are mutually exclusive")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    rows = [
        row
        for row in _read(
            args.post_acquisition_jsonl, PostAcquisitionToolPairExample
        )
        if row.split == args.split
    ]
    if args.limit is not None:
        rows = rows[: args.limit]
    if args.limit_per_class is not None:
        counts: Counter[str] = Counter()
        selected = []
        for row in rows:
            operation = row.gt_edit.value
            if counts[operation] < args.limit_per_class:
                selected.append(row)
                counts[operation] += 1
        rows = selected
    task_ids = {row.task_id for row in rows}
    episodes = [
        row
        for row in _read(args.episodes_jsonl, EpisodeRecord)
        if row.split == args.split and public_task_id(row.episode_id) in task_ids
    ]
    samples = [
        row
        for row in load_updater_samples(args.updater_manifest, split=args.split)
        if public_task_id(f"{row.sample_id}__temporal") in task_ids
    ]
    if {row.task_id for row in rows} != set(_index_episodes(episodes)):
        raise ValueError("post-acquisition and episode task support differ")
    if {row.task_id for row in rows} != set(_index_samples(samples)):
        raise ValueError("post-acquisition and updater task support differ")
    args.output_root.mkdir(parents=True)
    if args.backend in {"sam-road", "prior-sam-road"}:
        required = {
            "--sam-road-repo": args.sam_road_repo,
            "--sam-road-config": args.sam_road_config,
            "--sam-base-checkpoint": args.sam_base_checkpoint,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"SAM-Road backend requires: {', '.join(missing)}")
        assert args.sam_road_repo is not None
        assert args.sam_road_config is not None
        assert args.sam_base_checkpoint is not None
        commit = args.sam_road_commit or subprocess.check_output(
            ["git", "-C", str(args.sam_road_repo), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        if args.backend == "prior-sam-road":
            if args.sam_road_source_checkpoint is None or args.change_summary is None:
                raise ValueError(
                    "prior-sam-road requires --sam-road-source-checkpoint and "
                    "--change-summary"
                )
            tool = MapConditionedSegmentationTool.from_prior_conditioned_sam_road(
                args.sam_road_repo,
                args.sam_road_config,
                args.sam_road_source_checkpoint,
                args.sam_base_checkpoint,
                args.checkpoint,
                args.output_root / "artifacts",
                device=args.device,
                parameterization=args.change_parameterization,
                operation_head=args.operation_head,
                upstream_commit=commit,
            )
        else:
            tool = MapConditionedSegmentationTool.from_sam_road(
                args.sam_road_repo,
                args.sam_road_config,
                args.checkpoint,
                args.sam_base_checkpoint,
                args.output_root / "artifacts",
                device=args.device,
                upstream_commit=commit,
            )
    else:
        commit = None
        tool = MapConditionedSegmentationTool.from_updater_checkpoint(
            args.checkpoint,
            args.output_root / "artifacts",
            device=args.device,
        )
    change_thresholds = None
    if args.backend == "prior-sam-road":
        assert args.change_summary is not None
        change_summary = json.loads(args.change_summary.read_text(encoding="utf-8"))
        if change_summary.get("frozen_test_access") is not False:
            raise ValueError("change summary does not prove frozen-test isolation")
        selected_thresholds = change_summary.get("selected_channel_thresholds")
        if not isinstance(selected_thresholds, dict) or not all(
            name in selected_thresholds for name in ("current", "add", "remove")
        ):
            raise ValueError("change summary has no calibrated channel thresholds")
        change_thresholds = {
            name: float(selected_thresholds[name])
            for name in ("current", "add", "remove")
        }
        if any(not 0.0 <= value <= 1.0 for value in change_thresholds.values()):
            raise ValueError("change summary contains invalid channel thresholds")
    threshold = args.threshold
    if threshold is None:
        if change_thresholds is not None:
            threshold = change_thresholds["current"]
        elif args.backend == "sam-road":
            threshold = float(tool.predictor.road_threshold)
        else:
            threshold = 0.5
    threshold_grid = (
        tuple(float(value) for value in args.threshold_grid.split(","))
        if args.threshold_grid
        else default_threshold_grid(args.backend, threshold)
    )
    augmented, summary = augment_examples(
        rows,
        _index_episodes(episodes),
        _index_samples(samples),
        tool,
        threshold=threshold,
        threshold_grid=threshold_grid,
        change_thresholds=change_thresholds,
        asset_root_maps=asset_root_maps,
    )
    output = args.output_root / f"{args.split}.jsonl"
    with output.open("x", encoding="utf-8") as handle:
        for row in augmented:
            handle.write(row.model_dump_json() + "\n")
    summary["sources"] = {
        "post_acquisition": _sha256(args.post_acquisition_jsonl),
        "episodes": _sha256(args.episodes_jsonl),
        "updater_manifest": _sha256(args.updater_manifest),
        "checkpoint": _sha256(args.checkpoint),
    }
    summary["backend"] = args.backend
    if args.backend in {"sam-road", "prior-sam-road"}:
        assert args.sam_road_config is not None
        assert args.sam_base_checkpoint is not None
        summary["sources"].update(
            {
                "sam_base_checkpoint": _sha256(args.sam_base_checkpoint),
                "sam_road_config": _sha256(args.sam_road_config),
                "sam_road_commit": commit,
            }
        )
    if args.backend == "prior-sam-road":
        assert args.sam_road_source_checkpoint is not None
        assert args.change_summary is not None
        summary["sources"].update(
            {
                "sam_road_source_checkpoint": _sha256(
                    args.sam_road_source_checkpoint
                ),
                "change_summary": _sha256(args.change_summary),
            }
        )
    (args.output_root / f"{args.split}.summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
