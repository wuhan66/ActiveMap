#!/usr/bin/env python3
"""Evaluate the model-backed segmentation tool on frozen MUNO21 validation data."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage
from tqdm import tqdm

from activemap.geo_tools.model_segmentation import MapConditionedSegmentationTool
from activemap.geo_tools.records import GeoToolCall, GeoToolName
from activemap.models import EditOperation, EpisodeRecord
from activemap.updater_records import UpdaterSample, load_updater_samples


def _parse_root_map(value: str) -> tuple[Path, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("root map must use SOURCE=TARGET")
    source, target = (Path(item) for item in value.split("=", 1))
    if not source.is_absolute() or not target.is_absolute():
        raise argparse.ArgumentTypeError("root map paths must be absolute")
    return source, target


def _remap_path(path: str | Path, mappings: list[tuple[Path, Path]]) -> Path:
    original = Path(path)
    for source, target in mappings:
        try:
            return target / original.relative_to(source)
        except ValueError:
            continue
    return original


def _read_validation_episodes(path: Path) -> dict[str, EpisodeRecord]:
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
                raise ValueError("evaluation input must not contain test episodes")
            if episode.split == "val":
                episodes[episode.episode_id] = episode
    return episodes


def _binary(path: str, shape: tuple[int, int] | None = None) -> np.ndarray:
    array = np.asarray(np.load(path)).squeeze()
    if array.ndim != 2:
        raise ValueError(f"expected a 2D mask at {path}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"mask shape mismatch at {path}: {array.shape} != {shape}")
    return array >= 0.5


def _iou(left: np.ndarray, right: np.ndarray, valid: np.ndarray) -> float:
    union = (left | right) & valid
    intersection = (left & right) & valid
    union_count = int(np.sum(union))
    return float(np.sum(intersection) / union_count) if union_count else 1.0


def _component_count(mask: np.ndarray, valid: np.ndarray) -> int:
    _, count = ndimage.label(mask & valid)
    return int(count)


def score_masks(
    prior: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float | int]:
    """Return map, change, safety, and fragmentation metrics for one sample."""
    prior = prior.astype(bool)
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    valid = valid.astype(bool)
    if not (prior.shape == prediction.shape == target.shape == valid.shape):
        raise ValueError("all masks must have the same shape")
    if not np.any(valid):
        raise ValueError("valid mask is empty")
    target_change = prior ^ target
    predicted_change = prior ^ prediction
    target_add = target & ~prior
    predicted_add = prediction & ~prior
    target_remove = prior & ~target
    predicted_remove = prior & ~prediction
    prior_components = _component_count(prior, valid)
    prediction_components = _component_count(prediction, valid)
    target_components = _component_count(target, valid)
    prior_target_iou = _iou(prior, target, valid)
    prediction_target_iou = _iou(prediction, target, valid)
    valid_count = int(np.sum(valid))
    return {
        "prior_target_iou": prior_target_iou,
        "prediction_target_iou": prediction_target_iou,
        "iou_delta": prediction_target_iou - prior_target_iou,
        "change_iou": _iou(predicted_change, target_change, valid),
        "added_change_iou": _iou(predicted_add, target_add, valid),
        "removed_change_iou": _iou(predicted_remove, target_remove, valid),
        "predicted_change_fraction": float(np.sum(predicted_change & valid) / valid_count),
        "target_change_fraction": float(np.sum(target_change & valid) / valid_count),
        "prior_component_error": abs(prior_components - target_components),
        "prediction_component_error": abs(prediction_components - target_components),
        "prior_components": prior_components,
        "prediction_components": prediction_components,
        "target_components": target_components,
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else float("nan")


def _summary(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    keys = (
        "prior_target_iou",
        "prediction_target_iou",
        "iou_delta",
        "change_iou",
        "added_change_iou",
        "removed_change_iou",
        "predicted_change_fraction",
        "target_change_fraction",
        "prior_component_error",
        "prediction_component_error",
    )
    return {"count": len(rows), **{key: _mean(rows, key) for key in keys}}


def grouped_bootstrap_delta(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    samples: int,
) -> dict[str, float]:
    """Bootstrap mean IoU delta by AOI, preserving correlated patches."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group_id"])].append(row)
    groups = sorted(grouped)
    if not groups:
        raise ValueError("cannot bootstrap an empty sample set")
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        chosen = rng.choice(groups, size=len(groups), replace=True)
        values = [float(row["iou_delta"]) for group in chosen for row in grouped[str(group)]]
        draws[index] = np.mean(values)
    return {
        "mean": _mean(rows, "iou_delta"),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "bootstrap_samples": samples,
        "group_count": len(groups),
    }


def promotion_decision(
    rows: list[dict[str, Any]],
    overall_ci: dict[str, float],
    edited_ci: dict[str, float],
) -> dict[str, Any]:
    """Apply frozen validation-only gates before Agent tool registration."""
    keep_rows = [row for row in rows if row["edit_type"] == EditOperation.KEEP.value]
    per_edit = {
        edit.value: _summary([row for row in rows if row["edit_type"] == edit.value])
        for edit in EditOperation
    }
    gates = {
        "all_outputs_valid": all(bool(row["success"]) for row in rows),
        "overall_iou_delta_ci_positive": overall_ci["ci95_low"] > 0.0,
        "edited_iou_delta_ci_positive": edited_ci["ci95_low"] > 0.0,
        "keep_false_change_at_most_0_01": bool(keep_rows)
        and _mean(keep_rows, "predicted_change_fraction") <= 0.01,
        "component_error_non_inferior": _mean(rows, "prediction_component_error")
        <= _mean(rows, "prior_component_error"),
        "no_edit_class_regresses_over_0_01": all(
            int(summary["count"]) > 0 and float(summary["iou_delta"]) >= -0.01
            for summary in per_edit.values()
        ),
    }
    return {
        "promoted_for_agent_registration": all(gates.values()),
        "gates": gates,
        "per_edit": per_edit,
        "failure_policy": (
            "Keep RASTER_SEGMENT as a diagnostic backend; do not expose it to the Agent "
            "or claim remote-sensing segmentation gain."
        ),
    }


def _episode_for_sample(
    sample: UpdaterSample, episodes: dict[str, EpisodeRecord]
) -> EpisodeRecord:
    episode_id = f"{sample.sample_id}__temporal"
    episode = episodes.get(episode_id)
    if episode is None:
        raise ValueError(f"missing validation episode {episode_id}")
    return episode


def _group_id(sample: UpdaterSample) -> str:
    """Group overlapping patches from the same source annotation."""
    annotation_index = sample.source_metadata.get("annotation_index")
    if sample.aoi_id is not None and annotation_index is not None:
        return f"{sample.aoi_id}:annotation:{annotation_index}"
    if sample.object_id is not None and "-patch-" in sample.object_id:
        return sample.object_id.split("-patch-", maxsplit=1)[0]
    return sample.sample_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("updater_manifest", type=Path)
    parser.add_argument("episodes", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument(
        "--backend",
        choices=("updater", "sam-road"),
        default="updater",
    )
    parser.add_argument("--sam-road-repo", type=Path)
    parser.add_argument("--sam-road-config", type=Path)
    parser.add_argument("--sam-base-checkpoint", type=Path)
    parser.add_argument("--sam-road-commit")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--fusion",
        choices=("replace", "add-only"),
        default="replace",
    )
    parser.add_argument(
        "--asset-root-map",
        action="append",
        type=_parse_root_map,
        default=[],
        metavar="SOURCE=TARGET",
    )
    parser.add_argument(
        "--reuse-artifacts",
        action="store_true",
        help="Recompute metrics from complete saved masks without model inference.",
    )
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if args.bootstrap <= 0:
        raise ValueError("bootstrap must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    if args.backend == "sam-road":
        missing = [
            name
            for name, value in (
                ("--sam-road-repo", args.sam_road_repo),
                ("--sam-road-config", args.sam_road_config),
                ("--sam-base-checkpoint", args.sam_base_checkpoint),
                ("--sam-road-commit", args.sam_road_commit),
            )
            if value is None
        ]
        if missing:
            raise ValueError(f"SAM-Road requires: {', '.join(missing)}")

    validation = load_updater_samples(args.updater_manifest, split="val")
    if args.limit is not None:
        validation = validation[: args.limit]
    episodes = _read_validation_episodes(args.episodes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_root = args.output_dir / "artifacts"
    if args.reuse_artifacts:
        tool = None
    elif args.backend == "sam-road":
        tool = MapConditionedSegmentationTool.from_sam_road(
            args.sam_road_repo,
            args.sam_road_config,
            args.checkpoint,
            args.sam_base_checkpoint,
            artifact_root,
            device=args.device,
            upstream_commit=args.sam_road_commit,
        )
    else:
        tool = MapConditionedSegmentationTool.from_updater_checkpoint(
            args.checkpoint,
            artifact_root,
            device=args.device,
    )
    rows: list[dict[str, Any]] = []
    for sample in tqdm(validation, desc="model segmentation validation", unit="sample"):
        _episode_for_sample(sample, episodes)
        prior = _binary(sample.prior_mask_path)
        target = _binary(sample.target_mask_path, prior.shape)
        valid = (
            _binary(sample.valid_mask_path, prior.shape)
            if sample.valid_mask_path is not None
            else np.ones(prior.shape, dtype=bool)
        )
        artifact = artifact_root / f"val-{sample.sample_id}_segment.npy"
        if args.reuse_artifacts:
            if not artifact.is_file():
                raise FileNotFoundError(f"missing reusable artifact: {artifact}")
            success = True
            tool_outputs: dict[str, Any] = {}
        else:
            if tool is None:
                raise AssertionError("segmentation tool was not initialized")
            result = tool.run(
                GeoToolCall(
                    call_id=f"val-{sample.sample_id}",
                    tool=GeoToolName.RASTER_SEGMENT,
                    inputs={
                        "image_path": str(
                            _remap_path(sample.image_path, args.asset_root_map)
                        ),
                        "prior_mask_path": sample.prior_mask_path,
                    },
                    parameters={
                        "out_size": [prior.shape[1], prior.shape[0]],
                        "threshold": args.threshold,
                    },
                )
            )
            if not result.success or result.artifacts != [str(artifact.resolve())]:
                raise RuntimeError(f"segmentation tool failed for {sample.sample_id}")
            success = result.success
            tool_outputs = result.outputs
        raw_prediction = _binary(str(artifact), prior.shape)
        prediction = (
            prior | raw_prediction
            if args.fusion == "add-only"
            else raw_prediction
        )
        scored_artifact = artifact
        if args.fusion != "replace":
            fused_root = args.output_dir / "fused_artifacts"
            fused_root.mkdir(parents=True, exist_ok=True)
            scored_artifact = fused_root / f"val-{sample.sample_id}_segment.npy"
            np.save(scored_artifact, prediction.astype(np.uint8))
        rows.append(
            {
                "sample_id": sample.sample_id,
                "group_id": _group_id(sample),
                "edit_type": sample.edit_type.value,
                "success": success,
                "artifact": str(scored_artifact.resolve()),
                "raw_artifact": str(artifact.resolve()),
                "tool_outputs": tool_outputs,
                **score_masks(prior, prediction, target, valid),
            }
        )

    edited = [row for row in rows if row["edit_type"] != EditOperation.KEEP.value]
    overall_ci = grouped_bootstrap_delta(
        rows, seed=args.seed, samples=args.bootstrap
    )
    edited_ci = grouped_bootstrap_delta(
        edited, seed=args.seed + 1, samples=args.bootstrap
    )
    report = {
        "schema_version": "muno21-model-segmentation-validation-v1",
        "split": "val",
        "sample_count": len(rows),
        "backend": args.backend,
        "fusion": args.fusion,
        "supported_operations": (
            ["KEEP", "ADD"]
            if args.fusion == "add-only"
            else [edit.value for edit in EditOperation]
        ),
        "threshold": args.threshold,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(
            args.checkpoint.read_bytes()
        ).hexdigest(),
        "sam_road_commit": args.sam_road_commit,
        "asset_root_maps": [
            {"source": str(source), "target": str(target)}
            for source, target in args.asset_root_map
        ],
        "reused_artifacts": args.reuse_artifacts,
        "bootstrap_group": "aoi_id+annotation_index",
        "test_assets_read": False,
        "overall": _summary(rows),
        "edited": _summary(edited),
        "overall_iou_delta_bootstrap": overall_ci,
        "edited_iou_delta_bootstrap": edited_ci,
        "decision": promotion_decision(rows, overall_ci, edited_ci),
    }
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
