#!/usr/bin/env python3
"""Replay Agent terminal decisions as auditable MUNO21 road-footprint vector deltas."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window
from tqdm.auto import tqdm

from activemap.agent.identifiers import public_task_id, resolve_evidence_id
from activemap.agent.writeback import EvidenceMaskPrediction, evaluate_typed_writeback
from activemap.data.prior_input_corruption import (
    deterministic_prior_translation,
    morph_prior_no_wrap,
)
from activemap.evaluation.episode_utility import (
    score_episode_profiles,
    utility_protocol,
)
from activemap.models import EditOperation, EpisodeRecord, EvidenceItem
from activemap.oracle.updater_counterfactual import _geometry, _read_candidate, load_episodes


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _parse_asset_root_maps(values: list[str]) -> tuple[tuple[Path, Path], ...]:
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


def _remap_path(value: str | None, mappings: tuple[tuple[Path, Path], ...]) -> str | None:
    if value is None:
        return None
    original = Path(value)
    for source, target in mappings:
        try:
            return str(target / original.relative_to(source))
        except ValueError:
            continue
    return value


def _remap_episode_assets(
    episodes: list[EpisodeRecord], mappings: tuple[tuple[Path, Path], ...]
) -> list[EpisodeRecord]:
    if not mappings:
        return episodes
    result = []
    for episode in episodes:
        catalog = [
            item.model_copy(
                update={
                    "image_path": _remap_path(item.image_path, mappings),
                    "udm_path": _remap_path(item.udm_path, mappings),
                    "prior_image_path": _remap_path(item.prior_image_path, mappings),
                    "prior_udm_path": _remap_path(item.prior_udm_path, mappings),
                }
            )
            for item in episode.evidence_catalog
        ]
        result.append(episode.model_copy(update={"evidence_catalog": catalog}))
    return result


def _load_rollouts(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"no rollout rows in {path}")
    required = {"task_id", "budget", "prediction", "selected_evidence_ids"}
    for index, row in enumerate(rows, 1):
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"rollout row {index} missing fields: {missing}")
    return rows


def _terminal_operation(value: str) -> EditOperation:
    if value == "REJECT":
        return EditOperation.KEEP
    prefix = "COMMIT:"
    if not value.startswith(prefix):
        raise ValueError(f"unsupported terminal action: {value}")
    return EditOperation(value[len(prefix) :])


def _operation_errors(
    target: EditOperation, prediction: EditOperation
) -> tuple[bool, bool, bool]:
    false_edit = target == EditOperation.KEEP and prediction != target
    missed_edit = target != EditOperation.KEEP and prediction == EditOperation.KEEP
    wrong_edit = (
        target != EditOperation.KEEP
        and prediction != EditOperation.KEEP
        and prediction != target
    )
    return false_edit, missed_edit, wrong_edit


def _executable_operation_errors(
    target_action: str,
    effective_operation: EditOperation,
) -> tuple[bool, bool, bool]:
    """Score safety from the map delta that was actually executed."""

    target_operation = _terminal_operation(target_action)
    map_changed = effective_operation != EditOperation.KEEP
    target_requires_edit = target_operation != EditOperation.KEEP
    false_edit = not target_requires_edit and map_changed
    missed_edit = target_requires_edit and not map_changed
    wrong_edit = (
        target_requires_edit
        and map_changed
        and effective_operation != target_operation
    )
    return false_edit, missed_edit, wrong_edit


def _road_width_source_pixels(episode: EpisodeRecord) -> float | None:
    value = episode.metadata.get("road_width_source_pixels")
    if value is None:
        return None
    width = float(value)
    if width <= 0.0:
        raise ValueError("road_width_source_pixels must be positive when provided")
    return width


def _temporal_pair_input(updater: Any) -> bool:
    """Read the input modality from the pinned updater checkpoint wrapper.

    Writeback must consume the same modality as selector-oracle construction.
    In particular, a six-channel temporal updater cannot be evaluated with two
    copies of the current RGB image in place of its prior/current pair.
    """

    config = getattr(getattr(updater, "model", None), "config", None)
    if config is None:
        raise TypeError("updater must expose model.config for writeback evaluation")
    temporal_pair_input = bool(getattr(config, "temporal_pair_input", False))
    image_channels = int(getattr(config, "image_channels", 0))
    if image_channels <= 0:
        raise ValueError("updater model config has an invalid image channel count")
    if temporal_pair_input and image_channels != 6:
        raise ValueError("temporal-pair writeback requires a six-channel updater")
    return temporal_pair_input


def _transform(item: EvidenceItem, image_size: int) -> Affine:
    x_min, y_min, x_max, y_max = item.region
    window = Window(x_min, y_min, x_max - x_min, y_max - y_min)
    if Path(item.image_path).suffix.lower() in {".jpg", ".jpeg"} and item.udm_path is None:
        return Affine.translation(x_min, y_min) * Affine.scale(
            float(window.width) / image_size, float(window.height) / image_size
        )
    with rasterio.open(item.image_path) as dataset:
        return dataset.window_transform(window) * Affine.scale(
            float(window.width) / image_size, float(window.height) / image_size
        )


def evaluate(
    updater: Any,
    episodes: list[EpisodeRecord],
    rollout_rows: list[dict[str, Any]],
    *,
    image_size: int,
    threshold: float,
    simplify_tolerance: float,
    min_delta_component_pixels: int = 0,
    add_min_delta_component_pixels: int | None = None,
    delete_min_delta_component_pixels: int | None = None,
    reshape_min_delta_component_pixels: int | None = None,
    preserve_largest_delta_component: bool = False,
    delta_margin: float = 0.0,
    confidence_floor: float = 0.0,
    evidence_fusion: str = "confidence_weighted",
    artifact_dir: Path | None = None,
    protocol_name: str = "muno21-road-footprint-vector-delta-v1",
    prior_input_translation_pixels: int = 0,
    prior_input_morphology: str = "none",
    prior_input_morphology_pixels: int = 0,
    corruption_seed: int = 0,
    progress_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if prior_input_translation_pixels < 0:
        raise ValueError("prior_input_translation_pixels must be non-negative")
    temporal_pair_input = _temporal_pair_input(updater)
    operation_component_pixels = {
        EditOperation.ADD: add_min_delta_component_pixels,
        EditOperation.DELETE: delete_min_delta_component_pixels,
        EditOperation.RESHAPE: reshape_min_delta_component_pixels,
    }
    for operation, pixels in operation_component_pixels.items():
        if pixels is not None and pixels < 0:
            raise ValueError(
                f"{operation.value.lower()} minimum component pixels must be non-negative"
            )
    morph_prior_no_wrap(
        np.zeros((1, 1), dtype=np.float32),
        operation=prior_input_morphology,
        pixels=prior_input_morphology_pixels,
    )
    splits = {episode.split for episode in episodes}
    if len(splits) != 1:
        raise ValueError(f"writeback episodes must use one split: {sorted(splits)}")
    test_assets_read = splits == {"test"}
    writeback_evidence_modes = {
        str(row.get("writeback_evidence_mode", "all")) for row in rollout_rows
    }
    if len(writeback_evidence_modes) != 1:
        raise ValueError("writeback rollouts must use one evidence mode")
    writeback_evidence_mode = writeback_evidence_modes.pop()
    if test_assets_read:
        from activemap.frozen_test import assert_frozen_test_access

        assert_frozen_test_access()
    by_task = {public_task_id(episode.episode_id): episode for episode in episodes}
    prediction_cache: dict[tuple[str, str], EvidenceMaskPrediction] = {}
    raster_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, Affine]] = {}
    output_rows = []
    total_rows = len(rollout_rows)
    for index, row in enumerate(
        tqdm(rollout_rows, desc="Agent map writeback"),
        start=1,
    ):
        task_id = str(row["task_id"])
        if task_id not in by_task:
            raise ValueError(f"rollout task does not match an episode: {task_id}")
        episode = by_task[task_id]
        if row.get("aoi_id") is not None and str(row["aoi_id"]) != str(episode.aoi_id):
            raise ValueError(f"rollout AOI does not match episode for {task_id}")
        catalog = {item.evidence_id: item for item in episode.evidence_catalog}
        anchor = next(
            (
                item
                for item in episode.evidence_catalog
                if item.timestamp == episode.anchor_timestamp
            ),
            episode.evidence_catalog[-1],
        )
        if task_id not in raster_cache:
            _, prior, target, valid, _ = _read_candidate(
                anchor,
                prior_geometry=_geometry(episode.prior_geometry),
                target_geometry=_geometry(episode.target_geometry),
                image_size=image_size,
                image_channels=updater.model.config.image_channels,
                temporal_pair_input=temporal_pair_input,
                road_width_source_pixels=_road_width_source_pixels(episode),
            )
            raster_cache[task_id] = (prior, target, valid, _transform(anchor, image_size))
        prior, target, valid, transform = raster_cache[task_id]
        _, prior_input_shift = deterministic_prior_translation(
            prior,
            identity=episode.episode_id,
            max_pixels=prior_input_translation_pixels,
            seed=corruption_seed,
        )
        selected_predictions = []
        for exposed_evidence_id in row["selected_evidence_ids"]:
            exposed_evidence_id = str(exposed_evidence_id)
            evidence_id = resolve_evidence_id(exposed_evidence_id, list(catalog))
            key = (task_id, evidence_id)
            if key not in prediction_cache:
                image, evidence_prior, _, _, _ = _read_candidate(
                    catalog[evidence_id],
                    prior_geometry=_geometry(episode.prior_geometry),
                    target_geometry=_geometry(episode.target_geometry),
                    image_size=image_size,
                    image_channels=updater.model.config.image_channels,
                    temporal_pair_input=temporal_pair_input,
                    road_width_source_pixels=_road_width_source_pixels(episode),
                )
                model_prior, _ = deterministic_prior_translation(
                    evidence_prior,
                    identity=episode.episode_id,
                    max_pixels=prior_input_translation_pixels,
                    seed=corruption_seed,
                )
                model_prior = morph_prior_no_wrap(
                    model_prior,
                    operation=prior_input_morphology,
                    pixels=prior_input_morphology_pixels,
                )
                prediction = updater.predict(image, model_prior)
                prediction_cache[key] = EvidenceMaskPrediction(
                    evidence_id=exposed_evidence_id,
                    target_probability=np.asarray(prediction["mask_probability"]),
                    confidence=float(prediction["confidence"]),
                )
            selected_predictions.append(prediction_cache[key])
        predicted_operation = _terminal_operation(str(row["prediction"]))
        operation_min_pixels = operation_component_pixels.get(predicted_operation)
        if operation_min_pixels is None:
            operation_min_pixels = min_delta_component_pixels
        metrics = evaluate_typed_writeback(
            selected_predictions,
            operation=predicted_operation,
            prior=prior,
            target=target,
            valid=valid,
            transform=transform,
            threshold=threshold,
            simplify_tolerance=simplify_tolerance,
            min_delta_component_pixels=operation_min_pixels,
            preserve_largest_delta_component=preserve_largest_delta_component,
            delta_margin=delta_margin,
            confidence_floor=confidence_floor,
            evidence_fusion=evidence_fusion,
            return_artifacts=artifact_dir is not None,
        )
        target_operation = _terminal_operation(str(row["target"]))
        effective_operation = EditOperation(str(metrics["effective_operation"]))
        policy_false_edit, policy_missed_edit, policy_wrong_edit = _operation_errors(
            target_operation, predicted_operation
        )
        false_edit, missed_edit, wrong_edit = _executable_operation_errors(
            str(row["target"]), effective_operation
        )
        utility_scores = score_episode_profiles(
            final_map_quality=float(metrics["raster_iou"]),
            prior_map_quality=float(metrics["prior_raster_iou"]),
            spent_cost=float(row.get("spent_cost", 0.0)),
            budget=float(row["budget"]),
            false_edit=false_edit,
            missed_edit=missed_edit,
            wrong_edit=wrong_edit,
            topology_valid=bool(metrics["vector_delta_topology_valid"]),
        )
        artifact_path = None
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            suffix = str(float(row["budget"])).replace(".", "p")
            artifact_path = artifact_dir / f"{task_id}__b{suffix}.npz"
            np.savez_compressed(
                artifact_path,
                committed_mask=metrics.pop("_committed_mask"),
                prior_mask=metrics.pop("_prior_mask"),
                target_mask=metrics.pop("_target_mask"),
                valid_mask=metrics.pop("_valid_mask"),
                transform=np.asarray(list(transform)[:6], dtype=np.float64),
            )
        output_rows.append(
            {
                "task_id": task_id,
                "aoi_id": str(episode.aoi_id),
                "budget": float(row["budget"]),
                "target": row.get("target"),
                "prediction": row["prediction"],
                "source_selected_evidence_ids": row.get(
                    "source_selected_evidence_ids", row["selected_evidence_ids"]
                ),
                "writeback_evidence_mode": writeback_evidence_mode,
                "source_example_id": row.get("source_example_id"),
                "semantic_tool_called": row.get("semantic_tool_called"),
                "semantic_tool_cost": row.get("semantic_tool_cost"),
                "policy_utility": row.get("policy_utility"),
                "spent_cost": float(row.get("spent_cost", 0.0)),
                "false_edit": false_edit,
                "missed_edit": missed_edit,
                "wrong_edit": wrong_edit,
                "policy_false_edit": policy_false_edit,
                "policy_missed_edit": policy_missed_edit,
                "policy_wrong_edit": policy_wrong_edit,
                "episode_utility_v2": utility_scores,
                "episode_utility_v2_balanced": utility_scores["balanced"]["value"],
                "episode_utility_v2_safety": utility_scores["safety"]["value"],
                "episode_utility_v2_cost_aware": utility_scores["cost_aware"]["value"],
                "map_quality_before": float(metrics["prior_raster_iou"]),
                "map_quality_after": float(metrics["raster_iou"]),
                "topology_quality_before": float(metrics["topology_quality_before"]),
                "topology_quality_after": float(metrics["topology_quality_after"]),
                "mask_artifact": str(artifact_path) if artifact_path else None,
                "split": episode.split,
                "test_assets_read": test_assets_read,
                "prior_input_shift_y": prior_input_shift[0],
                "prior_input_shift_x": prior_input_shift[1],
                **metrics,
            }
        )
        if progress_path is not None and (
            index == 1 or index % 25 == 0 or index == total_rows
        ):
            _write_progress(
                progress_path,
                {
                    "status": "running" if index < total_rows else "inference_complete",
                    "rows_processed": index,
                    "rows_total": total_rows,
                    "split": next(iter(splits)),
                    "protocol_name": protocol_name,
                    "test_assets_read": test_assets_read,
                },
            )

    by_budget: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in output_rows:
        by_budget[row["budget"]].append(row)
    metric_names = (
        "raster_iou",
        "prior_raster_iou",
        "raster_iou_gain",
        "added_change_iou",
        "removed_change_iou",
        "added_polygon_iou",
        "removed_polygon_iou",
        "vector_replay_iou",
        "component_count_absolute_error",
        "prior_component_count_absolute_error",
    )
    summaries = []
    for budget, selected in sorted(by_budget.items()):
        summary = {"budget": budget, "sample_count": len(selected)}
        summary.update(
            {
                f"mean_{name}": float(np.mean([row[name] for row in selected]))
                for name in metric_names
            }
        )
        summary["vector_delta_topology_valid_rate"] = float(
            np.mean([row["vector_delta_topology_valid"] for row in selected])
        )
        for profile in ("balanced", "safety", "cost_aware"):
            summary[f"mean_episode_utility_v2_{profile}"] = float(
                np.mean([row[f"episode_utility_v2_{profile}"] for row in selected])
            )
        summaries.append(summary)
    return output_rows, {
        "protocol": {
            "name": protocol_name,
            "fusion": evidence_fusion,
            "writeback_evidence_mode": writeback_evidence_mode,
            "temporal_pair_input": temporal_pair_input,
            "typed_constraints": {
                "KEEP": "prior",
                "ADD": "max(prior, fused_target)",
                "DELETE": "min(prior, fused_target)",
                "RESHAPE": "fused_target",
            },
            "centerline_apls_available": False,
            "geometry_rasterization": (
                "buffered_polyline only when road_width_source_pixels is explicitly present; "
                "otherwise native polygon"
            ),
            "centerline_export_inputs_retained": artifact_dir is not None,
            "min_delta_component_pixels": min_delta_component_pixels,
            "operation_min_delta_component_pixels": {
                "ADD": add_min_delta_component_pixels,
                "DELETE": delete_min_delta_component_pixels,
                "RESHAPE": reshape_min_delta_component_pixels,
            },
            "preserve_largest_delta_component": preserve_largest_delta_component,
            "delta_margin": delta_margin,
            "confidence_floor": confidence_floor,
            "prior_input_corruption": {
                "translation_pixels": prior_input_translation_pixels,
                "morphology": prior_input_morphology,
                "morphology_pixels": prior_input_morphology_pixels,
                "corruption_seed": corruption_seed,
                "scope": "model_input_only",
            },
            "test_assets_read": test_assets_read,
            "executable_error_protocol": (
                "v2: REJECT+changed-map=false-edit; COMMIT+unchanged-map=missed-edit; "
                "COMMIT+wrong-effective-operation=wrong-edit"
            ),
            "episode_utility_v2": utility_protocol(
                quality_source="executable_writeback_raster_iou",
                paper_primary=True,
                terminal_error_source="effective_executable_map_delta",
            ),
        },
        "sample_count": len(output_rows),
        "unique_inferences": len(prediction_cache),
        "budgets": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("updater_checkpoint", type=Path)
    parser.add_argument("episodes", type=Path)
    parser.add_argument("rollouts", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--frozen-test", action="store_true")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--simplify-tolerance", type=float, default=0.0)
    parser.add_argument("--min-delta-component-pixels", type=int, default=0)
    parser.add_argument("--add-min-delta-component-pixels", type=int)
    parser.add_argument("--delete-min-delta-component-pixels", type=int)
    parser.add_argument("--reshape-min-delta-component-pixels", type=int)
    parser.add_argument("--preserve-largest-delta-component", action="store_true")
    parser.add_argument("--delta-margin", type=float, default=0.0)
    parser.add_argument("--confidence-floor", type=float, default=0.0)
    parser.add_argument("--prior-input-translation-pixels", type=int, default=0)
    parser.add_argument(
        "--prior-input-morphology",
        choices=("none", "dilate", "erode"),
        default="none",
    )
    parser.add_argument("--prior-input-morphology-pixels", type=int, default=0)
    parser.add_argument("--corruption-seed", type=int, default=0)
    parser.add_argument(
        "--evidence-fusion",
        choices=("confidence_weighted", "max_confidence"),
        default="confidence_weighted",
    )
    parser.add_argument(
        "--protocol-name", default="muno21-road-footprint-vector-delta-v1"
    )
    parser.add_argument("--asset-root-map", action="append", default=[])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if args.split == "test":
        if not args.frozen_test:
            raise PermissionError("test writeback requires --frozen-test")
        from activemap.frozen_test import assert_frozen_test_access

        assert_frozen_test_access()
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise FileExistsError(
                f"refusing to overwrite frozen test writeback: {args.output_dir}"
            )
    elif args.frozen_test:
        raise ValueError("--frozen-test is valid only with --split test")

    from activemap.inference import UpdaterPredictor

    asset_root_maps = _parse_asset_root_maps(args.asset_root_map)
    episodes = _remap_episode_assets(
        load_episodes(args.episodes, split=args.split), asset_root_maps
    )
    rollout_rows = _load_rollouts(args.rollouts)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        rollout_rows = rollout_rows[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "progress.json"
    rows, summary = evaluate(
        UpdaterPredictor(args.updater_checkpoint, device=args.device),
        episodes,
        rollout_rows,
        image_size=args.image_size,
        threshold=args.threshold,
        simplify_tolerance=args.simplify_tolerance,
        min_delta_component_pixels=args.min_delta_component_pixels,
        add_min_delta_component_pixels=args.add_min_delta_component_pixels,
        delete_min_delta_component_pixels=args.delete_min_delta_component_pixels,
        reshape_min_delta_component_pixels=args.reshape_min_delta_component_pixels,
        preserve_largest_delta_component=args.preserve_largest_delta_component,
        delta_margin=args.delta_margin,
        confidence_floor=args.confidence_floor,
        evidence_fusion=args.evidence_fusion,
        artifact_dir=args.output_dir / "mask_artifacts",
        protocol_name=args.protocol_name,
        prior_input_translation_pixels=args.prior_input_translation_pixels,
        prior_input_morphology=args.prior_input_morphology,
        prior_input_morphology_pixels=args.prior_input_morphology_pixels,
        corruption_seed=args.corruption_seed,
        progress_path=progress_path,
    )
    summary["protocol"]["asset_root_maps"] = [
        {"source": str(source), "target": str(target)}
        for source, target in asset_root_maps
    ]
    with (args.output_dir / "writeback.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _write_progress(
        progress_path,
        {
            "status": "complete",
            "rows_processed": len(rows),
            "rows_total": len(rows),
            "split": args.split,
            "protocol_name": args.protocol_name,
            "test_assets_read": args.split == "test",
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
