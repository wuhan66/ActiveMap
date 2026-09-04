#!/usr/bin/env python3
"""Evaluate true online editable-map maintenance over chronological chains.

Unlike the legacy chronological stress test, this evaluator rasterizes the
committed carry state at every timestamp and runs the frozen updater again.
The canonical episode prior is used only by the independent-reset reference;
it never replaces a carried prior for either online branch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry

from activemap.agent.writeback import (
    EvidenceMaskPrediction,
    evaluate_typed_writeback,
)
from activemap.data.raster_masks import rasterize_geometry_mask
from activemap.models import EditOperation, EpisodeRecord, EvidenceItem


EDIT_ORDER = list(EditOperation)
BranchName = str


@dataclass(frozen=True)
class SafeCommitGate:
    """Observable, fixed terminal gate used only by the online safe branch."""

    confidence_threshold: float
    replay_iou_threshold: float
    require_topology: bool = True

    def accepts(self, metrics: dict[str, Any]) -> bool:
        if not bool(metrics["writeback_changed"]):
            return True
        return bool(
            float(metrics["fused_confidence"]) >= self.confidence_threshold
            and float(metrics["vector_replay_iou"]) >= self.replay_iou_threshold
            and (
                not self.require_topology
                or bool(metrics["vector_delta_topology_valid"])
            )
        )


def _digest_geometry(geometry: BaseGeometry | None) -> str:
    if geometry is None or geometry.is_empty:
        return "empty"
    return hashlib.sha256(geometry.wkb).hexdigest()[:16]


def _episode_geometry(value: Any) -> BaseGeometry | None:
    if value is None:
        return None
    geometry = shape(value.model_dump(mode="json") if hasattr(value, "model_dump") else value)
    if geometry.is_empty:
        return None
    return geometry if geometry.is_valid else geometry.buffer(0)


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


def _remap_episode_assets(
    episodes: list[EpisodeRecord], mappings: tuple[tuple[Path, Path], ...]
) -> list[EpisodeRecord]:
    if not mappings:
        return episodes
    def remap(value: str | None) -> str | None:
        if value is None:
            return None
        original = Path(value)
        for source, target in mappings:
            try:
                return str(target / original.relative_to(source))
            except ValueError:
                continue
        return value
    return [
        episode.model_copy(
            update={
                "evidence_catalog": [
                    item.model_copy(
                        update={
                            "image_path": remap(item.image_path),
                            "udm_path": remap(item.udm_path),
                            "prior_image_path": remap(item.prior_image_path),
                            "prior_udm_path": remap(item.prior_udm_path),
                        }
                    )
                    for item in episode.evidence_catalog
                ]
            }
        )
        for episode in episodes
    ]


def _apply_delta(
    prior: BaseGeometry | None,
    add_payload: dict[str, Any] | None,
    remove_payload: dict[str, Any] | None,
) -> BaseGeometry | None:
    current = prior
    added = shape(add_payload) if add_payload is not None else None
    removed = shape(remove_payload) if remove_payload is not None else None
    if added is not None and not added.is_empty:
        current = added if current is None or current.is_empty else current.union(added)
    if current is not None and not current.is_empty and removed is not None and not removed.is_empty:
        current = current.difference(removed)
    if current is None or current.is_empty:
        return None
    return current if current.is_valid else current.buffer(0)


def _same_state(left: BaseGeometry | None, right: BaseGeometry | None, tolerance: float) -> bool:
    if left is None or left.is_empty:
        return right is None or right.is_empty
    if right is None or right.is_empty:
        return False
    union = left.union(right).area
    iou = left.intersection(right).area / union if union > 0 else 1.0
    return iou >= 1.0 - tolerance


def build_contiguous_chains(
    episodes: list[EpisodeRecord], *, minimum_length: int, continuity_tolerance: float
) -> list[list[EpisodeRecord]]:
    grouped: dict[tuple[str, str], list[EpisodeRecord]] = defaultdict(list)
    for episode in episodes:
        if episode.aoi_id and episode.hypothesis.object_id and episode.anchor_timestamp:
            grouped[(str(episode.aoi_id), str(episode.hypothesis.object_id))].append(episode)

    chains: list[list[EpisodeRecord]] = []
    for values in grouped.values():
        values.sort(key=lambda item: (str(item.anchor_timestamp), item.episode_id))
        current: list[EpisodeRecord] = []
        previous_target: BaseGeometry | None = None
        previous_timestamp: str | None = None
        for episode in values:
            canonical_prior = _episode_geometry(episode.prior_geometry)
            timestamp = str(episode.anchor_timestamp)
            continuous = bool(
                current
                and timestamp > str(previous_timestamp)
                and _same_state(previous_target, canonical_prior, continuity_tolerance)
            )
            if current and not continuous:
                if len(current) >= minimum_length:
                    chains.append(current)
                current = []
            current.append(episode)
            previous_target = _episode_geometry(episode.target_geometry)
            previous_timestamp = timestamp
        if len(current) >= minimum_length:
            chains.append(current)
    return chains


def _anchor(episode: EpisodeRecord) -> EvidenceItem:
    return next(
        (item for item in episode.evidence_catalog if item.timestamp == episode.anchor_timestamp),
        episode.evidence_catalog[-1],
    )


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


def _operation_errors(target: EditOperation, prediction: EditOperation) -> tuple[bool, bool, bool]:
    false_edit = target == EditOperation.KEEP and prediction != EditOperation.KEEP
    missed_edit = target != EditOperation.KEEP and prediction == EditOperation.KEEP
    wrong_edit = (
        target != EditOperation.KEEP
        and prediction != EditOperation.KEEP
        and prediction != target
    )
    return false_edit, missed_edit, wrong_edit


def _step_inputs(
    episode: EpisodeRecord,
    prior_geometry: BaseGeometry | None,
    *,
    image_size: int,
    image_channels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Affine, str]:
    item = _anchor(episode)
    # This import keeps geometry-only protocol tests independent of PyTorch.
    from activemap.oracle.updater_counterfactual import _read_candidate

    target_geometry = _episode_geometry(episode.target_geometry)
    image, prior, target, valid, _ = _read_candidate(
        item,
        prior_geometry=prior_geometry,
        target_geometry=target_geometry,
        image_size=image_size,
        image_channels=image_channels,
        road_width_source_pixels=(
            float(episode.metadata["road_width_source_pixels"])
            if episode.metadata.get("road_width_source_pixels") is not None
            else None
        ),
    )
    return image, prior, target, valid, _transform(item, image_size), item.evidence_id


InputReader = Callable[
    [EpisodeRecord, BaseGeometry | None], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Affine, str]
]


def _run_branch_step(
    predictor: Any,
    episode: EpisodeRecord,
    prior_geometry: BaseGeometry | None,
    *,
    branch: BranchName,
    gate: SafeCommitGate | None,
    input_reader: InputReader,
    threshold: float,
    delta_margin: float,
    min_delta_component_pixels: int,
) -> tuple[BaseGeometry | None, dict[str, Any]]:
    image, prior, target, valid, transform, evidence_id = input_reader(episode, prior_geometry)
    prediction = predictor.predict(image, prior)
    operation = EDIT_ORDER[int(np.argmax(np.asarray(prediction["edit_probabilities"]))) ]
    metrics = evaluate_typed_writeback(
        [
            EvidenceMaskPrediction(
                evidence_id=evidence_id,
                target_probability=np.asarray(prediction["mask_probability"]),
                confidence=float(prediction["confidence"]),
            )
        ],
        operation=operation,
        prior=prior,
        target=target,
        valid=valid,
        transform=transform,
        threshold=threshold,
        delta_margin=delta_margin,
        min_delta_component_pixels=min_delta_component_pixels,
        return_artifacts=False,
    )
    accepted = True if gate is None else gate.accepts(metrics)
    effective = EditOperation(str(metrics["effective_operation"])) if accepted else EditOperation.KEEP
    next_state = (
        _apply_delta(
            prior_geometry,
            metrics["predicted_add_geometry"],
            metrics["predicted_remove_geometry"],
        )
        if accepted
        else prior_geometry
    )
    false_edit, missed_edit, wrong_edit = _operation_errors(episode.gt_edit.op, effective)
    final_quality = float(metrics["raster_iou"] if accepted else metrics["prior_raster_iou"])
    committed_edit = bool(metrics["writeback_changed"]) and accepted
    record = {
        "branch": branch,
        "input_prior_hash": _digest_geometry(prior_geometry),
        "canonical_prior_hash": _digest_geometry(_episode_geometry(episode.prior_geometry)),
        "input_prior_matches_canonical": _same_state(
            prior_geometry, _episode_geometry(episode.prior_geometry), 1e-9
        ),
        "input_prior_foreground_fraction": float(np.mean(prior >= threshold)),
        "next_state_hash": _digest_geometry(next_state),
        "predicted_operation": operation.value,
        "effective_operation": effective.value,
        "commit_accepted": committed_edit,
        "safe_commit_rejected": bool(metrics["writeback_changed"]) and not accepted,
        "false_edit": false_edit,
        "missed_edit": missed_edit,
        "wrong_edit": wrong_edit,
        "recovered_from_prior_error": bool(
            float(metrics["prior_raster_iou"]) < 1.0 - 1e-6
            and float(metrics["raster_iou"] if accepted else metrics["prior_raster_iou"])
            > float(metrics["prior_raster_iou"]) + 1e-6
        ),
        "final_raster_iou": final_quality,
        "executed_raster_iou_gain": final_quality - float(metrics["prior_raster_iou"]),
        **{
            key: value
            for key, value in metrics.items()
            if key
            not in {
                "operation",
                "effective_operation",
                "predicted_add_geometry",
                "predicted_remove_geometry",
            }
        },
    }
    return next_state, record


def evaluate_chains(
    predictor: Any,
    chains: list[list[EpisodeRecord]],
    *,
    input_reader: InputReader,
    gate: SafeCommitGate,
    threshold: float = 0.5,
    delta_margin: float = 0.15,
    min_delta_component_pixels: int = 0,
) -> list[dict[str, Any]]:
    """Run reset, unconditional carry, and gated carry branches side by side."""

    records: list[dict[str, Any]] = []
    for chain_index, chain in enumerate(chains):
        states = {
            "carry_always_commit": _episode_geometry(chain[0].prior_geometry),
            "carry_safe_commit": _episode_geometry(chain[0].prior_geometry),
        }
        for step, episode in enumerate(chain):
            canonical_prior = _episode_geometry(episode.prior_geometry)
            branch_inputs: list[tuple[BranchName, BaseGeometry | None, SafeCommitGate | None]] = [
                ("independent_reset", canonical_prior, None),
                ("carry_always_commit", states["carry_always_commit"], None),
                ("carry_safe_commit", states["carry_safe_commit"], gate),
            ]
            for branch, prior_geometry, branch_gate in branch_inputs:
                next_state, record = _run_branch_step(
                    predictor,
                    episode,
                    prior_geometry,
                    branch=branch,
                    gate=branch_gate,
                    input_reader=input_reader,
                    threshold=threshold,
                    delta_margin=delta_margin,
                    min_delta_component_pixels=min_delta_component_pixels,
                )
                if branch in states:
                    states[branch] = next_state
                record.update(
                    {
                        "chain_id": f"chain-{chain_index:05d}",
                        "step": step,
                        "task_id": episode.episode_id,
                        "aoi_id": str(episode.aoi_id),
                        "object_id": str(episode.hypothesis.object_id),
                        "timestamp": str(episode.anchor_timestamp),
                        "target_edit": episode.gt_edit.op.value,
                        "prior_source": (
                            "episode_canonical_prior"
                            if branch == "independent_reset"
                            else "carried_committed_state"
                        ),
                        "test_assets_read": False,
                    }
                )
                records.append(record)
    return records


def _mean(rows: list[dict[str, Any]], name: str) -> float:
    return float(np.mean([float(row[name]) for row in rows])) if rows else 0.0


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("no online persistence records")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["branch"])].append(row)
    final_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for branch, rows in grouped.items():
        by_chain: dict[str, dict[str, Any]] = {}
        for row in rows:
            by_chain[str(row["chain_id"])] = row
        final_rows[branch] = list(by_chain.values())

    branches = {}
    for branch, rows in grouped.items():
        final = final_rows[branch]
        branches[branch] = {
            "transition_count": len(rows),
            "chain_count": len(final),
            "mean_step_raster_iou": _mean(rows, "final_raster_iou"),
            "mean_final_chain_raster_iou": _mean(final, "final_raster_iou"),
        "mean_step_raster_iou_gain": _mean(rows, "executed_raster_iou_gain"),
            "false_edit_rate": _mean(rows, "false_edit"),
            "missed_edit_rate": _mean(rows, "missed_edit"),
            "wrong_edit_rate": _mean(rows, "wrong_edit"),
            "commit_rate": _mean(rows, "commit_accepted"),
            "safe_rejection_rate": _mean(rows, "safe_commit_rejected"),
            "recovery_rate_after_prior_error": _mean(rows, "recovered_from_prior_error"),
            "noncanonical_online_input_rate": _mean(
                rows, "input_prior_matches_canonical"
            ),
        }
        branches[branch]["noncanonical_online_input_rate"] = 1.0 - branches[branch][
            "noncanonical_online_input_rate"
        ]

    paired: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in records:
        paired[(str(row["chain_id"]), int(row["step"]))][str(row["branch"])] = row
    complete = [
        row for row in paired.values()
        if set(row) == {"independent_reset", "carry_always_commit", "carry_safe_commit"}
    ]
    comparisons = {}
    for name, left, right in (
        ("always_minus_reset", "carry_always_commit", "independent_reset"),
        ("safe_minus_always", "carry_safe_commit", "carry_always_commit"),
        ("safe_minus_reset", "carry_safe_commit", "independent_reset"),
    ):
        comparisons[name] = {
            "mean_step_raster_iou_delta": float(
                np.mean([item[left]["final_raster_iou"] - item[right]["final_raster_iou"] for item in complete])
            ),
            "false_edit_rate_delta": float(
                np.mean([float(item[left]["false_edit"]) - float(item[right]["false_edit"]) for item in complete])
            ),
            "missed_edit_rate_delta": float(
                np.mean([float(item[left]["missed_edit"]) - float(item[right]["missed_edit"]) for item in complete])
            ),
        }
    return {
        "aoi_count": len({str(row["aoi_id"]) for row in records}),
        "branches": branches,
        "paired_step_comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--minimum-chain-length", type=int, default=2)
    parser.add_argument("--continuity-tolerance", type=float, default=1e-6)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--delta-margin", type=float, default=0.15)
    parser.add_argument("--min-delta-component-pixels", type=int, default=0)
    parser.add_argument("--safe-confidence-threshold", type=float, required=True)
    parser.add_argument("--safe-replay-iou-threshold", type=float, default=0.99)
    parser.add_argument("--asset-root-map", action="append", default=[])
    parser.add_argument("--max-chains", type=int)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    from activemap.oracle.updater_counterfactual import load_episodes

    episodes = _remap_episode_assets(
        [episode for episode in load_episodes(args.episodes) if episode.split == args.split],
        _parse_asset_root_maps(args.asset_root_map),
    )
    chains = build_contiguous_chains(
        episodes,
        minimum_length=args.minimum_chain_length,
        continuity_tolerance=args.continuity_tolerance,
    )
    if args.max_chains is not None:
        chains = chains[: args.max_chains]
    if not chains:
        raise ValueError("no chronological chains on requested split")
    # Keep protocol-only unit tests usable in the lightweight local environment.
    from activemap.inference import UpdaterPredictor

    predictor = UpdaterPredictor(args.checkpoint, device=args.device)
    reader = lambda episode, prior: _step_inputs(
        episode,
        prior,
        image_size=args.image_size,
        image_channels=predictor.model.config.image_channels,
    )
    gate = SafeCommitGate(
        confidence_threshold=args.safe_confidence_threshold,
        replay_iou_threshold=args.safe_replay_iou_threshold,
    )
    records = evaluate_chains(
        predictor,
        chains,
        input_reader=reader,
        gate=gate,
        threshold=args.threshold,
        delta_margin=args.delta_margin,
        min_delta_component_pixels=args.min_delta_component_pixels,
    )
    args.output_dir.mkdir(parents=True)
    trace = args.output_dir / "online_persistent_traces.jsonl"
    trace.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in records),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "activemap-online-persistent-maintenance-v1",
        "split": args.split,
        "test_assets_read": False,
        "protocol": {
            "online_inference": True,
            "per_step_prior": "actual carried committed vector state for online branches",
            "reference": "independent reset with canonical episode prior",
            "branches": ["independent_reset", "carry_always_commit", "carry_safe_commit"],
            "safe_commit_gate": {
                "confidence_threshold": gate.confidence_threshold,
                "replay_iou_threshold": gate.replay_iou_threshold,
                "require_topology": gate.require_topology,
            },
            "target_labels_used_as_model_input": False,
        },
        "chain_count": len(chains),
        "transition_count": sum(len(chain) for chain in chains),
        "metrics": summarize(records),
        "sources": {
            "episodes": str(args.episodes.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "trace": str(trace.resolve()),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
