"""Checkpoint inference and standardized evaluation for the raster updater."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from affine import Affine
from shapely.geometry import mapping, shape
from torch import Tensor
from torch.utils.data import DataLoader

from activemap.evaluation.road_topology import road_connectivity_metrics
from activemap.evaluation.statistics import grouped_bootstrap_intervals
from activemap.evaluation.update import (
    UpdatePrediction,
    evaluate_updates,
    object_scale_strata_metrics,
    save_update_evaluation,
    segmentation_strata_metrics,
    write_update_predictions,
)
from activemap.geometry import geometry_iou
from activemap.models import EditOperation, GeoJSONGeometry
from activemap.nn.updater import (
    PriorConditionedUNet,
    UpdaterConfig,
    hierarchical_edit_predictions,
    operation_probabilities,
)
from activemap.training.selector import resolve_device
from activemap.training.updater_data import EDIT_TO_INDEX, UpdaterDataset
from activemap.updater_records import load_updater_samples
from activemap.vector_map import topology_is_valid, vectorize_mask

INDEX_TO_EDIT = {index: operation for operation, index in EDIT_TO_INDEX.items()}


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def evaluate_updater_checkpoint(
    checkpoint_path: Path,
    samples_path: Path,
    output_dir: Path,
    *,
    split: str = "test",
    device: str = "auto",
    batch_size: int = 32,
    num_workers: int = 0,
    commit_threshold: float = 0.0,
    bootstrap_iterations: int = 0,
    bootstrap_seed: int = 20260710,
    presence_threshold: float | None = None,
    change_threshold: float = 0.5,
    add_threshold: float = 0.5,
    remove_threshold: float = 0.5,
    edit_decoding: str = "auto",
    road_topology_tolerance: int | None = None,
) -> dict[str, Any]:
    """Run inference once and emit per-sample records, summary, and optional CIs."""

    if not 0.0 <= commit_threshold <= 1.0:
        raise ValueError("commit_threshold must be between zero and one")
    if edit_decoding not in {"auto", "hierarchical", "auxiliary"}:
        raise ValueError("edit_decoding must be auto, hierarchical, or auxiliary")
    if not 0.0 <= add_threshold <= 1.0 or not 0.0 <= remove_threshold <= 1.0:
        raise ValueError("ADD and REMOVE thresholds must be between zero and one")
    if road_topology_tolerance is not None and road_topology_tolerance < 0:
        raise ValueError("road topology tolerance must be non-negative")
    target_device = resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=target_device, weights_only=False)
    model = PriorConditionedUNet(UpdaterConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(target_device).eval()
    samples = load_updater_samples(samples_path, split=split)
    sample_by_id = {sample.sample_id: sample for sample in samples}
    loader = DataLoader(
        UpdaterDataset(samples, temporal_pair_input=model.config.temporal_pair_input),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    records: list[UpdatePrediction] = []
    added_ious: list[float] = []
    removed_ious: list[float] = []
    road_topology_values: list[dict[str, Any]] = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = _to_device(raw_batch, target_device)
            outputs = model(batch["image"], batch["prior_mask"])
            prior_binary = batch["prior_mask"] >= 0.5
            if "temporal_change_logits" in outputs:
                temporal_probabilities = torch.sigmoid(outputs["temporal_change_logits"])
                predicted_masks = torch.where(
                    prior_binary,
                    temporal_probabilities[:, 1:2] < remove_threshold,
                    temporal_probabilities[:, 0:1] >= add_threshold,
                )
            else:
                predicted_masks = torch.sigmoid(outputs["segmentation_logits"]) >= 0.5
            target_masks = batch["target_mask"] >= 0.5
            valid_masks = batch["valid_mask"] >= 0.5
            intersection = (
                ((predicted_masks & target_masks) & valid_masks).sum(dim=(1, 2, 3)).float()
            )
            union = ((predicted_masks | target_masks) & valid_masks).sum(dim=(1, 2, 3)).float()
            iou = torch.where(union > 0, intersection / union, torch.ones_like(union))
            target_pixels = (target_masks & valid_masks).sum(dim=(1, 2, 3))
            predicted_pixels = (predicted_masks & valid_masks).sum(dim=(1, 2, 3)).float()
            valid_pixels = valid_masks.sum(dim=(1, 2, 3)).float().clamp_min(1.0)
            hierarchical_edits = hierarchical_edit_predictions(
                outputs,
                batch["prior_mask"],
                presence_threshold=presence_threshold,
                change_threshold=change_threshold,
            )
            auxiliary_edits = torch.argmax(
                operation_probabilities(outputs, batch["prior_mask"]), dim=-1
            )
            if edit_decoding == "hierarchical":
                predicted_edits = hierarchical_edits
            elif edit_decoding == "auxiliary":
                predicted_edits = auxiliary_edits
            else:
                full_scene = torch.tensor(
                    [value == "full_scene_temporal" for value in raw_batch["supervision_type"]],
                    device=target_device,
                )
                predicted_edits = torch.where(full_scene, auxiliary_edits, hierarchical_edits)
            confidence = torch.sigmoid(outputs["confidence_logits"])
            for index, sample_id in enumerate(raw_batch["sample_id"]):
                sample = sample_by_id[str(sample_id)]
                target_index = int(batch["edit_target"][index].detach().cpu())
                predicted_index = int(predicted_edits[index].detach().cpu())
                value = float(confidence[index].detach().cpu())
                predicted_operation = INDEX_TO_EDIT[predicted_index]
                prior_mask = batch["prior_mask"][index] >= 0.5
                predicted_mask = predicted_masks[index]
                target_mask = target_masks[index]
                valid_mask = valid_masks[index]
                target_added = target_mask & ~prior_mask & valid_mask
                predicted_added = predicted_mask & ~prior_mask & valid_mask
                target_removed = prior_mask & ~target_mask & valid_mask
                predicted_removed = prior_mask & ~predicted_mask & valid_mask

                def change_iou(left: Tensor, right: Tensor) -> float:
                    change_union = (left | right).sum().float()
                    if not bool(change_union.detach().cpu()):
                        return 1.0
                    return float(((left & right).sum().float() / change_union).detach().cpu())

                added_iou = change_iou(predicted_added, target_added)
                removed_iou = change_iou(predicted_removed, target_removed)
                road_topology = None
                if road_topology_tolerance is not None:
                    road_topology = road_connectivity_metrics(
                        predicted_mask.detach().cpu().numpy(),
                        target_mask.detach().cpu().numpy(),
                        valid_mask.detach().cpu().numpy(),
                        tolerance_pixels=road_topology_tolerance,
                    )
                    road_topology_values.append(road_topology)
                if bool(target_added.any().detach().cpu()):
                    added_ious.append(added_iou)
                if bool(target_removed.any().detach().cpu()):
                    removed_ious.append(removed_iou)
                predicted_geometry = None
                polygon_iou = None
                topology_valid = None
                if sample.crop_transform is not None:
                    probability_mask = predicted_masks[index, 0].detach().cpu().numpy()
                    geometry = vectorize_mask(
                        probability_mask,
                        Affine(*sample.crop_transform),
                    )
                    target_geometry = (
                        shape(sample.target_geometry.model_dump(mode="json"))
                        if sample.target_geometry is not None
                        else None
                    )
                    if geometry is None and target_geometry is None:
                        polygon_iou = 1.0
                    elif geometry is None or target_geometry is None:
                        polygon_iou = 0.0
                    else:
                        polygon_iou = geometry_iou(geometry, target_geometry)
                    if predicted_operation in {EditOperation.ADD, EditOperation.RESHAPE}:
                        topology_valid = topology_is_valid(geometry)
                        if geometry is not None:
                            predicted_geometry = GeoJSONGeometry.model_validate(mapping(geometry))
                    else:
                        topology_valid = True
                records.append(
                    UpdatePrediction(
                        sample_id=str(sample_id),
                        aoi_id=str(raw_batch["aoi_id"][index]),
                        target_edit=INDEX_TO_EDIT[target_index],
                        predicted_edit=predicted_operation,
                        confidence=value,
                        committed=value >= commit_threshold,
                        raster_iou=float(iou[index].detach().cpu()),
                        polygon_iou=polygon_iou,
                        topology_valid=topology_valid,
                        object_id=sample.object_id,
                        predicted_geometry=predicted_geometry,
                        metadata={
                            "split": split,
                            "target_foreground": bool(target_pixels[index].detach().cpu()),
                            "target_foreground_pixels": int(
                                target_pixels[index].detach().cpu()
                            ),
                            "target_foreground_fraction": float(
                                (target_pixels[index].float() / valid_pixels[index])
                                .detach()
                                .cpu()
                            ),
                            "empty_scene_false_positive_fraction": (
                                None
                                if bool(target_pixels[index].detach().cpu())
                                else float(
                                    (predicted_pixels[index] / valid_pixels[index]).detach().cpu()
                                )
                            ),
                            "presence_probability": (
                                float(
                                    torch.sigmoid(outputs["presence_logits"])[index].detach().cpu()
                                )
                                if "presence_logits" in outputs
                                else None
                            ),
                            "change_probability": (
                                float(torch.sigmoid(outputs["change_logits"])[index].detach().cpu())
                                if "change_logits" in outputs
                                else None
                            ),
                            "added_change_iou": added_iou,
                            "removed_change_iou": removed_iou,
                            "target_added_pixels": int(target_added.sum().detach().cpu()),
                            "target_removed_pixels": int(target_removed.sum().detach().cpu()),
                            "road_topology": road_topology,
                        },
                    )
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.jsonl"
    write_update_predictions(records, prediction_path)
    summary = evaluate_updates(records)
    summary["segmentation_strata"] = segmentation_strata_metrics(records)
    summary["object_scale_strata"] = object_scale_strata_metrics(records)
    summary["temporal_change"] = {
        "added_sample_count": len(added_ious),
        "removed_sample_count": len(removed_ious),
        "mean_added_iou": (float(sum(added_ious) / len(added_ious)) if added_ious else None),
        "mean_removed_iou": (
            float(sum(removed_ious) / len(removed_ious)) if removed_ious else None
        ),
    }
    if road_topology_values:
        summary["road_topology"] = {
            "sample_count": len(road_topology_values),
            "tolerance_pixels": road_topology_tolerance,
            "mean_connectivity_precision": float(
                sum(value["connectivity_precision"] for value in road_topology_values)
                / len(road_topology_values)
            ),
            "mean_connectivity_recall": float(
                sum(value["connectivity_recall"] for value in road_topology_values)
                / len(road_topology_values)
            ),
            "mean_connectivity_f1": float(
                sum(value["connectivity_f1"] for value in road_topology_values)
                / len(road_topology_values)
            ),
            "mean_absolute_component_count_error": float(
                sum(abs(value["component_count_error"]) for value in road_topology_values)
                / len(road_topology_values)
            ),
        }
    summary.update(
        {
            "checkpoint": str(checkpoint_path.resolve()),
            "samples": str(samples_path.resolve()),
            "split": split,
            "device": str(target_device),
            "commit_threshold": commit_threshold,
            "prediction_path": str(prediction_path.resolve()),
            "model_config": checkpoint["model_config"],
            "presence_threshold": presence_threshold,
            "change_threshold": change_threshold,
            "add_threshold": add_threshold,
            "remove_threshold": remove_threshold,
            "edit_decoding": edit_decoding,
            "road_topology_tolerance": road_topology_tolerance,
        }
    )
    if bootstrap_iterations:
        summary["bootstrap"] = grouped_bootstrap_intervals(
            records,
            iterations=bootstrap_iterations,
            seed=bootstrap_seed,
        )
    save_update_evaluation(summary, output_dir / "summary.json")
    (output_dir / "checkpoint_metadata.json").write_text(
        json.dumps(
            {
                key: value
                for key, value in checkpoint.items()
                if key not in {"state_dict", "optimizer_state_dict"}
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary
