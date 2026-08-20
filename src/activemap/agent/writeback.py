"""Typed raster-delta writeback and vector replay metrics for road-map patches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from affine import Affine
from rasterio.features import rasterize
from scipy import ndimage
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry

from activemap.models import EditOperation
from activemap.vector_map import topology_is_valid, vectorize_mask


@dataclass(frozen=True)
class EvidenceMaskPrediction:
    evidence_id: str
    target_probability: np.ndarray
    confidence: float


def mask_iou(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float:
    predicted = np.asarray(prediction) >= 0.5
    expected = np.asarray(target) >= 0.5
    visible = np.asarray(valid) >= 0.5
    intersection = np.sum(predicted & expected & visible)
    union = np.sum((predicted | expected) & visible)
    return float(intersection / union) if union else 1.0


def geometry_iou(left: BaseGeometry | None, right: BaseGeometry | None) -> float:
    if left is None or left.is_empty:
        return 1.0 if right is None or right.is_empty else 0.0
    if right is None or right.is_empty:
        return 0.0
    union = left.union(right).area
    return float(left.intersection(right).area / union) if union > 0.0 else 0.0


def _fuse(
    predictions: list[EvidenceMaskPrediction],
    strategy: str = "confidence_weighted",
) -> tuple[np.ndarray, list[float], float]:
    if not predictions:
        raise ValueError("at least one selected evidence prediction is required")
    shape = predictions[0].target_probability.shape
    if any(item.target_probability.shape != shape for item in predictions):
        raise ValueError("selected evidence masks must share a spatial grid")
    weights = np.asarray(
        [np.clip(item.confidence, 0.05, 1.0) for item in predictions], dtype=np.float64
    )
    masks = np.stack([item.target_probability for item in predictions]).astype(np.float64)
    if strategy == "max_confidence":
        selected_index = int(np.argmax([item.confidence for item in predictions]))
        routing_weights = np.zeros(len(predictions), dtype=np.float64)
        routing_weights[selected_index] = 1.0
        return (
            masks[selected_index].astype(np.float32),
            routing_weights.tolist(),
            float(predictions[selected_index].confidence),
        )
    if strategy != "confidence_weighted":
        raise ValueError(f"unsupported evidence fusion strategy: {strategy}")
    fused_confidence = float(
        np.average(
            np.asarray([item.confidence for item in predictions], dtype=np.float64),
            weights=weights,
        )
    )
    return (
        np.average(masks, axis=0, weights=weights).astype(np.float32),
        weights.tolist(),
        fused_confidence,
    )


def apply_typed_mask_edit(
    prior: np.ndarray,
    fused_target: np.ndarray,
    operation: EditOperation,
) -> np.ndarray:
    """Constrain a full-scene target prediction by the terminal typed edit."""

    prior_probability = np.asarray(prior, dtype=np.float32)
    target_probability = np.asarray(fused_target, dtype=np.float32)
    if prior_probability.shape != target_probability.shape:
        raise ValueError("prior and target prediction must share a spatial grid")
    if operation == EditOperation.KEEP:
        return prior_probability.copy()
    if operation == EditOperation.ADD:
        return np.maximum(prior_probability, target_probability)
    if operation == EditOperation.DELETE:
        return np.minimum(prior_probability, target_probability)
    if operation == EditOperation.RESHAPE:
        return target_probability.copy()
    raise ValueError(f"unsupported operation: {operation}")


def apply_typed_binary_edit(
    prior: np.ndarray,
    fused_target: np.ndarray,
    operation: EditOperation,
    *,
    threshold: float,
    delta_margin: float,
) -> np.ndarray:
    """Apply only target evidence that clears a symmetric margin around threshold."""

    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between zero and one")
    if delta_margin < 0.0 or threshold - delta_margin < 0.0 or threshold + delta_margin > 1.0:
        raise ValueError("delta_margin must remain inside the probability threshold range")
    prior_mask = np.asarray(prior) >= threshold
    target_probability = np.asarray(fused_target, dtype=np.float32)
    if prior_mask.shape != target_probability.shape:
        raise ValueError("prior and target prediction must share a spatial grid")
    proposed_add = target_probability >= threshold + delta_margin
    proposed_remove = target_probability < threshold - delta_margin
    if operation == EditOperation.KEEP:
        return prior_mask.copy()
    if operation == EditOperation.ADD:
        return prior_mask | proposed_add
    if operation == EditOperation.DELETE:
        return prior_mask & ~proposed_remove
    if operation == EditOperation.RESHAPE:
        return (prior_mask | proposed_add) & ~proposed_remove
    raise ValueError(f"unsupported operation: {operation}")


def effective_operation_from_masks(
    prior: np.ndarray, committed: np.ndarray, valid: np.ndarray
) -> EditOperation:
    """Derive the operation that the executable map delta actually performs."""

    prior_mask = np.asarray(prior) >= 0.5
    committed_mask = np.asarray(committed) >= 0.5
    visible = np.asarray(valid) >= 0.5
    added = bool(np.any(committed_mask & ~prior_mask & visible))
    removed = bool(np.any(~committed_mask & prior_mask & visible))
    if added and removed:
        return EditOperation.RESHAPE
    if added:
        return EditOperation.ADD
    if removed:
        return EditOperation.DELETE
    return EditOperation.KEEP


def _rasterize_geometry(
    geometry: BaseGeometry | None, transform: Affine, shape: tuple[int, int]
) -> np.ndarray:
    if geometry is None or geometry.is_empty:
        return np.zeros(shape, dtype=np.float32)
    return rasterize(
        [(mapping(geometry), 1)],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    ).astype(np.float32)


def _component_count(mask: np.ndarray) -> int:
    _, count = ndimage.label(np.asarray(mask) >= 0.5, structure=np.ones((3, 3)))
    return int(count)


def _filter_small_components(
    mask: np.ndarray,
    min_pixels: int,
    *,
    preserve_largest: bool = False,
) -> np.ndarray:
    """Remove isolated delta components smaller than a validation-selected area."""

    binary = np.asarray(mask, dtype=bool)
    if min_pixels <= 1 or not binary.any():
        return binary.copy()
    labels, count = ndimage.label(binary, structure=np.ones((3, 3)))
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    keep = sizes >= min_pixels
    keep[0] = False
    filtered = keep[labels]
    if preserve_largest and binary.any() and not filtered.any():
        largest_label = int(np.argmax(sizes[1:]) + 1)
        filtered = labels == largest_label
    return filtered


def regularize_typed_delta(
    committed: np.ndarray,
    prior: np.ndarray,
    operation: EditOperation,
    *,
    min_component_pixels: int,
    preserve_largest: bool = False,
) -> tuple[np.ndarray, dict[str, int]]:
    """Filter tiny edit deltas and replay them while preserving typed constraints."""

    committed_mask = np.asarray(committed, dtype=bool)
    prior_mask = np.asarray(prior, dtype=bool)
    if committed_mask.shape != prior_mask.shape:
        raise ValueError("committed and prior masks must share a spatial grid")
    if min_component_pixels < 0:
        raise ValueError("min_component_pixels must be non-negative")
    predicted_add = committed_mask & ~prior_mask
    predicted_remove = ~committed_mask & prior_mask
    filtered_add = _filter_small_components(
        predicted_add,
        min_component_pixels,
        preserve_largest=preserve_largest,
    )
    filtered_remove = _filter_small_components(
        predicted_remove,
        min_component_pixels,
        preserve_largest=preserve_largest,
    )
    if operation == EditOperation.KEEP:
        regularized = prior_mask.copy()
    elif operation == EditOperation.ADD:
        regularized = prior_mask | filtered_add
    elif operation == EditOperation.DELETE:
        regularized = prior_mask & ~filtered_remove
    elif operation == EditOperation.RESHAPE:
        regularized = (prior_mask | filtered_add) & ~filtered_remove
    else:
        raise ValueError(f"unsupported operation: {operation}")
    return regularized, {
        "raw_add_component_count": _component_count(predicted_add),
        "retained_add_component_count": _component_count(filtered_add),
        "raw_remove_component_count": _component_count(predicted_remove),
        "retained_remove_component_count": _component_count(filtered_remove),
    }


def evaluate_typed_writeback(
    predictions: list[EvidenceMaskPrediction],
    *,
    operation: EditOperation,
    prior: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    transform: Affine,
    threshold: float = 0.5,
    simplify_tolerance: float = 0.0,
    min_delta_component_pixels: int = 0,
    preserve_largest_delta_component: bool = False,
    delta_margin: float = 0.0,
    confidence_floor: float = 0.0,
    evidence_fusion: str = "confidence_weighted",
    return_artifacts: bool = False,
) -> dict[str, Any]:
    """Fuse selected evidence, apply one typed edit, vectorize deltas, and replay them."""

    if not 0.0 <= confidence_floor <= 1.0:
        raise ValueError("confidence_floor must be between zero and one")
    fused, weights, fused_confidence = _fuse(predictions, strategy=evidence_fusion)
    prior_mask = (np.asarray(prior) >= threshold).astype(np.float32)
    target_mask = (np.asarray(target) >= threshold).astype(np.float32)
    visible = np.asarray(valid) >= 0.5
    confidence_gate_passed = fused_confidence >= confidence_floor
    committed = (
        apply_typed_binary_edit(
            prior,
            fused,
            operation,
            threshold=threshold,
            delta_margin=delta_margin,
        )
        if confidence_gate_passed
        else prior_mask.astype(bool)
    )
    committed, delta_components = regularize_typed_delta(
        committed,
        prior_mask,
        operation,
        min_component_pixels=min_delta_component_pixels,
        preserve_largest=preserve_largest_delta_component,
    )
    committed = committed.astype(np.float32)
    committed *= visible
    prior_mask *= visible
    target_mask *= visible
    effective_operation = effective_operation_from_masks(prior_mask, committed, visible)

    predicted_add = (committed >= 0.5) & (prior_mask < 0.5)
    predicted_remove = (committed < 0.5) & (prior_mask >= 0.5)
    target_add = (target_mask >= 0.5) & (prior_mask < 0.5)
    target_remove = (target_mask < 0.5) & (prior_mask >= 0.5)

    predicted_add_geometry = vectorize_mask(
        predicted_add.astype(np.float32), transform, simplify_tolerance=simplify_tolerance
    )
    predicted_remove_geometry = vectorize_mask(
        predicted_remove.astype(np.float32), transform, simplify_tolerance=simplify_tolerance
    )
    target_add_geometry = vectorize_mask(target_add.astype(np.float32), transform)
    target_remove_geometry = vectorize_mask(target_remove.astype(np.float32), transform)

    replay_add = _rasterize_geometry(predicted_add_geometry, transform, committed.shape)
    replay_remove = _rasterize_geometry(predicted_remove_geometry, transform, committed.shape)
    replay = np.maximum(prior_mask, replay_add)
    replay[replay_remove >= 0.5] = 0.0
    replay *= visible

    # Keep topology quality explicit so the executable GRPO reward can be
    # assembled directly from the writeback artifact without reconstructing
    # the map state later.
    prior_geometry = vectorize_mask(prior_mask, transform)
    committed_geometry = vectorize_mask(committed, transform)
    prior_topology_valid = (
        True if prior_geometry is None else topology_is_valid(prior_geometry)
    )
    committed_topology_valid = (
        True if committed_geometry is None else topology_is_valid(committed_geometry)
    )

    add_topology_valid = (
        True if predicted_add_geometry is None else topology_is_valid(predicted_add_geometry)
    )
    remove_topology_valid = (
        True if predicted_remove_geometry is None else topology_is_valid(predicted_remove_geometry)
    )
    result = {
        "operation": operation.value,
        "effective_operation": effective_operation.value,
        "writeback_changed": effective_operation != EditOperation.KEEP,
        "selected_evidence_ids": [item.evidence_id for item in predictions],
        "fusion_weights": weights,
        "evidence_fusion": evidence_fusion,
        "fused_confidence": fused_confidence,
        "confidence_floor": confidence_floor,
        "confidence_gate_passed": confidence_gate_passed,
        "delta_margin": delta_margin,
        "min_delta_component_pixels": min_delta_component_pixels,
        "preserve_largest_delta_component": preserve_largest_delta_component,
        **delta_components,
        "raster_iou": mask_iou(committed, target_mask, visible),
        "prior_raster_iou": mask_iou(prior_mask, target_mask, visible),
        "raster_iou_gain": mask_iou(committed, target_mask, visible)
        - mask_iou(prior_mask, target_mask, visible),
        "added_change_iou": mask_iou(predicted_add, target_add, visible),
        "removed_change_iou": mask_iou(predicted_remove, target_remove, visible),
        "added_polygon_iou": geometry_iou(predicted_add_geometry, target_add_geometry),
        "removed_polygon_iou": geometry_iou(predicted_remove_geometry, target_remove_geometry),
        "vector_replay_iou": mask_iou(replay, committed, visible),
        "topology_quality_before": float(prior_topology_valid),
        "topology_quality_after": float(committed_topology_valid),
        "vector_delta_topology_valid": bool(add_topology_valid and remove_topology_valid),
        "predicted_component_count": _component_count(committed),
        "prior_component_count": _component_count(prior_mask),
        "target_component_count": _component_count(target_mask),
        "component_count_absolute_error": abs(
            _component_count(committed) - _component_count(target_mask)
        ),
        "prior_component_count_absolute_error": abs(
            _component_count(prior_mask) - _component_count(target_mask)
        ),
        "predicted_add_geometry": (
            mapping(predicted_add_geometry) if predicted_add_geometry is not None else None
        ),
        "predicted_remove_geometry": (
            mapping(predicted_remove_geometry) if predicted_remove_geometry is not None else None
        ),
    }
    if return_artifacts:
        result["_committed_mask"] = committed
        result["_prior_mask"] = prior_mask
        result["_target_mask"] = target_mask
        result["_valid_mask"] = visible.astype(np.float32)
    return result
