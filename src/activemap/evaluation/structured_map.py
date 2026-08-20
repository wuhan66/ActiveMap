"""Evaluation protocol for object-level editable structured maps."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

from activemap.data.structured_map import (
    StructuredMapSample,
    _load_feature_map,
    derive_structured_atomic_edits,
)
from activemap.integrations.baselines.contracts import StructuredMapPrediction
from activemap.models import EditOperation, GeoJSONGeometry

CHANGED_OPERATIONS = (EditOperation.ADD, EditOperation.DELETE, EditOperation.RESHAPE)
TOPOLOGY_FIELDS = ("predecessors", "successors", "left_neighbor_id", "right_neighbor_id")


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": _safe_div(2.0 * precision * recall, precision + recall),
    }


def _geometry_paths(geometry: GeoJSONGeometry) -> list[list[list[float]]]:
    coordinates = geometry.coordinates
    if geometry.type == "LineString":
        return [coordinates]
    if geometry.type == "MultiLineString":
        return coordinates
    if geometry.type == "Polygon":
        return coordinates
    if geometry.type == "MultiPolygon":
        return [ring for polygon in coordinates for ring in polygon]
    raise ValueError(f"unsupported geometry type: {geometry.type}")


def _sample_geometry(geometry: GeoJSONGeometry, samples_per_segment: int = 8) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for path in _geometry_paths(geometry):
        if len(path) == 1:
            points.append((float(path[0][0]), float(path[0][1])))
            continue
        for start, end in zip(path, path[1:]):
            x0, y0 = float(start[0]), float(start[1])
            x1, y1 = float(end[0]), float(end[1])
            for index in range(samples_per_segment):
                alpha = index / samples_per_segment
                points.append((x0 + alpha * (x1 - x0), y0 + alpha * (y1 - y0)))
        points.append((float(path[-1][0]), float(path[-1][1])))
    return points


def symmetric_chamfer(first: GeoJSONGeometry, second: GeoJSONGeometry) -> float:
    """Symmetric mean nearest-point distance after deterministic segment sampling."""

    first_points = _sample_geometry(first)
    second_points = _sample_geometry(second)
    if not first_points or not second_points:
        return math.inf

    def directed(source: Iterable[tuple[float, float]], target: list[tuple[float, float]]) -> float:
        distances = [
            min(math.hypot(x - tx, y - ty) for tx, ty in target)
            for x, y in source
        ]
        return sum(distances) / len(distances)

    return 0.5 * (directed(first_points, second_points) + directed(second_points, first_points))


def _relation_values(value: Any) -> list[str]:
    if value in {None, ""}:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(value)]


def _topology_edges(objects: dict[str, tuple[GeoJSONGeometry, dict[str, Any]]]) -> set[tuple[str, str, str]]:
    existing = set(objects)
    edges: set[tuple[str, str, str]] = set()
    for object_id, (_, attributes) in objects.items():
        for relation in TOPOLOGY_FIELDS:
            for target_id in _relation_values(attributes.get(relation)):
                if target_id in existing:
                    edges.add((object_id, relation, target_id))
    return edges


def _predicted_map(
    prior: dict[str, tuple[GeoJSONGeometry, dict[str, Any]]],
    predictions: dict[str, StructuredMapPrediction],
) -> dict[str, tuple[GeoJSONGeometry, dict[str, Any]]]:
    result = {key: (geometry, dict(attributes)) for key, (geometry, attributes) in prior.items()}
    for object_id, prediction in predictions.items():
        if prediction.operation == EditOperation.DELETE:
            result.pop(object_id, None)
        elif prediction.operation in {EditOperation.ADD, EditOperation.RESHAPE}:
            attributes = dict(result.get(object_id, (prediction.geometry, {}))[1])
            attributes.update(prediction.attributes_after)
            assert prediction.geometry is not None
            result[object_id] = (prediction.geometry, attributes)
    return result


def _load_samples(path: Path, *, allow_test: bool) -> list[StructuredMapSample]:
    samples = [
        StructuredMapSample.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not samples:
        raise ValueError(f"no structured-map samples in {path}")
    if any(sample.split == "test" for sample in samples) and not allow_test:
        raise PermissionError("test evaluation requires allow_test=True after protocol freeze")
    return samples


def _load_predictions(path: Path) -> list[StructuredMapPrediction]:
    predictions = [
        StructuredMapPrediction.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not predictions:
        raise ValueError(f"no structured-map predictions in {path}")
    keys = [(row.sample_id, row.object_id) for row in predictions]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate sample/object predictions")
    return predictions


def evaluate_structured_map_predictions(
    samples_path: Path,
    predictions_path: Path,
    *,
    allow_test: bool = False,
) -> dict[str, Any]:
    """Evaluate exact edit semantics, safety, geometry, and map topology."""

    samples = _load_samples(samples_path, allow_test=allow_test)
    predictions = _load_predictions(predictions_path)
    sample_by_id = {sample.sample_id: sample for sample in samples}
    if len(sample_by_id) != len(samples):
        raise ValueError("duplicate sample IDs")
    unknown_samples = sorted({row.sample_id for row in predictions} - set(sample_by_id))
    if unknown_samples:
        raise ValueError(f"predictions reference unknown samples: {unknown_samples[:3]}")

    grouped: dict[str, dict[str, StructuredMapPrediction]] = {sample.sample_id: {} for sample in samples}
    for row in predictions:
        sample = sample_by_id[row.sample_id]
        if row.split != sample.split or row.dataset != sample.dataset:
            raise ValueError(f"prediction provenance mismatch for {row.sample_id}")
        grouped[row.sample_id][row.object_id] = row

    total_tp = total_fp = total_fn = 0
    operation_counts = {operation.value: [0, 0, 0] for operation in CHANGED_OPERATIONS}
    evaluated_objects = unchanged_objects = preserved_objects = predicted_changes = false_changes = 0
    chamfer_values: list[float] = []
    topology_tp = topology_fp = topology_fn = 0

    for sample in samples:
        prior_path = Path(sample.prior_map_path)
        target_path = Path(sample.target_map_path)
        ground_truth, _ = derive_structured_atomic_edits(prior_path, target_path, include_keep=True)
        truth_by_object = {row.object_id: row for row in ground_truth}
        prediction_by_object = grouped[sample.sample_id]
        object_ids = sorted(set(truth_by_object) | set(prediction_by_object))
        evaluated_objects += len(object_ids)

        for object_id in object_ids:
            truth = truth_by_object.get(object_id)
            truth_operation = truth.operation if truth is not None else EditOperation.KEEP
            prediction = prediction_by_object.get(object_id)
            predicted_operation = prediction.operation if prediction is not None else EditOperation.KEEP
            exact_changed = truth_operation in CHANGED_OPERATIONS and predicted_operation == truth_operation
            if exact_changed:
                total_tp += 1
            if predicted_operation in CHANGED_OPERATIONS:
                predicted_changes += 1
                if not exact_changed:
                    total_fp += 1
                    false_changes += 1
            if truth_operation in CHANGED_OPERATIONS and not exact_changed:
                total_fn += 1
            if truth_operation == EditOperation.KEEP:
                unchanged_objects += 1
                if predicted_operation == EditOperation.KEEP:
                    preserved_objects += 1

            for operation in CHANGED_OPERATIONS:
                tp, fp, fn = operation_counts[operation.value]
                if truth_operation == operation and predicted_operation == operation:
                    tp += 1
                elif predicted_operation == operation:
                    fp += 1
                elif truth_operation == operation:
                    fn += 1
                operation_counts[operation.value] = [tp, fp, fn]

            if (
                exact_changed
                and truth_operation in {EditOperation.ADD, EditOperation.RESHAPE}
                and prediction is not None
                and prediction.geometry is not None
                and truth is not None
                and truth.target_geometry is not None
            ):
                chamfer_values.append(symmetric_chamfer(prediction.geometry, truth.target_geometry))

        prior = _load_feature_map(prior_path)
        target = _load_feature_map(target_path)
        predicted = _predicted_map(prior, prediction_by_object)
        predicted_edges = _topology_edges(predicted)
        target_edges = _topology_edges(target)
        topology_tp += len(predicted_edges & target_edges)
        topology_fp += len(predicted_edges - target_edges)
        topology_fn += len(target_edges - predicted_edges)

    edit_metrics = _f1(total_tp, total_fp, total_fn)
    topology_metrics = _f1(topology_tp, topology_fp, topology_fn)
    return {
        "schema_version": "activemap-structured-map-evaluation-v1",
        "protocol": {
            "omitted_prediction": "KEEP",
            "atomic_match": "exact object_id and operation over ADD/DELETE/RESHAPE",
            "false_edit_rate_denominator": "all evaluated objects",
            "geometry": "symmetric Chamfer on exact ADD/RESHAPE matches",
            "topology": list(TOPOLOGY_FIELDS),
        },
        "sample_count": len(samples),
        "evaluated_object_count": evaluated_objects,
        "atomic_edit": edit_metrics,
        "per_operation": {
            operation: _f1(*counts) for operation, counts in operation_counts.items()
        },
        "unchanged_preservation": _safe_div(preserved_objects, unchanged_objects),
        "unchanged_object_count": unchanged_objects,
        "false_edit_rate": _safe_div(false_changes, evaluated_objects),
        "false_edit_discovery_rate": _safe_div(false_changes, predicted_changes),
        "predicted_change_count": predicted_changes,
        "geometry_chamfer": (
            sum(chamfer_values) / len(chamfer_values) if chamfer_values else None
        ),
        "geometry_match_count": len(chamfer_values),
        "topology": topology_metrics,
        "test_assets_read": any(sample.split == "test" for sample in samples),
    }
