#!/usr/bin/env python3
"""Audit prior-geometry and target-centering signal in SN7 updater crops."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.train_sn7_changemamba import Record, _read_records

EDIT_TYPES = ("KEEP", "ADD", "DELETE", "RESHAPE")
FEATURE_NAMES = (
    "area_fraction",
    "centroid_x",
    "centroid_y",
    "bbox_width",
    "bbox_height",
    "perimeter_fraction",
    "border_fraction",
    "moment_xx",
    "moment_yy",
    "moment_xy",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _mask(path: Path) -> np.ndarray:
    value = np.asarray(np.load(path), dtype=np.float32).squeeze()
    if value.ndim != 2:
        raise ValueError(f"expected 2D mask: {path}")
    return value >= 0.5


def mask_features(mask: np.ndarray, valid: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool) & np.asarray(valid, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    height, width = mask.shape
    valid_count = max(int(valid.sum()), 1)
    coordinates = np.argwhere(mask)
    if not len(coordinates):
        return np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    y = coordinates[:, 0].astype(np.float64) / max(height - 1, 1)
    x = coordinates[:, 1].astype(np.float64) / max(width - 1, 1)
    min_y, min_x = coordinates.min(axis=0)
    max_y, max_x = coordinates.max(axis=0)
    horizontal = np.logical_xor(mask[:, 1:], mask[:, :-1]).sum()
    vertical = np.logical_xor(mask[1:, :], mask[:-1, :]).sum()
    border = (
        mask[0].sum() + mask[-1].sum() + mask[:, 0].sum() + mask[:, -1].sum()
    )
    centered_x = x - x.mean()
    centered_y = y - y.mean()
    return np.asarray(
        (
            mask.sum() / valid_count,
            x.mean(),
            y.mean(),
            (max_x - min_x + 1) / width,
            (max_y - min_y + 1) / height,
            (horizontal + vertical) / valid_count,
            border / max(2 * height + 2 * width, 1),
            np.mean(centered_x**2),
            np.mean(centered_y**2),
            np.mean(centered_x * centered_y),
        ),
        dtype=np.float64,
    )


def _load(record: Record) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prior = _mask(record.prior)
    target = _mask(record.target)
    valid = _mask(record.valid) if record.valid is not None else np.ones_like(prior)
    return prior, target, valid


def _iou(left: np.ndarray, right: np.ndarray, valid: np.ndarray) -> float:
    intersection = (left & right & valid).sum()
    union = ((left | right) & valid).sum()
    return float(intersection / union) if union else 1.0


def _fit_geometry_classifier(
    features: np.ndarray, labels: list[str]
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (features - mean) / scale
    centroids = {
        edit: standardized[np.asarray(labels) == edit].mean(axis=0)
        for edit in EDIT_TYPES
    }
    return mean, scale, centroids


def _predict_geometry(
    feature: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    centroids: dict[str, np.ndarray],
) -> str:
    standardized = (feature - mean) / scale
    return min(
        EDIT_TYPES,
        key=lambda edit: float(np.sum((standardized - centroids[edit]) ** 2)),
    )


def _templates(
    records: list[Record],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    target_sums: dict[str, np.ndarray] = {}
    valid_sums: dict[str, np.ndarray] = {}
    tuning: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for record in records:
        prior, target, valid = _load(record)
        if record.edit_type not in target_sums:
            target_sums[record.edit_type] = np.zeros_like(target, dtype=np.float64)
            valid_sums[record.edit_type] = np.zeros_like(target, dtype=np.float64)
        target_sums[record.edit_type] += target & valid
        valid_sums[record.edit_type] += valid
        if len(tuning[record.edit_type]) < 256:
            tuning[record.edit_type].append((target, valid))
    probabilities = {
        edit: target_sums[edit] / np.maximum(valid_sums[edit], 1.0)
        for edit in EDIT_TYPES
    }
    thresholds: dict[str, float] = {}
    for edit in EDIT_TYPES:
        if edit == "KEEP":
            thresholds[edit] = 0.5
            continue
        scores = []
        for threshold in np.linspace(0.1, 0.9, 9):
            template = probabilities[edit] >= threshold
            values = []
            for target, valid in tuning[edit]:
                values.append(_iou(template, target, valid))
            scores.append((float(np.mean(values)), float(threshold)))
        thresholds[edit] = max(scores)[1]
    return probabilities, thresholds


def audit(manifest: Path) -> dict[str, Any]:
    train_records = _read_records(manifest, "train", None)
    val_records = _read_records(manifest, "val", None)
    train_features = []
    train_labels = []
    geometry: dict[str, list[np.ndarray]] = defaultdict(list)
    target_center_distance: dict[str, list[float]] = defaultdict(list)
    prior_empty: dict[str, list[bool]] = defaultdict(list)
    for record in train_records:
        prior, target, valid = _load(record)
        feature = mask_features(prior, valid)
        train_features.append(feature)
        train_labels.append(record.edit_type)
        geometry[record.edit_type].append(feature)
        target_feature = mask_features(target, valid)
        target_center_distance[record.edit_type].append(
            float(np.hypot(target_feature[1] - 0.5, target_feature[2] - 0.5))
            if target_feature[0] > 0
            else 0.0
        )
        prior_empty[record.edit_type].append(feature[0] == 0.0)
    feature_matrix = np.stack(train_features)
    mean, scale, centroids = _fit_geometry_classifier(feature_matrix, train_labels)
    templates, thresholds = _templates(train_records)

    operation_correct = {"empty_rule": 0, "nearest_centroid": 0}
    map_scores: dict[str, list[float]] = defaultdict(list)
    per_edit_delta: dict[str, dict[str, list[float]]] = {
        edit: defaultdict(list) for edit in EDIT_TYPES
    }
    for record in val_records:
        prior, target, valid = _load(record)
        feature = mask_features(prior, valid)
        empty_prediction = "ADD" if feature[0] == 0.0 else "KEEP"
        geometry_prediction = _predict_geometry(feature, mean, scale, centroids)
        operation_correct["empty_rule"] += int(empty_prediction == record.edit_type)
        operation_correct["nearest_centroid"] += int(
            geometry_prediction == record.edit_type
        )
        prior_iou = _iou(prior, target, valid)
        map_scores["prior"].append(prior_iou)
        for name, predicted_edit in (
            ("geometry_template", geometry_prediction),
            ("oracle_edit_template", record.edit_type),
        ):
            if predicted_edit == "KEEP":
                committed = prior
            elif predicted_edit == "DELETE":
                committed = np.zeros_like(prior)
            else:
                committed = templates[predicted_edit] >= thresholds[predicted_edit]
            committed_iou = _iou(committed, target, valid)
            map_scores[name].append(committed_iou)
            per_edit_delta[record.edit_type][name].append(
                committed_iou - prior_iou
            )
    prior_mean = float(np.mean(map_scores["prior"]))
    return {
        "schema_version": "sn7-candidate-localization-audit-v1",
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "train_count": len(train_records),
        "validation_count": len(val_records),
        "feature_names": FEATURE_NAMES,
        "train_geometry_by_edit": {
            edit: {
                "count": len(geometry[edit]),
                "prior_empty_rate": float(np.mean(prior_empty[edit])),
                "target_center_distance_mean": float(
                    np.mean(target_center_distance[edit])
                ),
                "target_center_distance_std": float(
                    np.std(target_center_distance[edit])
                ),
                "feature_mean": np.mean(geometry[edit], axis=0).tolist(),
                "feature_std": np.std(geometry[edit], axis=0).tolist(),
            }
            for edit in EDIT_TYPES
        },
        "operation_accuracy": {
            key: value / len(val_records) for key, value in operation_correct.items()
        },
        "template_thresholds_selected_on_train": thresholds,
        "validation_map_iou": {
            key: float(np.mean(values)) for key, values in map_scores.items()
        },
        "validation_map_iou_delta": {
            key: float(np.mean(values) - prior_mean)
            for key, values in map_scores.items()
            if key != "prior"
        },
        "validation_delta_by_edit": {
            edit: {
                key: float(np.mean(values))
                for key, values in per_edit_delta[edit].items()
            }
            for edit in EDIT_TYPES
        },
        "interpretation_contract": (
            "diagnoses candidate-crop conditioning; does not estimate "
            "scene-level open-world discovery"
        ),
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
