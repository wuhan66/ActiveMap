#!/usr/bin/env python3
"""Build a statistically correct updater bundle from per-seed predictions."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from activemap.evaluation.update import EDIT_LABELS, load_update_predictions
from activemap.models import EditOperation
from scripts.build_paper_result_bundle import _select_cell, _sha256

SUPPORTED_METRICS = {
    "macro_f1",
    "update_f1",
    "false_edit_rate",
    "missed_edit_rate",
    "raster_iou",
    "polygon_iou",
    "calibration_ece",
}


@dataclass
class GroupStatistics:
    confusion: np.ndarray
    update_counts: np.ndarray
    raster: np.ndarray
    polygon: np.ndarray
    calibration_count: np.ndarray
    calibration_confidence: np.ndarray
    calibration_correct: np.ndarray
    sample_count: int

    @classmethod
    def zeros(cls, calibration_bins: int) -> GroupStatistics:
        return cls(
            confusion=np.zeros((4, 4), dtype=np.int64),
            update_counts=np.zeros(5, dtype=np.int64),
            raster=np.zeros(2, dtype=np.float64),
            polygon=np.zeros(2, dtype=np.float64),
            calibration_count=np.zeros(calibration_bins, dtype=np.int64),
            calibration_confidence=np.zeros(calibration_bins, dtype=np.float64),
            calibration_correct=np.zeros(calibration_bins, dtype=np.float64),
            sample_count=0,
        )

    def add(self, other: GroupStatistics) -> None:
        self.confusion += other.confusion
        self.update_counts += other.update_counts
        self.raster += other.raster
        self.polygon += other.polygon
        self.calibration_count += other.calibration_count
        self.calibration_confidence += other.calibration_confidence
        self.calibration_correct += other.calibration_correct
        self.sample_count += other.sample_count


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _group_statistics(records: list[Any], *, calibration_bins: int) -> GroupStatistics:
    statistics = GroupStatistics.zeros(calibration_bins)
    label_to_index = {label: index for index, label in enumerate(EDIT_LABELS)}
    for record in records:
        predicted = record.predicted_edit if record.committed else EditOperation.KEEP
        target_index = label_to_index[record.target_edit]
        predicted_index = label_to_index[predicted]
        statistics.confusion[target_index, predicted_index] += 1
        true_update = record.target_edit != EditOperation.KEEP
        predicted_update = record.committed and predicted != EditOperation.KEEP
        statistics.update_counts[0] += int(true_update and predicted_update)
        statistics.update_counts[1] += int(not true_update and predicted_update)
        statistics.update_counts[2] += int(true_update and not predicted_update)
        statistics.update_counts[3] += int(not true_update)
        statistics.update_counts[4] += int(true_update)
        if record.raster_iou is not None:
            statistics.raster += (float(record.raster_iou), 1.0)
        if record.polygon_iou is not None:
            statistics.polygon += (float(record.polygon_iou), 1.0)
        confidence = float(np.clip(record.confidence, 1e-7, 1.0 - 1e-7))
        bin_index = min(int(confidence * calibration_bins), calibration_bins - 1)
        statistics.calibration_count[bin_index] += 1
        statistics.calibration_confidence[bin_index] += confidence
        statistics.calibration_correct[bin_index] += float(record.target_edit == predicted)
        statistics.sample_count += 1
    return statistics


def _metrics(statistics: GroupStatistics) -> dict[str, float]:
    class_f1 = []
    for index in range(4):
        true_positive = int(statistics.confusion[index, index])
        false_positive = int(statistics.confusion[:, index].sum() - true_positive)
        false_negative = int(statistics.confusion[index, :].sum() - true_positive)
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        class_f1.append(_safe_divide(2.0 * precision * recall, precision + recall))
    update_tp, update_fp, update_fn, stable_count, changed_count = map(
        int, statistics.update_counts
    )
    update_precision = _safe_divide(update_tp, update_tp + update_fp)
    update_recall = _safe_divide(update_tp, update_tp + update_fn)
    ece = 0.0
    total = int(statistics.calibration_count.sum())
    for index, count in enumerate(statistics.calibration_count):
        if count:
            accuracy = statistics.calibration_correct[index] / count
            confidence = statistics.calibration_confidence[index] / count
            ece += float(count / total) * abs(float(accuracy - confidence))
    return {
        "macro_f1": float(np.mean(class_f1)),
        "update_f1": _safe_divide(
            2.0 * update_precision * update_recall, update_precision + update_recall
        ),
        "false_edit_rate": _safe_divide(update_fp, stable_count),
        "missed_edit_rate": _safe_divide(update_fn, changed_count),
        "raster_iou": _safe_divide(statistics.raster[0], statistics.raster[1]),
        "polygon_iou": _safe_divide(statistics.polygon[0], statistics.polygon[1]),
        "calibration_ece": ece,
    }


def _parse_seed_paths(specifications: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for specification in specifications:
        seed, separator, raw_path = specification.partition("=")
        if not separator or not seed or not raw_path:
            raise ValueError("seed predictions must use SEED=PATH")
        if seed in result:
            raise ValueError(f"duplicate seed prediction {seed}")
        result[seed] = Path(raw_path)
    return result


def build_updater_bundle(
    registry_path: Path,
    seed_prediction_paths: dict[str, Path],
    *,
    experiment_id: str,
    variant: str | None,
    calibration_bins: int = 15,
    frozen_test_ledger: Path | None = None,
) -> dict[str, Any]:
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    cell = _select_cell(registry, experiment_id, variant, None)
    if cell["family"] != "updater":
        raise ValueError("updater bundle builder requires an updater registry cell")
    required_metrics = set(cell["required_metrics"]) | set(cell["primary_metrics"])
    unsupported = sorted(required_metrics - SUPPORTED_METRICS)
    if unsupported:
        raise ValueError(f"unsupported updater metrics: {unsupported}")
    expected_seeds = set(map(str, cell["seeds"]))
    if set(seed_prediction_paths) != expected_seeds:
        raise ValueError("prediction seeds do not match the registry cell")
    if calibration_bins <= 0:
        raise ValueError("calibration_bins must be positive")

    grouped: dict[str, dict[str, GroupStatistics]] = {}
    rows_by_seed: dict[str, int] = {}
    for seed, path in sorted(seed_prediction_paths.items()):
        records = load_update_predictions(path)
        by_aoi: dict[str, list[Any]] = {}
        for record in records:
            if record.metadata.get("split") != cell["split"]:
                raise ValueError(f"{path}: prediction split does not match registry cell")
            by_aoi.setdefault(record.aoi_id, []).append(record)
        grouped[seed] = {
            aoi_id: _group_statistics(aoi_records, calibration_bins=calibration_bins)
            for aoi_id, aoi_records in by_aoi.items()
        }
        rows_by_seed[seed] = len(records)
    unit_sets = {seed: set(values) for seed, values in grouped.items()}
    seed_order = sorted(expected_seeds)
    unit_order = sorted(unit_sets[seed_order[0]])
    if not unit_order or any(unit_sets[seed] != set(unit_order) for seed in seed_order):
        raise ValueError("AOI ids must be non-empty and identical across seeds")

    def aggregate(seed: str, selected_units: list[str]) -> dict[str, float]:
        statistics = GroupStatistics.zeros(calibration_bins)
        for unit in selected_units:
            statistics.add(grouped[seed][unit])
        return _metrics(statistics)

    seed_metrics = {seed: aggregate(seed, unit_order) for seed in seed_order}
    replicates = int(registry["protocol"]["bootstrap_replicates"])
    confidence_level = float(registry["protocol"]["confidence_level"])
    bootstrap_seed = int(registry["protocol"].get("bootstrap_seed", 20260715))
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_indices = rng.integers(
        0, len(unit_order), size=(replicates, len(unit_order))
    )
    bootstrap_values = {metric: [] for metric in required_metrics}
    for indices in bootstrap_indices:
        selected_units = [unit_order[index] for index in indices]
        replicate_by_seed = {
            seed: aggregate(seed, selected_units) for seed in seed_order
        }
        for metric in required_metrics:
            bootstrap_values[metric].append(
                float(np.mean([replicate_by_seed[seed][metric] for seed in seed_order]))
            )
    alpha = (1.0 - confidence_level) / 2.0
    summaries: dict[str, dict[str, Any]] = {}
    for metric in sorted(required_metrics):
        seed_values = np.asarray(
            [seed_metrics[seed][metric] for seed in seed_order], dtype=np.float64
        )
        if not np.all(np.isfinite(seed_values)):
            raise ValueError(f"non-finite updater metric {metric}")
        standard_deviation = (
            float(np.std(seed_values, ddof=1)) if len(seed_values) > 1 else 0.0
        )
        bootstrap = np.asarray(bootstrap_values[metric], dtype=np.float64)
        summaries[metric] = {
            "mean": float(seed_values.mean()),
            "std": standard_deviation,
            "ci95": [
                float(np.quantile(bootstrap, alpha)),
                float(np.quantile(bootstrap, 1.0 - alpha)),
            ],
            "seed_means": {
                seed: float(seed_metrics[seed][metric]) for seed in seed_order
            },
        }

    bundle: dict[str, Any] = {
        "schema_version": "activemap-paper-result-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "variant": variant,
        "budget": None,
        "family": "updater",
        "split": cell["split"],
        "seeds": seed_order,
        "sample_count": sum(rows_by_seed.values()),
        "unit_count": len(unit_order),
        "rows_by_seed": rows_by_seed,
        "bootstrap_unit": registry["protocol"]["bootstrap_unit"],
        "bootstrap_replicates": replicates,
        "bootstrap_seed": bootstrap_seed,
        "confidence_level": confidence_level,
        "registry_sha256": _sha256(registry_path),
        "source_observations": [
            {"seed": seed, "path": str(path.resolve()), "sha256": _sha256(path)}
            for seed, path in sorted(seed_prediction_paths.items())
        ],
        "aggregation": {
            "type": "updater_sufficient_statistics_v1",
            "calibration_bins": calibration_bins,
        },
        "metrics": summaries,
    }
    if cell["split"] == "test":
        if frozen_test_ledger is None:
            raise ValueError("test bundles require a frozen test ledger")
        ledger = json.loads(frozen_test_ledger.read_text(encoding="utf-8"))
        if ledger.get("status") != "complete" or ledger.get("returncode") != 0:
            raise ValueError("frozen test ledger is not complete")
        bundle["frozen_test_ledger"] = str(frozen_test_ledger.resolve())
        bundle["frozen_test_ledger_sha256"] = _sha256(frozen_test_ledger)
    elif frozen_test_ledger is not None:
        raise ValueError("validation bundles must not reference a frozen test ledger")
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("registry", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--variant")
    parser.add_argument("--seed-predictions", action="append", required=True)
    parser.add_argument("--calibration-bins", type=int, default=15)
    parser.add_argument("--frozen-test-ledger", type=Path)
    args = parser.parse_args()
    bundle = build_updater_bundle(
        args.registry,
        _parse_seed_paths(args.seed_predictions),
        experiment_id=args.experiment_id,
        variant=args.variant,
        calibration_bins=args.calibration_bins,
        frozen_test_ledger=args.frozen_test_ledger,
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "sample_count": bundle["sample_count"]}))


if __name__ == "__main__":
    main()
