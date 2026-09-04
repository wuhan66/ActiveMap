import json
from pathlib import Path

import pytest
import yaml

from activemap.evaluation.update import (
    UpdatePrediction,
    evaluate_updates,
    write_update_predictions,
)
from activemap.models import EditOperation
from scripts.audit_paper_result_bundles import audit_result_bundles
from scripts.build_paper_updater_bundle import build_updater_bundle


def _registry(path: Path) -> Path:
    metrics = [
        "macro_f1",
        "update_f1",
        "false_edit_rate",
        "missed_edit_rate",
        "raster_iou",
        "polygon_iou",
        "calibration_ece",
    ]
    payload = {
        "protocol": {
            "muno21_budgets": [3.0],
            "bootstrap_unit": "task_or_aoi",
            "bootstrap_replicates": 100,
            "bootstrap_seed": 5,
            "confidence_level": 0.95,
        },
        "required_metrics": {
            "updater": metrics,
            "selector": ["score"],
            "agent": ["score"],
            "writeback": ["score"],
        },
        "primary_metrics": {
            "updater": ["macro_f1", "false_edit_rate", "polygon_iou"],
            "selector": ["score"],
            "agent": ["score"],
            "rl": ["score"],
            "writeback": ["score"],
        },
        "experiments": [
            {
                "id": "updater",
                "family": "updater",
                "test_policy": "validation_only",
                "seeds": [7, 8],
            }
        ],
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _predictions(path: Path, *, seed: int, make_error: bool) -> list[UpdatePrediction]:
    records = []
    for aoi_id in ("a", "b"):
        for index, operation in enumerate(EditOperation):
            predicted = operation
            if make_error and aoi_id == "a" and operation == EditOperation.KEEP:
                predicted = EditOperation.ADD
            records.append(
                UpdatePrediction(
                    sample_id=f"{seed}-{aoi_id}-{index}",
                    aoi_id=aoi_id,
                    target_edit=operation,
                    predicted_edit=predicted,
                    confidence=0.9 if predicted == operation else 0.6,
                    committed=True,
                    raster_iou=0.8 + 0.01 * index,
                    polygon_iou=0.7 + 0.01 * index,
                    metadata={"split": "val"},
                )
            )
    write_update_predictions(records, path)
    return records


def test_updater_bundle_matches_full_seed_metrics_and_audits(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "registry.yaml")
    seed7 = _predictions(tmp_path / "seed7.jsonl", seed=7, make_error=False)
    seed8 = _predictions(tmp_path / "seed8.jsonl", seed=8, make_error=True)
    bundle = build_updater_bundle(
        registry,
        {"7": tmp_path / "seed7.jsonl", "8": tmp_path / "seed8.jsonl"},
        experiment_id="updater",
        variant=None,
    )

    expected7 = evaluate_updates(seed7)
    expected8 = evaluate_updates(seed8)
    assert bundle["sample_count"] == 16
    assert bundle["unit_count"] == 2
    assert bundle["metrics"]["macro_f1"]["seed_means"] == pytest.approx(
        {"7": expected7["macro_f1"], "8": expected8["macro_f1"]}
    )
    assert bundle["metrics"]["update_f1"]["seed_means"] == pytest.approx(
        {"7": expected7["update_f1"], "8": expected8["update_f1"]}
    )
    assert bundle["metrics"]["calibration_ece"]["seed_means"] == pytest.approx(
        {"7": expected7["ece"], "8": expected8["ece"]}
    )
    assert bundle["aggregation"]["type"] == "updater_sufficient_statistics_v1"

    bundles = tmp_path / "bundles"
    bundles.mkdir()
    (bundles / "updater.json").write_text(json.dumps(bundle), encoding="utf-8")
    report = audit_result_bundles(registry, bundles)
    assert report["paper_tables_complete"] is True


def test_updater_bundle_rejects_mismatched_aoi_sets(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "registry.yaml")
    _predictions(tmp_path / "seed7.jsonl", seed=7, make_error=False)
    records = _predictions(tmp_path / "seed8.jsonl", seed=8, make_error=False)
    write_update_predictions(
        [record for record in records if record.aoi_id == "a"], tmp_path / "seed8.jsonl"
    )
    with pytest.raises(ValueError, match="AOI ids"):
        build_updater_bundle(
            registry,
            {"7": tmp_path / "seed7.jsonl", "8": tmp_path / "seed8.jsonl"},
            experiment_id="updater",
            variant=None,
        )
