from pathlib import Path

import pytest

from activemap.evaluation.statistics import (
    grouped_bootstrap_intervals,
    paired_group_bootstrap_difference,
)
from activemap.evaluation.update import (
    UpdatePrediction,
    evaluate_updates,
    load_update_predictions,
    object_scale_strata_metrics,
    segmentation_strata_metrics,
    select_commit_threshold,
    write_update_predictions,
)
from activemap.models import EditOperation


def _record(
    sample_id: str,
    aoi_id: str,
    target: EditOperation,
    predicted: EditOperation,
    confidence: float,
) -> UpdatePrediction:
    return UpdatePrediction(
        sample_id=sample_id,
        aoi_id=aoi_id,
        target_edit=target,
        predicted_edit=predicted,
        confidence=confidence,
        raster_iou=0.75,
    )


def test_update_metrics_preserve_stable_map_and_detect_changes(tmp_path: Path) -> None:
    records = [
        _record("a", "aoi-1", EditOperation.KEEP, EditOperation.KEEP, 0.9),
        _record("b", "aoi-1", EditOperation.KEEP, EditOperation.ADD, 0.6),
        _record("c", "aoi-2", EditOperation.ADD, EditOperation.ADD, 0.8),
        _record("d", "aoi-2", EditOperation.DELETE, EditOperation.KEEP, 0.4),
    ]
    metrics = evaluate_updates(records, calibration_bins=4)
    assert metrics["sample_count"] == 4
    assert metrics["aoi_count"] == 2
    assert metrics["edit_accuracy"] == pytest.approx(0.5)
    assert metrics["false_edit_rate"] == pytest.approx(0.5)
    assert metrics["missed_update_rate"] == pytest.approx(0.5)
    assert metrics["mean_raster_iou"] == pytest.approx(0.75)
    assert len(metrics["risk_coverage"]) == 4
    calibration = select_commit_threshold(records, thresholds=[0.0, 0.5, 1.0])
    assert calibration["best_threshold"] in {0.0, 0.5, 1.0}

    path = tmp_path / "predictions.jsonl"
    assert write_update_predictions(records, path) == 4
    assert load_update_predictions(path) == records


def test_grouped_and_paired_bootstrap_are_deterministic() -> None:
    baseline = [
        _record("a", "aoi-1", EditOperation.KEEP, EditOperation.ADD, 0.6),
        _record("b", "aoi-1", EditOperation.ADD, EditOperation.KEEP, 0.4),
        _record("c", "aoi-2", EditOperation.ADD, EditOperation.ADD, 0.8),
        _record("d", "aoi-2", EditOperation.KEEP, EditOperation.KEEP, 0.9),
    ]
    challenger = [
        record.model_copy(update={"predicted_edit": record.target_edit}) for record in baseline
    ]
    first = grouped_bootstrap_intervals(baseline, iterations=20, seed=7)
    second = grouped_bootstrap_intervals(baseline, iterations=20, seed=7)
    assert first == second
    comparison = paired_group_bootstrap_difference(
        baseline,
        challenger,
        metric="edit_accuracy",
        iterations=20,
        seed=7,
    )
    assert comparison["difference"] > 0
    assert comparison["lower"] >= 0


def test_segmentation_strata_exclude_empty_scenes_from_foreground_iou() -> None:
    records = [
        _record("positive", "aoi-1", EditOperation.KEEP, EditOperation.KEEP, 0.9).model_copy(
            update={"raster_iou": 0.6, "metadata": {"target_foreground": True}}
        ),
        _record("empty", "aoi-2", EditOperation.KEEP, EditOperation.KEEP, 0.9).model_copy(
            update={
                "raster_iou": 1.0,
                "metadata": {
                    "target_foreground": False,
                    "empty_scene_false_positive_fraction": 0.02,
                },
            }
        ),
    ]

    metrics = segmentation_strata_metrics(records)

    assert metrics["foreground_sample_count"] == 1
    assert metrics["empty_sample_count"] == 1
    assert metrics["mean_foreground_iou"] == pytest.approx(0.6)
    assert metrics["mean_empty_scene_false_positive_fraction"] == pytest.approx(0.02)


def test_object_scale_strata_use_resolution_independent_area_fraction() -> None:
    fractions = (0.0005, 0.003, 0.01, 0.05)
    records = []
    for index, fraction in enumerate(fractions):
        records.append(
            _record(
                f"sample-{index}",
                f"aoi-{index}",
                EditOperation.ADD,
                EditOperation.ADD if index != 1 else EditOperation.KEEP,
                0.9,
            ).model_copy(
                update={
                    "raster_iou": 0.5 + index * 0.1,
                    "polygon_iou": 0.4 + index * 0.1,
                    "metadata": {"target_foreground_fraction": fraction},
                }
            )
        )

    metrics = object_scale_strata_metrics(records)

    assert metrics["available_sample_count"] == 4
    assert metrics["missing_fraction_sample_count"] == 0
    assert metrics["strata"]["tiny"]["sample_count"] == 1
    assert metrics["strata"]["small"]["typed_edit_accuracy"] == 0.0
    assert metrics["strata"]["medium"]["mean_raster_iou"] == pytest.approx(0.7)
    assert metrics["strata"]["large"]["mean_polygon_iou"] == pytest.approx(0.7)
