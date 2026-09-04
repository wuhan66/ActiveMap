import numpy as np

from activemap.evaluation.update import UpdatePrediction
from scripts.evaluate_sn7_changemamba import (
    aggregate_rows,
    paper_prediction,
    sample_metrics,
)


def test_sample_metrics_identifies_correct_add():
    prior = np.zeros((2, 2), dtype=bool)
    target = np.array([[1, 0], [0, 0]], dtype=bool)
    prediction = target.copy()
    valid = np.ones((2, 2), dtype=bool)

    metrics = sample_metrics(
        prior,
        target,
        prediction,
        valid,
        target_edit="ADD",
        predicted_edit="ADD",
    )

    assert metrics["committed_map_iou"] == 1.0
    assert metrics["change_iou"] == 1.0
    assert metrics["operation_correct"] is True
    assert metrics["false_edit"] is False
    assert metrics["missed_edit"] is False


def test_sample_metrics_identifies_false_edit():
    prior = np.zeros((2, 2), dtype=bool)
    target = prior.copy()
    prediction = np.array([[1, 0], [0, 0]], dtype=bool)
    valid = np.ones((2, 2), dtype=bool)

    metrics = sample_metrics(
        prior,
        target,
        prediction,
        valid,
        target_edit="KEEP",
        predicted_edit="ADD",
    )

    assert metrics["false_edit"] is True
    assert metrics["committed_map_iou"] == 0.0


def test_aggregate_rows_uses_conditional_safety_denominators():
    base = {
        "prior_map_iou": 0.5,
        "committed_map_iou": 0.6,
        "map_iou_delta": 0.1,
        "change_iou": 0.7,
        "operation_correct": True,
        "false_edit": False,
        "missed_edit": False,
        "wrong_edit": False,
    }
    rows = [
        {**base, "target_edit": "KEEP", "false_edit": True},
        {**base, "target_edit": "KEEP"},
        {**base, "target_edit": "ADD", "missed_edit": True},
        {**base, "target_edit": "DELETE"},
    ]

    aggregate = aggregate_rows(rows)

    assert aggregate["false_edit_rate"] == 0.5
    assert aggregate["missed_edit_rate"] == 0.5


def test_paper_prediction_matches_update_prediction_contract():
    row = {
        "sample_id": "sample",
        "aoi_id": "aoi",
        "split": "val",
        "target_edit": "ADD",
        "predicted_edit": "ADD",
        "confidence": 0.8,
        "prior_map_iou": 0.5,
        "committed_map_iou": 0.75,
        "map_iou_delta": 0.25,
        "change_iou": 0.7,
    }

    prediction = UpdatePrediction.model_validate(paper_prediction(row))

    assert prediction.raster_iou == 0.75
    assert prediction.metadata["confidence_source"] == (
        "mean_valid_pixel_max_softmax"
    )
