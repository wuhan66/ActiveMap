from activemap.evaluation.update import UpdatePrediction
from scripts.calibrate_sn7_changemamba_safe_commit import (
    calibrate,
    deserialize_logistic,
    paper_prediction,
    predict_logistic,
    policy_metrics,
)
from scripts.apply_sn7_changemamba_safe_commit import apply_frozen_calibration


def _row(
    index: int,
    *,
    split: str,
    aoi_id: str,
    target_edit: str,
    predicted_edit: str,
    confidence: float,
    delta: float,
):
    return {
        "sample_id": f"{split}-{index}",
        "aoi_id": aoi_id,
        "split": split,
        "target_edit": target_edit,
        "predicted_edit": predicted_edit,
        "confidence": confidence,
        "mean_change_probability": confidence * 0.5,
        "max_change_probability": confidence,
        "p95_change_probability": confidence * 0.9,
        "mean_predictive_entropy": 1.0 - confidence,
        "prior_foreground_fraction": 0.25,
        "predicted_change_fraction": 0.1 if predicted_edit != "KEEP" else 0.0,
        "prior_map_iou": 0.5,
        "committed_map_iou": 0.5 + delta,
        "map_iou_delta": delta,
        "change_iou": 0.5,
        "operation_correct": predicted_edit == target_edit,
        "false_edit": target_edit == "KEEP" and predicted_edit != "KEEP",
        "missed_edit": target_edit != "KEEP" and predicted_edit == "KEEP",
        "wrong_edit": False,
    }


def _rows(split: str, aoi_count: int):
    rows = []
    for aoi in range(aoi_count):
        offset = len(rows)
        rows.extend(
            (
                _row(
                    offset,
                    split=split,
                    aoi_id=f"aoi-{aoi}",
                    target_edit="ADD",
                    predicted_edit="ADD",
                    confidence=0.9,
                    delta=0.4,
                ),
                _row(
                    offset + 1,
                    split=split,
                    aoi_id=f"aoi-{aoi}",
                    target_edit="KEEP",
                    predicted_edit="ADD",
                    confidence=0.2,
                    delta=-0.4,
                ),
                _row(
                    offset + 2,
                    split=split,
                    aoi_id=f"aoi-{aoi}",
                    target_edit="ADD",
                    predicted_edit="KEEP",
                    confidence=0.8,
                    delta=0.0,
                ),
                _row(
                    offset + 3,
                    split=split,
                    aoi_id=f"aoi-{aoi}",
                    target_edit="KEEP",
                    predicted_edit="KEEP",
                    confidence=0.8,
                    delta=0.0,
                ),
            )
        )
    return rows


def test_policy_metrics_rejecting_harmful_edit_improves_safety():
    rows = _rows("val", 2)
    scores = [0.9, 0.1, 0.0, 0.0] * 2

    always, _ = policy_metrics(rows, scores, threshold=0.0)
    gated, accepted = policy_metrics(rows, scores, threshold=0.5)

    assert always["false_edit_rate"] == 0.5
    assert gated["false_edit_rate"] == 0.0
    assert gated["map_iou_delta"] > always["map_iou_delta"]
    assert accepted.sum() == 2


def test_calibrate_uses_train_oof_threshold_on_validation():
    result, rows = calibrate(
        _rows("train", 10),
        _rows("val", 5),
        folds=5,
        l2=0.01,
    )

    validation = result["validation"]
    assert result["test_assets_read"] is False
    assert result["calibration"]["selected_threshold"] > 0.0
    assert (
        validation["safe_commit"]["false_edit_rate"]
        < validation["always_commit"]["false_edit_rate"]
    )
    assert (
        validation["safe_commit"]["map_iou_delta"]
        > validation["always_commit"]["map_iou_delta"]
    )
    assert all("safe_commit_score" in row for row in rows)
    frozen_model = deserialize_logistic(
        result["calibration"]["frozen_logistic_model"]
    )
    assert frozen_model["mean"].shape == (11,)
    assert predict_logistic(frozen_model, frozen_model["mean"][None]).shape == (1,)


def test_frozen_calibration_reproduces_calibration_validation_metrics():
    validation_rows = _rows("val", 5)
    calibration, expected_rows = calibrate(
        _rows("train", 10),
        validation_rows,
        folds=5,
        l2=0.01,
    )

    applied, actual_rows = apply_frozen_calibration(
        calibration,
        validation_rows,
    )

    assert applied["safe_commit"] == calibration["validation"]["safe_commit"]
    assert applied["direct"] == calibration["validation"]["always_commit"]
    assert [row["safe_commit_score"] for row in actual_rows] == [
        row["safe_commit_score"] for row in expected_rows
    ]


def test_frozen_calibration_rejects_missing_model():
    calibration, _ = calibrate(
        _rows("train", 10),
        _rows("val", 5),
        folds=5,
        l2=0.01,
    )
    del calibration["calibration"]["frozen_logistic_model"]

    try:
        apply_frozen_calibration(calibration, _rows("val", 5))
    except ValueError as error:
        assert "frozen logistic model" in str(error)
    else:
        raise AssertionError("missing frozen model should fail closed")


def test_safe_commit_paper_prediction_matches_update_contract():
    row = _rows("val", 1)[0]
    row.update(
        {
            "safe_commit_score": 0.8,
            "safe_commit_threshold": 0.6,
            "safe_commit_accepted": True,
            "safe_commit_map_iou": row["committed_map_iou"],
        }
    )

    prediction = UpdatePrediction.model_validate(paper_prediction(row))

    assert prediction.committed is True
    assert prediction.raster_iou == row["committed_map_iou"]
    assert prediction.metadata["confidence_source"] == (
        "train_oof_beneficial_commit_probability"
    )

    row["safe_commit_accepted"] = False
    rejected = UpdatePrediction.model_validate(paper_prediction(row))
    assert abs(rejected.confidence - 0.2) < 1e-12
    assert rejected.metadata["safe_commit_score"] == 0.8
