import pytest

from scripts.evaluate_sn7_frozen_safe_commit import evaluate_frozen


def _rows(split: str, aoi_count: int = 5):
    rows = []
    for aoi in range(aoi_count):
        for target, predicted, probability, delta in (
            ("ADD", "ADD", 0.9, 0.4),
            ("KEEP", "ADD", 0.1, -0.4),
            ("DELETE", "KEEP", 0.2, 0.0),
            ("KEEP", "KEEP", 0.05, 0.0),
        ):
            rows.append(
                {
                    "sample_id": f"{split}-{len(rows)}",
                    "aoi_id": f"aoi-{aoi}",
                    "split": split,
                    "target_edit": target,
                    "predicted_edit": predicted,
                    "confidence": 0.5 + probability / 2,
                    "mean_change_probability": probability,
                    "max_change_probability": probability,
                    "p95_change_probability": probability,
                    "mean_predictive_entropy": 1.0 - probability,
                    "prior_foreground_fraction": 0.2,
                    "predicted_change_fraction": probability / 10,
                    "prior_map_iou": 0.5,
                    "committed_map_iou": 0.5 + delta,
                    "map_iou_delta": delta,
                }
            )
    return rows


@pytest.mark.parametrize("mode", ["source_oof", "all_target_train"])
def test_frozen_evaluation_uses_predeclared_threshold_data(mode):
    result, rows = evaluate_frozen(
        _rows("train"),
        _rows("train"),
        _rows("test"),
        source_backend="source",
        target_backend="target",
        threshold_mode=mode,
        folds=5,
        l2=0.01,
    )

    assert result["protocol"]["threshold_mode"] == mode
    assert result["test"]["sample_count"] == 20
    assert result["test_assets_read"] is True
    assert {row["split"] for row in rows} == {"test"}
    assert all("safe_commit_accepted" in row for row in rows)


def test_frozen_evaluation_rejects_unregistered_threshold_mode():
    with pytest.raises(ValueError, match="unknown threshold mode"):
        evaluate_frozen(
            _rows("train"),
            _rows("train"),
            _rows("test"),
            source_backend="source",
            target_backend="target",
            threshold_mode="test_selected",
            folds=5,
            l2=0.01,
        )
