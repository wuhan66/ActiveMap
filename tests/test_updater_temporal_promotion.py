import pytest

from activemap.evaluation.updater_temporal_promotion import (
    aggregate_temporal_seed_decisions,
    aggregate_temporal_validation_evaluations,
    select_temporal_checkpoint,
)


def _calibration(name: str, harmonic: float, add: float, remove: float) -> dict:
    return {
        "checkpoint": f"/{name}.pt",
        "checkpoint_epoch": 10,
        "samples": "/samples.jsonl",
        "split": "val",
        "sample_count": 12,
        "constraint_satisfied": True,
        "selected": {
            "add_threshold": 0.25,
            "remove_threshold": 0.6,
            "harmonic_mean_positive_iou": harmonic,
            "add": {
                "mean_positive_iou": add,
                "stable_false_positive_fraction": 0.004,
            },
            "remove": {
                "mean_positive_iou": remove,
                "stable_false_positive_fraction": 0.003,
            },
        },
    }


def test_temporal_promotion_uses_calibrated_harmonic_score() -> None:
    decision = select_temporal_checkpoint(
        {
            "best_quality": _calibration("quality", 0.10, 0.08, 0.14),
            "best_val_loss": _calibration("loss", 0.12, 0.09, 0.18),
        }
    )
    assert decision["promoted_checkpoint"] == "best_val_loss"
    assert decision["test_evaluation"] is None


def test_temporal_seed_aggregate_reports_sample_standard_deviation() -> None:
    first = select_temporal_checkpoint({"quality": _calibration("a", 0.10, 0.08, 0.14)})
    second = select_temporal_checkpoint({"quality": _calibration("b", 0.12, 0.10, 0.16)})
    summary = aggregate_temporal_seed_decisions({"1": first, "2": second})
    assert summary["seed_count"] == 2
    assert summary["aggregate"]["temporal_change_harmonic_iou"]["mean"] == pytest.approx(0.11)
    assert summary["aggregate"]["temporal_change_harmonic_iou"]["sample_std"] == pytest.approx(
        2**0.5 * 0.01
    )


def test_temporal_promotion_rejects_test_or_misaligned_validation_data() -> None:
    first = _calibration("a", 0.10, 0.08, 0.14)
    second = _calibration("b", 0.12, 0.10, 0.16)
    second["sample_count"] = 13
    with pytest.raises(ValueError, match="same validation samples"):
        select_temporal_checkpoint({"a": first, "b": second})

    decision = select_temporal_checkpoint({"a": first})
    decision["test_evaluation"] = {"accessed": True}
    with pytest.raises(ValueError, match="must not contain test"):
        aggregate_temporal_seed_decisions({"1": decision})


def test_temporal_validation_aggregate_includes_safety_and_classification() -> None:
    def evaluation(seed: int, macro_f1: float) -> dict:
        return {
            "split": "val",
            "sample_count": 12,
            "checkpoint": f"/{seed}.pt",
            "add_threshold": 0.25,
            "remove_threshold": 0.6,
            "mean_raster_iou": 0.7,
            "temporal_change": {"mean_added_iou": 0.08, "mean_removed_iou": 0.16},
            "edit_accuracy": 0.65,
            "macro_f1": macro_f1,
            "false_edit_rate": 0.05,
            "missed_update_rate": 0.4,
        }

    summary = aggregate_temporal_validation_evaluations(
        {"1": evaluation(1, 0.4), "2": evaluation(2, 0.5)}
    )
    assert summary["test_evaluation_count"] == 0
    assert summary["aggregate"]["macro_f1"]["mean"] == pytest.approx(0.45)
    assert summary["aggregate"]["temporal_change_harmonic_iou"]["mean"] == pytest.approx(
        2.0 * 0.08 * 0.16 / 0.24
    )
