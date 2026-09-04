from __future__ import annotations

import pytest

from activemap.evaluation.operation_selector_promotion import (
    aggregate_operation_selector_seeds,
    select_operation_selector_checkpoint,
)


def _calibration(macro_f1: float, false_edit: float) -> dict:
    return {
        "constraint_satisfied": false_edit <= 0.05,
        "sample_count": 10,
        "selected": {
            "update_threshold": 0.8,
            "accuracy": macro_f1,
            "macro_f1": macro_f1,
            "balanced_accuracy": macro_f1,
            "false_edit_rate": false_edit,
            "missed_update_rate": 0.4,
            "f1_keep": macro_f1,
            "f1_add": macro_f1,
            "f1_delete": macro_f1,
            "f1_reshape": macro_f1,
        },
        "test_evaluation": None,
    }


def test_promotion_prefers_best_feasible_checkpoint() -> None:
    decision = select_operation_selector_checkpoint(
        {
            "unsafe": _calibration(0.9, 0.2),
            "quality": _calibration(0.6, 0.04),
            "loss": _calibration(0.5, 0.03),
        }
    )
    assert decision["promoted_checkpoint"] == "quality"
    assert decision["test_evaluation"] is None


def test_seed_aggregate_uses_sample_standard_deviation_and_rejects_test() -> None:
    first = select_operation_selector_checkpoint({"best": _calibration(0.5, 0.04)})
    second = select_operation_selector_checkpoint({"best": _calibration(0.7, 0.02)})
    summary = aggregate_operation_selector_seeds({"1": first, "2": second})
    assert summary["aggregate"]["macro_f1"]["mean"] == pytest.approx(0.6)
    assert summary["aggregate"]["macro_f1"]["std"] == pytest.approx(2**0.5 / 10)
    assert summary["all_constraints_satisfied"] is True
    second["test_evaluation"] = {"macro_f1": 1.0}
    with pytest.raises(ValueError, match="test evaluation"):
        aggregate_operation_selector_seeds({"1": first, "2": second})
