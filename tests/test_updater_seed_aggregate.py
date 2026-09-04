from __future__ import annotations

from typing import Any

import pytest

from activemap.evaluation.updater_seed_aggregate import aggregate_seed_decisions


def _decision(seed_offset: float, *, tested: bool) -> dict[str, Any]:
    return {
        "status": "promoted" if tested else "promoted_validation_only",
        "promoted_checkpoint": "best_quality",
        "test_evaluation": {"macro_f1": 0.8} if tested else None,
        "candidates": [
            {
                "checkpoint_name": "best_quality",
                "constraint_satisfied": True,
                "selected": {
                    "macro_f1": 0.80 + seed_offset,
                    "delete_f1": 0.41 + seed_offset,
                    "false_edit_rate": 0.04,
                    "edit_accuracy": 0.91,
                },
            }
        ],
    }


def test_seed_aggregate_reports_mean_std_and_single_test_access() -> None:
    summary = aggregate_seed_decisions(
        {
            "20260716": _decision(0.0, tested=True),
            "20260717": _decision(0.02, tested=False),
        }
    )
    assert summary["seed_count"] == 2
    assert summary["promotion_count"] == 2
    assert summary["test_evaluation_count"] == 1
    assert summary["test_seed"] == "20260716"
    assert summary["aggregate"]["macro_f1"]["mean"] == pytest.approx(0.81)
    assert summary["aggregate"]["macro_f1"]["sample_std"] > 0.0


def test_seed_aggregate_rejects_repeated_test_access() -> None:
    with pytest.raises(ValueError, match="multiple seeds"):
        aggregate_seed_decisions(
            {
                "20260716": _decision(0.0, tested=True),
                "20260717": _decision(0.01, tested=True),
            }
        )
