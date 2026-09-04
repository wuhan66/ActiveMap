import pytest

from scripts.calibrate_active_catalog_online_utility_threshold import (
    select_threshold,
    threshold_metrics,
)


def records():
    return [
        {"predicted_utility": 0.9, "realized_gain": 0.5, "cost": 1.0},
        {"predicted_utility": 0.8, "realized_gain": 0.2, "cost": 1.0},
        {"predicted_utility": 0.7, "realized_gain": -0.4, "cost": 1.0},
        {"predicted_utility": 0.1, "realized_gain": -0.1, "cost": 1.0},
    ]


def test_threshold_metrics_use_all_state_denominator():
    metrics = threshold_metrics(records(), 0.8)
    assert metrics["call_rate"] == 0.5
    assert metrics["precision"] == 1.0
    assert metrics["realized_utility_mean"] == pytest.approx(0.175)


def test_select_threshold_respects_false_call_constraint():
    selected, _ = select_threshold(
        records(), maximum_false_call_rate=0.0, maximum_call_rate=0.75
    )
    assert selected is not None
    assert selected["threshold"] == 0.8
    assert selected["calls"] == 2


def test_select_threshold_fails_closed_without_positive_utility():
    negative = [
        {"predicted_utility": 0.9, "realized_gain": -0.1, "cost": 1.0},
        {"predicted_utility": 0.1, "realized_gain": -0.2, "cost": 1.0},
    ]
    selected, _ = select_threshold(
        negative, maximum_false_call_rate=1.0, maximum_call_rate=1.0
    )
    assert selected is None
