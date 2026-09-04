import numpy as np
import pytest

from scripts.train_visual_tool_gate import (
    utility_proxy_metrics,
    utility_promotion_checks,
    utility_risk_weights,
    utility_sample_weights,
)


def test_utility_proxy_metrics_scores_only_predicted_calls():
    result = utility_proxy_metrics(
        np.asarray([0.9, 0.8, 0.2]),
        np.asarray([1.0, -0.5, 3.0]),
        0.5,
    )
    assert result["proxy_utility_sum"] == 0.5
    assert result["proxy_utility_mean"] == pytest.approx(1.0 / 6.0)
    assert result["proxy_risk_sum"] == 0.5


def test_utility_proxy_metrics_rejects_missing_metadata():
    with pytest.raises(ValueError, match="finite utility"):
        utility_proxy_metrics(
            np.asarray([0.9]),
            np.asarray([np.nan]),
            0.5,
        )


def test_utility_sample_weights_preserve_class_mass_and_rank_magnitude():
    labels = np.asarray([0, 0, 0, 1, 1])
    utilities = np.asarray([-0.1, -1.0, -2.0, 0.25, 1.25])
    weights = utility_sample_weights(labels, utilities)

    assert np.all(np.isfinite(weights))
    assert np.all(weights > 0.0)
    assert weights[0] < weights[1] < weights[2]
    assert weights[3] < weights[4]
    assert weights[labels == 0].mean() == pytest.approx(1.0)
    assert weights[labels == 1].mean() == pytest.approx(1.0)


def test_utility_sample_weights_reject_missing_metadata():
    with pytest.raises(ValueError, match="finite utility"):
        utility_sample_weights(np.asarray([0]), np.asarray([np.nan]))


def test_utility_risk_weights_preserve_asymmetric_cost_and_floor():
    weights = utility_risk_weights(np.asarray([0.0, 0.25, -2.75]))

    assert weights.tolist() == pytest.approx([0.05, 0.25, 2.75])


def test_utility_risk_weights_reject_missing_metadata():
    with pytest.raises(ValueError, match="finite utility"):
        utility_risk_weights(np.asarray([np.nan]))


def test_utility_promotion_requires_positive_oof_and_validation_value():
    checks = utility_promotion_checks(
        {"proxy_utility_sum": -1.0},
        {"proxy_utility_sum": 2.0},
        selection_objective="proxy_utility",
    )

    assert checks == {
        "oof_proxy_utility_positive": False,
        "validation_proxy_utility_positive": True,
    }
