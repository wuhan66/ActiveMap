import numpy as np
import pytest

from scripts.train_visual_utility_gate import (
    candidate_thresholds,
    feature_protocol_metadata,
    validation_seed_metrics,
)


def test_candidate_thresholds_follow_requested_call_rate_quantiles():
    scores = np.arange(100, dtype=np.float64)
    thresholds = candidate_thresholds(scores, (0.01, 0.10, 0.50))

    assert thresholds == pytest.approx((99.0, 90.0, 50.0))


def test_candidate_thresholds_reject_invalid_rates():
    with pytest.raises(ValueError, match=r"in \(0, 1\]"):
        candidate_thresholds(np.asarray([0.0, 1.0]), (0.0,))


def test_validation_seed_metrics_exposes_negative_seed_utility():
    result = validation_seed_metrics(
        np.asarray([1, 0, 1, 0]),
        np.asarray([0.2, -0.1, -0.3, 0.1]),
        np.asarray([0.9, 0.1, 0.8, 0.2]),
        0.5,
        [1, 1, 2, 2],
    )

    assert result["1"]["proxy_utility_sum"] > 0.0
    assert result["2"]["proxy_utility_sum"] < 0.0


def test_feature_protocol_metadata_preserves_select_stage():
    summary = {"controller_stage": "SELECT", "pooling": "last_mean"}

    result = feature_protocol_metadata(summary, summary)

    assert result == {
        "controller_stage": "SELECT",
        "pooling": "last_mean",
        "feature_protocol": "frozen-adapter-SELECT-prompt-state-with-recorded-pooling",
    }


def test_feature_protocol_metadata_rejects_stage_mismatch():
    with pytest.raises(ValueError, match="controller stages differ"):
        feature_protocol_metadata(
            {"controller_stage": "SELECT", "pooling": "last_mean"},
            {"controller_stage": "PRE_TOOL", "pooling": "last_mean"},
        )
