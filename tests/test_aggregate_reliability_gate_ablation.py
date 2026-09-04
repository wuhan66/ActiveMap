from scripts.aggregate_reliability_gate_ablation import aggregate


def _component():
    return {
        "schema_version": "tool-belief-reliability-gate-ablation-v1",
        "paired_seeds": [1, 2, 3],
        "promotion_passed": True,
    }


def _closed_loop(utility: float, *, passed: bool = True):
    def interval(value: float, low: float = 0.0):
        return {"observed_delta": value, "ci95_low": low, "ci95_high": value}

    return {
        "schema_version": "active-catalog-closed-loop-reliability-gate-ablation-v1",
        "paired_aoi_bootstrap": {
            "intervals": {
                "mean_quality_cost_utility": interval(utility, utility / 2),
                "terminal_accuracy": interval(0.01),
                "false_edit_rate": interval(-0.01),
            }
        },
        "promotion_checks": {
            "tool_calls_nonzero": True,
            "accuracy_noninferior": True,
            "false_edit_noninferior": True,
            "passed": passed,
        },
    }


def test_three_seed_reliability_promotion_requires_replicated_gain():
    report = aggregate(
        _component(),
        {1: _closed_loop(0.2), 2: _closed_loop(0.1), 3: _closed_loop(0.05)},
    )
    assert report["promotion_passed"] is True
    assert report["checks"]["utility_ci_positive_majority"] is True
    assert report["test_assets_read"] is False


def test_three_seed_reliability_rejects_single_seed_gain():
    report = aggregate(
        _component(),
        {
            1: _closed_loop(0.3),
            2: _closed_loop(-0.1, passed=False),
            3: _closed_loop(-0.1, passed=False),
        },
    )
    assert report["promotion_passed"] is False
    assert report["checks"]["utility_ci_positive_majority"] is False
