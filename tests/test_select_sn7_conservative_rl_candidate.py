import pytest

from scripts.select_sn7_conservative_rl_candidate import select_candidate


def _comparison(utility, quality, false_edit, accuracy, cost):
    def interval(value, low=None, high=None):
        return {
            "observed_delta": value,
            "ci95_low": value if low is None else low,
            "ci95_high": value if high is None else high,
        }

    return {
        "intervals": {
            "terminal_accuracy": interval(accuracy, accuracy - 0.01, accuracy + 0.01),
            "false_edit_rate": interval(false_edit, false_edit - 0.01, false_edit),
            "mean_quality_gain": interval(quality, quality - 0.01, quality + 0.01),
            "mean_cost": interval(cost, cost - 0.01, cost + 0.01),
            "mean_quality_cost_utility": interval(
                utility, utility - 0.01, utility + 0.01
            ),
        },
        "non_dominated_observed": (
            utility > 0 and accuracy >= 0 and false_edit <= 0
        ),
    }


def _table(comparisons):
    return {
        "schema_version": "sn7-common-controller-validation-table-v1",
        "split": "val",
        "test_assets_read": False,
        "reference": "sft",
        "record_count": 512,
        "paired_vs_reference": comparisons,
    }


def test_selector_promotes_only_best_fully_eligible_candidate():
    table = _table(
        {
            "rl_a": _comparison(0.03, 0.04, 0.0, 0.02, 0.01),
            "rl_b": _comparison(0.04, 0.03, 0.0, 0.01, 0.02),
        }
    )
    result = select_candidate(table, ["rl_a", "rl_b"])
    assert result["decision"] == "promote_one"
    assert result["selected_candidate"] == "rl_b"


def test_selector_stops_rl_when_utility_ci_crosses_zero():
    table = _table({"rl": _comparison(0.005, 0.04, 0.0, 0.02, 0.01)})
    result = select_candidate(table, ["rl"])
    assert result["decision"] == "stop_rl"
    assert result["selected_candidate"] is None
    assert result["candidate_assessments"]["rl"]["gates"]["utility_ci_positive"] is False


def test_selector_rejects_false_edit_regression():
    table = _table({"rl": _comparison(0.03, 0.04, 0.01, 0.02, 0.01)})
    result = select_candidate(table, ["rl"])
    assert result["decision"] == "stop_rl"
    assert (
        result["candidate_assessments"]["rl"]["gates"][
            "false_edit_ci_noninferior"
        ]
        is False
    )


def test_selector_rejects_missing_candidate():
    with pytest.raises(ValueError, match="missing paired"):
        select_candidate(_table({}), ["rl"])
