import pytest

from scripts.evaluate_active_catalog_selector import (
    active_catalog_metrics,
    grouped_bootstrap,
    promotion_gate,
)


def _row(task, target, predicted, utility, oracle, predicted_id=None, target_id=None):
    return {
        "task_id": task,
        "aoi_id": f"aoi-{task}",
        "source_episode": task,
        "target_selection": target,
        "predicted_selection": predicted,
        "target_evidence_id": target_id,
        "predicted_evidence_id": predicted_id,
        "candidate_count": 4,
        "stop_utility": 0.0,
        "policy_utility": utility,
        "oracle_utility": oracle,
        "policy_cost": 1.0 if predicted == "ACQUIRE" else 0.0,
        "regret": oracle - utility,
    }


def test_active_metrics_distinguish_call_and_exact_evidence_quality():
    rows = [
        _row("1", "ACQUIRE", "ACQUIRE", 0.4, 0.4, "e1", "e1"),
        _row("2", "ACQUIRE", "ACQUIRE", -0.1, 0.3, "e2", "e3"),
        _row("3", "STOP", "STOP", 0.0, 0.0),
        _row("4", "STOP", "ACQUIRE", -0.2, 0.0, "e4"),
    ]
    metrics = active_catalog_metrics(rows)
    assert metrics["acquire_recall"] == 1.0
    assert metrics["exact_evidence_recall"] == 0.5
    assert metrics["false_call_rate"] == 0.5
    assert metrics["harmful_call_fraction_of_calls"] == pytest.approx(2 / 3)
    assert metrics["realized_utility_sum"] == pytest.approx(0.1)
    assert metrics["mean_regret"] == pytest.approx(0.15)


def test_grouped_bootstrap_uses_aoi_groups():
    rows = [
        _row("1", "ACQUIRE", "ACQUIRE", 0.4, 0.4, "e1", "e1"),
        _row("2", "STOP", "STOP", 0.0, 0.0),
    ]
    result = grouped_bootstrap(rows, group_key="aoi_id", repetitions=20, seed=7)
    assert result["group_count"] == 2
    assert result["intervals"]["realized_utility_mean"]["observed"] == 0.2


def test_promotion_uses_frozen_99_percent_schema_gate():
    metrics = {
        "predicted_call_rate": 0.2,
        "selection_macro_f1_delta_vs_always_stop": 0.1,
        "realized_utility_mean": 0.02,
        "false_call_rate": 0.05,
        "exact_evidence_recall": 0.2,
        "random_exact_evidence_recall": 0.1,
    }
    bootstrap = {
        "intervals": {"realized_utility_mean": {"ci95_low": 0.01}}
    }
    gate = promotion_gate(0.9997, metrics, bootstrap)
    assert gate["passed"]
    assert gate["valid_action_rate_at_least_0_99"]
    assert not gate["valid_action_rate_1_audit"]
    assert not promotion_gate(0.98, metrics, bootstrap)["passed"]
