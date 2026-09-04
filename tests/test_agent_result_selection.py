import pytest

from activemap.agent.result_selection import (
    _normalized_auc,
    select_checkpoint,
    select_intervention_base,
)


def _metrics(label: str, utility: float, **updates):
    row = {
        "label": label,
        "schema_valid_rate": 1.0,
        "executable_valid_rate": 1.0,
        "rollout_schema_valid_rate": 1.0,
        "rollout_executable_valid_rate": 1.0,
        "acquire_recall": 0.4,
        "false_edit_rate": 0.02,
        "rollout_fallback_rate": 0.0,
        "joint_utility_delta_vs_greedy": 0.1,
        "joint_utility_auc": utility,
        "macro_f1": 0.7,
        "exact_action_accuracy": 0.8,
        "mean_regret": 0.1,
    }
    row.update(updates)
    return row


def test_selector_uses_utility_after_safety_gates() -> None:
    decision = select_checkpoint(
        [_metrics("checkpoint-100", 0.5), _metrics("checkpoint-200", 0.7)]
    )
    assert decision["promotion_passed"] is True
    assert decision["selected_checkpoint"] == "checkpoint-200"


def test_selector_prefers_explicit_quality_cost_primary_utility() -> None:
    lower_legacy = _metrics(
        "quality-winner",
        0.2,
        primary_utility_auc=0.8,
        primary_utility_delta_vs_selector=0.1,
    )
    higher_legacy = _metrics(
        "legacy-winner",
        0.9,
        primary_utility_auc=0.4,
        primary_utility_delta_vs_selector=0.1,
    )
    decision = select_checkpoint([lower_legacy, higher_legacy])
    assert decision["selected_checkpoint"] == "quality-winner"


def test_selector_rejects_high_utility_unsafe_checkpoint() -> None:
    decision = select_checkpoint(
        [
            _metrics("safe", 0.5),
            _metrics("unsafe", 0.9, false_edit_rate=0.2),
        ]
    )
    assert decision["selected_checkpoint"] == "safe"
    unsafe = next(row for row in decision["checkpoints"] if row["label"] == "unsafe")
    assert unsafe["eligible"] is False
    assert "false_edit_rate" in unsafe["failed_gates"]


def test_selector_reports_failure_without_forcing_promotion() -> None:
    decision = select_checkpoint(
        [_metrics("bad", 0.9, executable_valid_rate=0.8)]
    )
    assert decision["promotion_passed"] is False
    assert decision["selected_checkpoint"] is None
    assert decision["best_observed_checkpoint"] == "bad"


def test_normalized_auc_integrates_adjacent_budget_intervals() -> None:
    rows = [
        {"budget": 4.0, "utility": 0.8},
        {"budget": 1.0, "utility": 0.2},
        {"budget": 2.0, "utility": 0.4},
    ]

    assert _normalized_auc(rows, "utility") == pytest.approx(0.5)


def test_normalized_auc_rejects_duplicate_budgets() -> None:
    rows = [
        {"budget": 1.0, "utility": 0.2},
        {"budget": 1.0, "utility": 0.4},
    ]

    with pytest.raises(ValueError, match="unique and strictly increasing"):
        _normalized_auc(rows, "utility")


def test_intervention_base_requires_active_behavior_but_not_promotion() -> None:
    active = _metrics(
        "v3-200",
        0.52,
        false_edit_rate=0.11,
        joint_utility_delta_vs_greedy=-0.04,
        acquire_recall=0.54,
    )
    collapsed = _metrics(
        "v4-200",
        0.54,
        false_edit_rate=0.08,
        joint_utility_delta_vs_greedy=-0.02,
        acquire_recall=0.0,
    )

    decision = select_intervention_base([active, collapsed])

    assert decision["selected_base"] == "v3-200"
    assert decision["selection_passed"] is True
    rejected = next(row for row in decision["candidates"] if row["label"] == "v4-200")
    assert rejected["failed_base_gates"] == ["acquire_recall"]
