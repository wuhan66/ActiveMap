from __future__ import annotations

import numpy as np

from scripts.train_residual_risk_terminal_scorer import (
    OOFExample,
    select_risk_cap,
    select_safe_policy,
)


def _row(
    *,
    baseline: int,
    candidate: int,
    target: int,
    baseline_utility: float,
    candidate_utility: float,
) -> OOFExample:
    return OOFExample(
        task_id="task",
        features=[0.0],
        baseline=baseline,
        candidate=candidate,
        target=target,
        baseline_utility=baseline_utility,
        candidate_utility=candidate_utility,
    )


def test_safe_policy_accepts_helpful_edit_without_adding_false_edits() -> None:
    rows = [
        _row(
            baseline=0,
            candidate=1,
            target=1,
            baseline_utility=-0.75,
            candidate_utility=1.0,
        ),
        _row(
            baseline=0,
            candidate=1,
            target=0,
            baseline_utility=1.0,
            candidate_utility=-1.25,
        ),
    ]
    result = select_safe_policy(
        np.asarray([1.5, -1.5]),
        np.asarray([0.05, 0.95]),
        rows,
        lambdas=(1.0,),
    )
    assert result["utility_gain"] > 0
    assert result["false_edit_rate"] == 0
    assert result["acceptance_rate"] == 0.5


def test_safe_policy_can_fall_back_to_baseline() -> None:
    rows = [
        _row(
            baseline=0,
            candidate=1,
            target=0,
            baseline_utility=1.0,
            candidate_utility=-1.25,
        )
    ]
    result = select_safe_policy(
        np.asarray([2.0]),
        np.asarray([0.9]),
        rows,
        lambdas=(1.0,),
    )
    assert result["utility_gain"] == 0
    assert result["acceptance_rate"] == 0
    assert result["false_edit_rate"] == 0


def test_risk_cap_recalls_at_least_95_percent_of_harmful_rows() -> None:
    probabilities = np.asarray([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.float32)
    result = select_risk_cap(probabilities, labels)
    assert result["false_edit_recall"] >= 0.95
    assert 0.8 <= result["risk_cap"] <= 0.9
