import numpy as np
import pytest

from activemap.features import EVIDENCE_DIM, HYPOTHESIS_DIM, STATE_DIM
from activemap.models import EditOperation
from activemap.policy.rollout import evaluate_rollouts, rollout_score_policy
from activemap.selector_records import SelectorSample


def _sample() -> SelectorSample:
    return SelectorSample(
        sample_id="sample",
        split="test",
        edit_type=EditOperation.ADD,
        hypothesis_features=[0.0] * HYPOTHESIS_DIM,
        state_features=[0.0] * STATE_DIM,
        evidence_ids=["cheap", "best", "medium"],
        evidence_features=[[0.0] * EVIDENCE_DIM for _ in range(3)],
        evidence_costs=[1.0, 2.0, 1.0],
        false_edit_risks=[0.1, 0.1, 0.1],
        oracle_utilities=[0.2, 0.9, 0.5],
        stop_utility=0.0,
    )


def test_rollout_respects_budget_and_records_trace() -> None:
    result = rollout_score_policy(
        _sample(),
        budget=2.5,
        score_fn=lambda sample: np.asarray(sample.oracle_utilities, dtype=np.float32),
    )
    assert result.selected_evidence_ids == ("best",)
    assert result.spent_cost == pytest.approx(2.0)
    assert result.utility == pytest.approx(0.9)
    assert result.regret == pytest.approx(0.0)
    assert result.stop_reason == "budget_exhausted"


def test_rollout_honors_explicit_stop_action() -> None:
    result = rollout_score_policy(
        _sample(),
        budget=4.0,
        score_fn=lambda sample: np.asarray([*sample.oracle_utilities, 2.0]),
    )
    assert result.selected_evidence_ids == ()
    assert result.stop_reason == "policy_stop"
    summary, traces = evaluate_rollouts(
        [_sample()],
        method="stop",
        budget=4.0,
        score_fn=lambda sample: np.asarray([*sample.oracle_utilities, 2.0]),
    )
    assert summary["policy_stop_rate"] == pytest.approx(1.0)
    assert len(traces) == 1


def test_rollout_accumulates_harmful_acquisition_costs() -> None:
    sample = _sample().model_copy(
        update={
            "evidence_ids": ["useful", "harmful"],
            "evidence_features": [[0.0] * EVIDENCE_DIM for _ in range(2)],
            "evidence_costs": [1.0, 1.0],
            "false_edit_risks": [0.0, 0.0],
            "oracle_utilities": [0.4, -0.2],
            "metadata": {"cost_weight": 0.2},
        }
    )
    result = rollout_score_policy(
        sample,
        budget=2.0,
        score_fn=lambda current: np.asarray([2.0] * len(current.evidence_ids)),
    )
    assert result.selected_evidence_ids == ("useful", "harmful")
    assert result.steps[0].utility_after == pytest.approx(0.4)
    assert result.steps[1].utility_after == pytest.approx(0.2)
    assert result.utility == pytest.approx(0.2)
    assert result.oracle_utility == pytest.approx(0.4)


def test_executable_rollout_uses_terminal_scores_and_budget_normalized_cost() -> None:
    sample = _sample().model_copy(
        update={
            "evidence_ids": ["better", "harmful"],
            "evidence_features": [[0.0] * EVIDENCE_DIM for _ in range(2)],
            "evidence_costs": [1.0, 1.0],
            "false_edit_risks": [1.0, 0.0],
            "oracle_utilities": [999.0, 999.0],
            "state_features": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
            "metadata": {
                "utility_mode": "executable",
                "utility_profile": "balanced",
                "selected_evidence_ids": ["initial"],
                "executable_outcomes": {
                    "initial": {"terminal_score_before_cost": 0.2},
                    "better": {"terminal_score_before_cost": 0.6},
                    "harmful": {"terminal_score_before_cost": -0.1},
                },
            },
        }
    )
    result = rollout_score_policy(
        sample,
        budget=2.0,
        score_fn=lambda current: np.asarray([2.0] * len(current.evidence_ids)),
    )
    assert result.selected_evidence_ids == ("better", "harmful")
    assert result.steps[0].utility == pytest.approx(0.35)
    assert result.steps[1].utility == pytest.approx(-0.75)
    assert result.utility == pytest.approx(0.3)
    assert result.oracle_utility == pytest.approx(0.35)
