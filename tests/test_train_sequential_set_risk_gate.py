import numpy as np
import pytest

from scripts.train_sequential_set_risk_gate import gated_action_traces, task_score_features


def test_task_score_features_are_permutation_invariant():
    left = task_score_features([np.asarray([0.1, 0.7, -0.2])])
    right = task_score_features([np.asarray([-0.2, 0.1, 0.7])])

    assert left == pytest.approx(right)


def test_risk_gate_can_stop_a_highest_scoring_negative_candidate():
    tasks = [
        {
            "task_id": "task-1",
            "candidate_ids": ["evidence-a", "evidence-b"],
            "advantages": [-0.4, -0.2],
        }
    ]
    traces = gated_action_traces(
        tasks,
        [np.asarray([0.8, 0.1])],
        np.asarray([0.2]),
        threshold=0.5,
    )

    assert traces[0]["predicted_selection"] == "STOP"
    assert traces[0]["realized_utility"] == 0.0
