import pytest

from scripts.audit_sequential_selector_task_variance import summarize_task_variance


def _row(task: str, trajectory: str, target: str, prediction: str, advantage: float) -> dict:
    return {
        "task_id": task,
        "trajectory_id": trajectory,
        "split": "val",
        "target_selection": target,
        "predicted_selection": prediction,
        "policy_relative_advantage": advantage,
    }


def test_task_variance_reports_seed_averaged_realized_utility_and_disagreement():
    seed_a = [
        _row("task-a", "a-1", "ACQUIRE", "ACQUIRE", 1.0),
        _row("task-a", "a-2", "STOP", "STOP", -1.0),
        _row("task-b", "b-1", "STOP", "ACQUIRE", -0.75),
    ]
    seed_b = [
        _row("task-a", "a-1", "ACQUIRE", "STOP", 1.0),
        _row("task-a", "a-2", "STOP", "STOP", -1.0),
        _row("task-b", "b-1", "STOP", "STOP", -0.75),
    ]

    report, tasks = summarize_task_variance({"a": seed_a, "b": seed_b})

    assert report["task_count"] == 2
    assert report["task_utility_distribution"]["mean"] == pytest.approx(0.0625)
    assert report["task_utility_distribution"]["sign_counts"] == {
        "negative": 1,
        "positive": 1,
    }
    assert report["total_state_prediction_disagreements"] == 2
    assert tasks[0]["mean_realized_utility_sum"] == pytest.approx(0.5)
    assert tasks[1]["mean_realized_utility_sum"] == pytest.approx(-0.375)
