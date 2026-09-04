import pytest

from scripts.train_sequential_set_utility_ranker import action_metrics, action_traces


def _task(task_id, advantages):
    return {
        "task_id": task_id,
        "candidate_ids": [f"{task_id}-a", f"{task_id}-b"],
        "advantages": advantages,
    }


def test_set_action_chooses_one_candidate_or_stop():
    tasks = [_task("task-1", [0.5, -0.2]), _task("task-2", [-0.4, -0.1])]
    traces = action_traces(tasks, [[0.8, -0.3], [-0.2, -0.5]], threshold=0.0)

    assert [row["predicted_selection"] for row in traces] == ["ACQUIRE", "STOP"]
    assert [row["realized_utility"] for row in traces] == pytest.approx([0.5, 0.0])
    metrics = action_metrics(traces)
    assert metrics["utility_mean"] == pytest.approx(0.25)
    assert metrics["false_call_rate"] == 0.0
    assert metrics["exact_candidate_recall"] == 1.0


def test_set_action_marks_negative_selected_candidate_as_false_call():
    tasks = [_task("task-1", [-0.3, -0.2])]
    traces = action_traces(tasks, [[0.7, 0.1]], threshold=0.0)

    metrics = action_metrics(traces)

    assert traces[0]["predicted_selection"] == "ACQUIRE"
    assert traces[0]["realized_utility"] == pytest.approx(-0.3)
    assert metrics["false_call_rate"] == 1.0
