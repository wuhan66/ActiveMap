import json

from scripts.calibrate_sequential_selector_threshold import (
    apply_threshold,
    select_threshold,
)
from scripts.split_sequential_selector_fit_calibration import split_by_task


def _trace(task: str, target: str, margin: float, advantage: float) -> dict:
    return {
        "task_id": task,
        "split": "train",
        "target_selection": target,
        "predicted_selection": "STOP",
        "policy_relative_advantage": advantage,
        "decision_token_log_probability_margin": margin,
    }


def _sft_row(task: str, trajectory: str, selection: str) -> dict:
    action = {"stage": "SELECT", "selection": selection}
    if selection == "ACQUIRE":
        action["evidence_id"] = "evidence-1"
    return {
        "task_id": task,
        "trajectory_id": trajectory,
        "split": "train",
        "stage": "SELECT",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "system"}]},
            {
                "role": "user",
                "content": [{"type": "text", "text": '{"controller_stage":"SELECT"}'}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": json.dumps(action)}],
            },
        ],
    }


def test_task_split_is_deterministic_disjoint_and_stratified():
    rows = [
        _sft_row(f"task-{index}", f"trajectory-{index}", "ACQUIRE" if index % 4 == 0 else "STOP")
        for index in range(20)
    ]
    first_fit, first_calibration, first_summary = split_by_task(
        rows, calibration_fraction=0.2, seed=7
    )
    second_fit, second_calibration, _ = split_by_task(
        rows, calibration_fraction=0.2, seed=7
    )

    assert [row["task_id"] for row in first_fit] == [row["task_id"] for row in second_fit]
    assert [row["task_id"] for row in first_calibration] == [
        row["task_id"] for row in second_calibration
    ]
    assert {row["task_id"] for row in first_fit}.isdisjoint(
        {row["task_id"] for row in first_calibration}
    )
    assert first_summary["task_overlap"] == 0
    assert {row["messages"][2]["content"][0]["text"] for row in first_calibration}


def test_threshold_selection_maximizes_feasible_utility():
    rows = [
        _trace("a", "ACQUIRE", 3.0, 1.0),
        _trace("b", "ACQUIRE", 2.0, 0.75),
        _trace("c", "STOP", 1.0, -0.75),
        _trace("d", "STOP", -1.0, -1.0),
        _trace("e", "STOP", -2.0, -0.75),
    ]

    selected, grid = select_threshold(
        rows, max_call_rate=0.5, max_false_call_rate=0.34, min_recall=0.5
    )
    evaluated = apply_threshold(rows, selected["threshold"])

    assert grid
    assert selected["metrics"]["realized_utility_sum"] == 1.75
    assert [row["predicted_selection"] for row in evaluated] == [
        "ACQUIRE",
        "ACQUIRE",
        "STOP",
        "STOP",
        "STOP",
    ]
