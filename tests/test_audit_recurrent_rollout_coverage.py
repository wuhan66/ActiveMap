from scripts.audit_recurrent_rollout_coverage import audit_executed_coverage


def _trajectory(prediction: str, *, tools: int = 0) -> dict:
    return {
        "task_id": "task-a",
        "budget": 1.5,
        "split": "train",
        "prediction": prediction,
        "tool_calls": tools,
    }


def _call(action: str, *, fallback: bool = False) -> dict:
    return {
        "task_id": "task-a",
        "budget": 1.5,
        "split": "train",
        "executed_action": action,
        "fallback_used": fallback,
    }


def test_executed_coverage_requires_all_real_action_strata():
    rollouts = [
        ([_trajectory("REJECT")], [_call("REJECT")]),
        ([_trajectory("COMMIT:ADD")], [_call("COMMIT:ADD")]),
        ([_trajectory("COMMIT:ADD", tools=1)], [_call("USE_TOOL")]),
        ([_trajectory("REJECT", tools=1)], [_call("USE_TOOL")]),
    ]
    report = audit_executed_coverage(
        rollouts,
        minimum_keep_trajectories=2,
        minimum_commit_trajectories=2,
        minimum_tool_trajectories=2,
        maximum_fallback_rate=0.0,
    )

    assert report["ready_for_executable_writeback"] is True
    assert report["executed_terminal_action_counts"] == {"COMMIT:ADD": 2, "REJECT": 2}


def test_executed_coverage_rejects_fallback_or_single_action_mode():
    rollouts = [
        ([_trajectory("COMMIT:ADD")], [_call("COMMIT:ADD", fallback=True)])
        for _ in range(4)
    ]
    report = audit_executed_coverage(
        rollouts,
        minimum_keep_trajectories=1,
        minimum_commit_trajectories=1,
        minimum_tool_trajectories=1,
        maximum_fallback_rate=0.01,
    )

    assert report["ready_for_executable_writeback"] is False
    assert {"executed_keep_support", "tool_trajectory_support", "fallback_rate"} <= set(report["failed_gates"])
