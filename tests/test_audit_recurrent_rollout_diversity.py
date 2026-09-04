from scripts.audit_recurrent_rollout_diversity import audit_rollout_diversity


def _trajectory(action: str, *, tool_calls: int = 0) -> dict:
    return {
        "task_id": "task-a",
        "budget": 1.5,
        "split": "train",
        "prediction": action,
        "acquisitions": int(tool_calls > 0),
        "tool_calls": tool_calls,
        "tool_successes": tool_calls,
        "contains_nonstop_action": tool_calls > 0,
        "mean_tool_belief_l1_delta": 0.1 if tool_calls else 0.0,
    }


def _call(raw: str, *, zero_logprob: bool = False) -> dict:
    return {
        "task_id": "task-a",
        "budget": 1.5,
        "step": 0,
        "raw_output": raw,
        "executed_action": raw,
        "fallback_used": False,
        "training_payload": {
            "old_token_logprobs": [0.0, 0.0] if zero_logprob else [-0.2, -0.3],
            "old_logprob_source": "teacher_forced_forward",
        },
    }


def test_identical_terminal_rollouts_are_blocked() -> None:
    rollouts = [
        ([_trajectory("COMMIT:ADD")], [_call("COMMIT:ADD")])
        for _ in range(4)
    ]

    report = audit_rollout_diversity(rollouts)

    assert report["variable_raw_completion_rate"] == 0.0
    assert report["variable_action_rate"] == 0.0
    assert report["ready_for_recurrent_grpo"] is False
    assert report["ready_for_tool_belief_grpo"] is False


def test_action_and_tool_diversity_can_pass_collection_gate() -> None:
    rollouts = [
        ([_trajectory("COMMIT:ADD")], [_call("COMMIT:ADD")]),
        ([_trajectory("REJECT")], [_call("REJECT")]),
        ([_trajectory("ACQUIRE", tool_calls=1)], [_call("ACQUIRE")]),
        ([_trajectory("USE_TOOL", tool_calls=2)], [_call("USE_TOOL")]),
    ]

    report = audit_rollout_diversity(rollouts)

    assert report["aligned_initial_states"] is True
    assert report["variable_action_rate"] == 1.0
    assert report["variable_tool_count_rate"] == 1.0
    assert report["ready_for_recurrent_grpo"] is True
    assert report["ready_for_tool_belief_grpo"] is True


def test_all_zero_old_logprobs_block_the_gate() -> None:
    rollouts = [
        ([_trajectory("COMMIT:ADD", tool_calls=index)], [_call(str(index), zero_logprob=True)])
        for index in range(4)
    ]

    report = audit_rollout_diversity(rollouts)

    assert report["zero_or_invalid_logprob_calls"] == 4
    assert "valid_old_logprobs" in report["failed_gates"]
    assert report["ready_for_recurrent_grpo"] is False
