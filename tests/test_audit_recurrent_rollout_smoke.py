from scripts.audit_recurrent_rollout_smoke import audit_smoke


def _trajectory(prediction: str, *, tools: int = 0) -> dict:
    return {
        "task_id": f"task-{prediction}-{tools}",
        "budget": 1.5,
        "split": "train",
        "prediction": prediction,
        "tool_calls": tools,
    }


def _call(action: str, *, fallback: bool = False, valid_logprob: bool = True) -> dict:
    return {
        "task_id": "task",
        "budget": 1.5,
        "split": "train",
        "executed_action": action,
        "fallback_used": fallback,
        "training_payload": {
            "old_token_logprobs": [-0.2] if valid_logprob else [],
            "old_sequence_logprob": -0.2 if valid_logprob else float("nan"),
        },
    }


def test_smoke_audit_requires_real_keep_commit_and_tool_support():
    report = audit_smoke(
        [
            _trajectory("REJECT"),
            _trajectory("COMMIT:ADD", tools=1),
            _trajectory("COMMIT:RESHAPE", tools=1),
        ],
        [_call("REJECT"), _call("USE_TOOL"), _call("COMMIT:ADD")],
        minimum_keep=1,
        minimum_commit=2,
        minimum_tool=2,
        maximum_fallback_rate=0.0,
    )

    assert report["ready_for_four_gpu_collection"] is True
    assert report["executed_keep_trajectories"] == 1
    assert report["executed_commit_trajectories"] == 2


def test_smoke_audit_blocks_invalid_logprobs_and_fallbacks():
    report = audit_smoke(
        [_trajectory("REJECT"), _trajectory("COMMIT:ADD", tools=1)],
        [_call("REJECT", fallback=True), _call("COMMIT:ADD", valid_logprob=False)],
        minimum_keep=1,
        minimum_commit=1,
        minimum_tool=1,
        maximum_fallback_rate=0.01,
    )

    assert report["ready_for_four_gpu_collection"] is False
    assert {"fallback_rate", "valid_old_logprobs"} <= set(report["failed_gates"])
