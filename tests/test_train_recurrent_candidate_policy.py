import pytest

from scripts.train_recurrent_proxy_grpo import candidate_training_payload_summary


def _call() -> dict[str, object]:
    return {
        "training_payload": {
            "prompt_token_ids": [1, 2, 3],
            "structured_decoder": {
                "decoder": "candidate-sample",
                "candidate_completions": ["{\"action\":\"REJECT\"}", "{\"action\":\"COMMIT\"}"],
                "candidate_completion_token_ids": [[10, 11], [12, 13]],
                "candidate_probabilities": [0.25, 0.75],
                "candidate_temperature": 1.5,
                "chosen_index": 1,
            },
        }
    }


def test_candidate_policy_summary_requires_complete_behaviour_distribution() -> None:
    assert candidate_training_payload_summary([_call()]) == {
        "calls": 1,
        "candidate_size_min": 2,
        "candidate_size_max": 2,
        "temperatures": [1.5],
    }


def test_candidate_policy_summary_rejects_legacy_partial_rollout_payload() -> None:
    call = _call()
    decoder = call["training_payload"]["structured_decoder"]  # type: ignore[index]
    del decoder["candidate_completions"]  # type: ignore[index]
    with pytest.raises(ValueError, match="candidate completions"):
        candidate_training_payload_summary([call])
