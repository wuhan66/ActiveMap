import pytest

from scripts.audit_tool_call_reachability_gap import audit


def _static(calls: int):
    return {
        "protocol": {"test_assets_read": False},
        "selected_checkpoint": "checkpoint-100",
        "checkpoints": [
            {
                "label": "checkpoint-100",
                "predicted_tool_calls": calls,
                "grounded_call_recall": 0.4,
            }
        ],
    }


def _recurrent(calls: int, acquisitions: int = 3):
    return {
        "protocol": {"test_assets_read": False},
        "action_counts": {
            "qwen3_4b_sft_tool_to_belief": {
                "ACQUIRE": acquisitions,
                "USE_TOOL": calls,
            }
        },
    }


def test_replicated_static_to_recurrent_gap_requires_hierarchical_gate():
    result = audit(
        {1: _static(4), 2: _static(3), 3: _static(5)},
        {1: _recurrent(0), 2: _recurrent(0), 3: _recurrent(1)},
    )
    assert result["summary"]["static_to_recurrent_reachability_gap_replicated"] is True
    assert result["summary"]["flat_policy_agentic_claim_supported"] is False
    assert result["summary"]["required_intervention"] == "explicit_hierarchical_tool_need_gate"


def test_reachability_audit_requires_paired_seeds():
    with pytest.raises(ValueError, match="same 2 or more seeds"):
        audit({1: _static(1), 2: _static(1)}, {1: _recurrent(0)})


def test_single_seed_reachability_is_diagnostic_not_replicated():
    result = audit(
        {1: _static(4)},
        {1: _recurrent(0)},
        minimum_seeds=1,
    )
    assert result["summary"]["single_seed_diagnostic"] is True
    assert result["summary"]["gap_observed"] is True
    assert result["summary"]["static_to_recurrent_reachability_gap_replicated"] is False
