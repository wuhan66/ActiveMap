import pytest

from scripts.combine_pointer_action_gates import combine_gates


def test_combined_gate_requires_both_independent_audits() -> None:
    report = combine_gates(
        {"test_assets_read": False, "ready_for_tool_belief_grpo": True},
        {"test_assets_read": False, "ready_for_executable_grpo": True},
    )

    assert report["ready_for_executable_grpo"] is True


def test_combined_gate_records_failed_validation() -> None:
    report = combine_gates(
        {"test_assets_read": False, "ready_for_tool_belief_grpo": True},
        {"test_assets_read": False, "ready_for_executable_grpo": False},
    )

    assert report["ready_for_executable_grpo"] is False
    assert report["failed_gates"] == ["greedy_pointer_validation"]


def test_combined_gate_rejects_ambiguous_test_access() -> None:
    with pytest.raises(ValueError, match="test"):
        combine_gates(
            {"test_assets_read": None, "ready_for_tool_belief_grpo": True},
            {"test_assets_read": False, "ready_for_executable_grpo": True},
        )
