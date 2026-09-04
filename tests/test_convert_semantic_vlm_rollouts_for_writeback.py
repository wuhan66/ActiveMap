import pytest

from scripts.convert_semantic_vlm_rollouts_for_writeback import convert_trace


def _trace(operation="ADD", split="val"):
    return {
        "split": split,
        "policy_operation": operation,
        "target_operation": "ADD",
        "evidence_id": "evidence",
        "task_id": "task",
        "predicted_use_tool": True,
        "tool_cost": 0.75,
        "policy_utility": 0.25,
        "example_id": "example",
    }


def test_convert_trace_binds_visual_evidence_and_typed_edit():
    row = convert_trace(_trace(), budget=0.75)
    assert row["prediction"] == "COMMIT:ADD"
    assert row["selected_evidence_ids"] == ["evidence"]
    assert row["semantic_tool_called"] is True


def test_convert_trace_maps_keep_to_reject():
    assert convert_trace(_trace("KEEP"), budget=0.75)["prediction"] == "REJECT"


def test_convert_trace_refuses_test_split():
    with pytest.raises(ValueError, match="train or validation"):
        convert_trace(_trace(split="test"), budget=0.75)
