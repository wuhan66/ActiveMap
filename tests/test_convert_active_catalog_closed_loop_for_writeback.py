import pytest

from activemap.agent.identifiers import public_task_id
from scripts.convert_active_catalog_closed_loop_for_writeback import convert_trace


def _trace(prediction="RESHAPE", split="val"):
    return {
        "sample_id": "sample",
        "source_episode": "episode",
        "aoi_id": "aoi",
        "split": split,
        "policy": "qwen",
        "budget": 3.0,
        "target_edit": "RESHAPE",
        "predicted_edit": prediction,
        "selected_evidence_ids": ["initial", "acquired"],
        "quality_cost_utility": 0.2,
        "spent_cost": 0.75,
        "tool_calls": 1,
        "tool_cost": 0.18,
        "test_assets_read": False,
    }


def test_convert_closed_loop_trace_preserves_all_acquired_evidence():
    row = convert_trace(_trace())
    assert row["task_id"] == public_task_id("episode")
    assert row["prediction"] == "COMMIT:RESHAPE"
    assert row["selected_evidence_ids"] == ["initial", "acquired"]
    assert row["source_selected_evidence_ids"] == ["initial", "acquired"]
    assert row["writeback_evidence_mode"] == "all"
    assert row["aoi_id"] == "aoi"
    assert row["spent_cost"] == 0.75
    assert row["semantic_tool_called"] is True
    assert row["semantic_tool_cost"] == 0.18


def test_convert_closed_loop_keep_to_reject():
    assert convert_trace(_trace("KEEP"))["prediction"] == "REJECT"


def test_convert_closed_loop_can_use_only_last_selected_evidence():
    row = convert_trace(_trace(), evidence_mode="last")
    assert row["selected_evidence_ids"] == ["acquired"]
    assert row["source_selected_evidence_ids"] == ["initial", "acquired"]
    assert row["writeback_evidence_mode"] == "last"
    assert row["spent_cost"] == 0.75


def test_convert_closed_loop_can_use_only_initial_evidence():
    row = convert_trace(_trace(), evidence_mode="first")
    assert row["selected_evidence_ids"] == ["initial"]
    assert row["source_selected_evidence_ids"] == ["initial", "acquired"]
    assert row["writeback_evidence_mode"] == "first"
    assert row["spent_cost"] == 0.75


def test_convert_closed_loop_last_mode_retains_initial_when_no_acquisition():
    trace = _trace()
    trace["selected_evidence_ids"] = ["initial"]
    assert convert_trace(trace, evidence_mode="last")["selected_evidence_ids"] == [
        "initial"
    ]


def test_convert_closed_loop_rejects_unknown_evidence_mode():
    with pytest.raises(ValueError, match="unsupported"):
        convert_trace(_trace(), evidence_mode="oracle")


def test_convert_closed_loop_refuses_test():
    with pytest.raises(ValueError, match="validation"):
        convert_trace(_trace(split="test"))
