import pytest

from scripts.evaluate_cached_policy_relative_gate import select_cached_branches


def test_select_cached_branches_uses_frozen_threshold_and_cached_utility():
    rows = [
        {
            "example_id": "a",
            "direct_operation": "KEEP",
            "post_tool_operation": "ADD",
            "direct_utility": 0.0,
            "post_tool_utility": 0.25,
        },
        {
            "example_id": "b",
            "direct_operation": "DELETE",
            "post_tool_operation": "KEEP",
            "direct_utility": 1.0,
            "post_tool_utility": -0.75,
        },
    ]
    result = select_cached_branches(rows, {"a": 0.8, "b": 0.2}, 0.5)

    assert result[0]["predicted_use_tool"] is True
    assert result[0]["policy_operation"] == "ADD"
    assert result[0]["policy_utility"] == 0.25
    assert result[1]["predicted_use_tool"] is False
    assert result[1]["policy_operation"] == "DELETE"
    assert result[1]["policy_utility"] == 1.0


def test_select_cached_branches_rejects_id_mismatch():
    with pytest.raises(ValueError, match="differ"):
        select_cached_branches([{"example_id": "a"}], {"b": 0.5}, 0.5)
