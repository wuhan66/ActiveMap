from scripts.build_muno21_direct_vlm_comparison import _budget_row


def test_budget_row_supports_writeback_and_rollout_summaries() -> None:
    assert _budget_row({"budgets": [{"budget": 3.0, "value": 1}]}, 3.0)["value"] == 1
    assert _budget_row({"results": [{"budget": 3.0, "value": 2}]}, 3.0)["value"] == 2
