from scripts.select_sparse_tool_sft_checkpoint import assess_sparse_tool_checkpoints


def _summary(*, recall: float, false_call: float, tool_exact: float = 0.2):
    return {
        "sample_count": 595,
        "schema_valid_rate": 1.0,
        "executable_valid_rate": 1.0,
        "exact_action_accuracy": 0.7,
        "macro_f1": 0.6,
        "test_assets_read": False,
        "tool_metrics": {
            "target_call_count": 10,
            "predicted_call_count": 8,
            "grounded_call_accuracy": recall,
            "tool_positive_exact_accuracy": tool_exact,
            "false_call_rate": false_call,
        },
    }


def test_sparse_tool_selection_applies_recall_and_false_call_gates():
    decision = assess_sparse_tool_checkpoints(
        {
            "safe": _summary(recall=0.3, false_call=0.01),
            "overcalls": _summary(recall=0.8, false_call=0.08),
            "never-calls": _summary(recall=0.0, false_call=0.0, tool_exact=0.0),
        }
    )

    assert decision["selection_passed"] is True
    assert decision["selected_checkpoint"] == "safe"
    failed = {row["label"]: row["failed_gates"] for row in decision["checkpoints"]}
    assert "false_call_rate" in failed["overcalls"]
    assert "grounded_call_recall" in failed["never-calls"]


def test_sparse_tool_selection_keeps_best_observed_when_all_fail():
    decision = assess_sparse_tool_checkpoints(
        {"a": _summary(recall=0.0, false_call=0.0, tool_exact=0.0)}
    )

    assert decision["selection_passed"] is False
    assert decision["selected_checkpoint"] is None
    assert decision["best_observed_checkpoint"] == "a"
