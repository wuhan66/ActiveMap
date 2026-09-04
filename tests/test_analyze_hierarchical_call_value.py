from scripts.analyze_hierarchical_call_value import analyze


def _row(*, target, called, direct, hierarchical, cost=0.0):
    return {
        "predicted_use_tool": called,
        "target_use_tool": target,
        "direct_operation": "KEEP",
        "hierarchical_operation": "ADD" if called else "KEEP",
        "direct_utility": direct,
        "hierarchical_utility": hierarchical,
        "tool_cost": cost,
    }


def test_call_value_separates_raw_edit_gain_from_executed_cost():
    rows = [
        _row(target=True, called=True, direct=0.0, hierarchical=0.25, cost=0.75),
        _row(target=False, called=True, direct=1.0, hierarchical=0.25, cost=0.75),
        _row(target=False, called=False, direct=1.0, hierarchical=1.0),
    ]

    result = analyze(rows, [0.0, 0.5, 0.75])

    assert result["called_count"] == 2
    assert result["current_total_cost"] == 1.5
    assert result["all_calls"]["raw_edit_gain_sum"] == 1.0
    assert result["all_calls"]["net_utility_gain_sum"] == -0.5
    assert result["break_even_cost_per_call"] == 0.5
    assert result["target_positive_calls"]["count"] == 1
    assert result["target_negative_calls"]["count"] == 1
