from scripts.analyze_terminal_arbitration_frontier import quantiles, summarize


def test_quantiles_preserve_endpoints() -> None:
    assert quantiles([0.0, 0.1, 0.2, 0.3], 3) == [0.0, 0.2, 0.3]


def test_summarize_rollout_rows() -> None:
    rows = [
        {
            "terminal_correct": True,
            "false_edit": False,
            "missed_edit": False,
            "episode_utility_v2_proxy_balanced": 1.0,
            "episode_utility_v2_proxy_safety": 1.0,
            "spent_cost": 0.2,
            "tool_calls": 1,
        },
        {
            "terminal_correct": False,
            "false_edit": True,
            "missed_edit": False,
            "episode_utility_v2_proxy_balanced": -1.0,
            "episode_utility_v2_proxy_safety": -2.0,
            "spent_cost": 0.0,
            "tool_calls": 0,
        },
    ]
    result = summarize(rows)
    assert result["terminal_accuracy"] == 0.5
    assert result["false_edit_rate"] == 0.5
    assert result["balanced_utility"] == 0.0
    assert result["safety_utility"] == -0.5
    assert result["mean_cost"] == 0.1
