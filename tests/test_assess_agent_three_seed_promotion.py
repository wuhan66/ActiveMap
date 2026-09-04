from scripts.assess_agent_three_seed_promotion import assess


def test_three_seed_promotion_requires_non_dominated_agent() -> None:
    metrics = {
        "false_edit_rate": 0.02,
        "mean_tool_calls": 0.4,
        "mean_tool_belief_l1_delta": 0.1,
        "quality_cost_utility_auc": 0.7,
    }
    delta = {
        "quality_cost_utility_auc": {"delta": 0.1, "ci95_low": 0.03, "ci95_high": 0.16},
        "false_edit_rate": {"delta": 0.0, "ci95_low": -0.01, "ci95_high": 0.005},
    }
    comparison = {
        "per_seed": [
            {"seed": seed, "candidate": metrics, "baseline": {}, "delta": {}}
            for seed in (1, 2, 3)
        ],
        "paired_delta": delta,
    }
    report = {
        "protocol": {"test_assets_read": False, "model_seeds": [1, 2, 3]},
        "comparisons": {
            name: comparison
            for name in (
                "edit_conditioned_selector",
                "generic_selector",
                "qwen3_4b_sft_tools_no_belief",
                "forced_tools",
            )
        },
    }
    decision = assess(report)
    assert decision["promote"] is True
    assert decision["failed_gates"] == []


def test_three_seed_promotion_rejects_zero_tool_use() -> None:
    metrics = {
        "false_edit_rate": 0.02,
        "mean_tool_calls": 0.0,
        "mean_tool_belief_l1_delta": 0.0,
        "quality_cost_utility_auc": 0.7,
    }
    delta = {
        "quality_cost_utility_auc": {"delta": 0.1, "ci95_low": 0.03, "ci95_high": 0.16},
        "false_edit_rate": {"delta": 0.0, "ci95_low": -0.01, "ci95_high": 0.005},
    }
    comparison = {
        "per_seed": [{"candidate": metrics} for _ in range(3)],
        "paired_delta": delta,
    }
    report = {
        "protocol": {"test_assets_read": False, "model_seeds": [1, 2, 3]},
        "comparisons": {
            name: comparison
            for name in (
                "edit_conditioned_selector",
                "generic_selector",
                "qwen3_4b_sft_tools_no_belief",
                "forced_tools",
            )
        },
    }
    decision = assess(report)
    assert decision["promote"] is False
    assert "nonzero_tool_use" in decision["failed_gates"]
