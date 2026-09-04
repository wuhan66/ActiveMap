from scripts.assess_muno21_v10_diagnostic import assess


def _static():
    return {
        "protocol": {"test_assets_read": False},
        "selection_passed": True,
        "selected_checkpoint": "checkpoint-300",
        "checkpoints": [
            {"label": "checkpoint-300", "predicted_tool_calls": 5}
        ],
    }


def _rollout(candidate_utility: float = 0.3):
    values = {
        "qwen3_4b_sft_tool_to_belief": (candidate_utility, 0.03, 0.2, 0.1),
        "edit_conditioned_selector": (0.1, 0.03, 0.0, 0.0),
        "qwen3_4b_sft_tools_no_belief": (0.15, 0.03, 0.2, 0.0),
        "forced_tools": (0.05, 0.04, 1.0, 0.1),
    }
    rows = []
    for method, (utility, false_edit, calls, belief) in values.items():
        for budget in (1.5, 3.0, 4.5):
            rows.append(
                {
                    "method": method,
                    "budget": budget,
                    "mean_quality_cost_utility": utility,
                    "false_edit_rate": false_edit,
                    "mean_tool_calls": calls,
                    "mean_tool_belief_l1_delta": belief,
                }
            )
    return {
        "protocol": {"test_assets_read": False},
        "results": rows,
        "action_counts": {
            "qwen3_4b_sft_tool_to_belief": {"USE_TOOL": 3}
        },
    }


def _reachability():
    return {
        "summary": {"single_seed_diagnostic": True},
        "test_assets_read": False,
    }


def test_v10_diagnostic_passes_only_end_to_end_gain():
    result = assess(_static(), _rollout(), _reachability())
    assert result["diagnostic_passed"] is True
    assert result["next_stage"] == "replicate_balanced_tool_sft"
    assert "not paper claim" in result["claim_boundary"]


def test_v10_diagnostic_rejects_no_utility_gain():
    result = assess(_static(), _rollout(candidate_utility=0.05), _reachability())
    assert result["diagnostic_passed"] is False
    assert result["checks"]["utility_above_edit_selector"] is False
