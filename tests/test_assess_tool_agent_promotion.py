from scripts.assess_tool_agent_promotion import assess_tool_agent


def _comparison(*, utility_low: float, false_high: float = 0.0):
    return {
        "paired_delta": {
            "quality_cost_utility_auc": {
                "delta": utility_low + 0.01,
                "ci95_low": utility_low,
                "ci95_high": utility_low + 0.02,
            },
            "false_edit_rate": {
                "delta": 0.0,
                "ci95_low": -0.01,
                "ci95_high": false_high,
            },
        }
    }


def _summary():
    method = "qwen3_4b_sft_tool_to_belief"
    return {
        "results": [
            {
                "method": method,
                "sample_count": 138,
                "false_edit_rate": 0.04,
                "mean_tool_calls": 0.1,
                "tool_success_rate": 1.0,
                "mean_tool_belief_l1_delta": 0.02,
                "tool_action_flip_rate": 0.01,
                "terminal_edit_flip_rate": 0.01,
            }
        ],
        "llm_validity_by_method": {
            method: {
                "schema_valid_rate": 1.0,
                "executable_valid_rate": 1.0,
                "fallback_rate": 0.0,
            }
        },
    }


def _writeback():
    positive = {
        "delta": 0.02,
        "ci95_low": 0.01,
        "ci95_high": 0.03,
    }
    neutral = {"delta": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    return {
        "paired_delta": {
            "raster_iou_auc": positive,
            "added_polygon_iou_auc": positive,
            "removed_polygon_iou_auc": positive,
            "vector_delta_topology_valid_auc": neutral,
            "component_count_absolute_error_auc": neutral,
        }
    }


def test_tool_agent_passes_only_with_three_positive_paired_comparisons():
    result = assess_tool_agent(
        _summary(),
        {
            "selector": _comparison(utility_low=0.01, false_high=0.005),
            "no_belief": _comparison(utility_low=0.005),
            "forced": _comparison(utility_low=0.02),
        },
        writeback_comparison=_writeback(),
    )
    assert result["single_seed_intervention_passed"] is True


def test_tool_agent_rejects_nonpositive_recurrent_gain():
    result = assess_tool_agent(
        _summary(),
        {
            "selector": _comparison(utility_low=0.01),
            "no_belief": _comparison(utility_low=-0.01),
            "forced": _comparison(utility_low=0.02),
        },
        writeback_comparison=_writeback(),
    )
    assert result["single_seed_intervention_passed"] is False
    assert "recurrent_belief_utility_positive_ci" in result["failed_gates"]
