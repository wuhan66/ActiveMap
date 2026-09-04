from scripts.assess_active_catalog_closed_loop_promotion import assess


def _interval(low, high, delta=0.1):
    return {"ci95_low": low, "ci95_high": high, "delta": delta}


def test_promotion_requires_independent_writeback_and_safety():
    closed_loop = {
        "paired_aoi_comparisons": {
            "qwen_minus_uncertainty_gate": {
                "intervals": {
                    "mean_quality_cost_utility": _interval(0.01, 0.2),
                    "false_edit_rate": _interval(-0.03, 0.01),
                    "terminal_accuracy": _interval(0.0, 0.1),
                }
            }
        }
    }
    writeback = {
        "group_key": "aoi_id",
        "paired_delta": {
            "raster_iou_gain_auc": _interval(0.01, 0.1),
            "vector_replay_iou_auc": _interval(-0.01, 0.1, delta=0.03),
            "vector_delta_topology_valid_auc": _interval(-0.005, 0.01),
            "episode_utility_v2_balanced_auc": _interval(0.01, 0.1),
            "episode_utility_v2_safety_auc": _interval(-0.005, 0.1),
            "episode_utility_v2_cost_aware_auc": _interval(-0.005, 0.1),
            "false_edit_auc": _interval(-0.03, 0.01),
        },
    }
    result = assess(
        closed_loop, writeback, candidate="qwen", baseline="uncertainty_gate"
    )
    assert result["promote"] is True
    assert result["checks"]["vector_replay_improves_observed"] is True
    assert result["checks"]["executable_utility_v2_balanced_significant"] is True


def test_promotion_fails_when_map_quality_ci_crosses_zero():
    closed_loop = {
        "paired_aoi_comparisons": {
            "qwen_minus_uncertainty_gate": {
                "intervals": {
                    "mean_quality_cost_utility": _interval(0.01, 0.2),
                    "false_edit_rate": _interval(-0.03, 0.01),
                    "terminal_accuracy": _interval(0.0, 0.1),
                }
            }
        }
    }
    writeback = {
        "group_key": "aoi_id",
        "paired_delta": {
            "raster_iou_gain_auc": _interval(-0.01, 0.1),
            "vector_replay_iou_auc": _interval(-0.01, 0.1, delta=0.03),
            "vector_delta_topology_valid_auc": _interval(-0.005, 0.01),
            "episode_utility_v2_balanced_auc": _interval(0.01, 0.1),
            "episode_utility_v2_safety_auc": _interval(-0.005, 0.1),
            "episode_utility_v2_cost_aware_auc": _interval(-0.005, 0.1),
            "false_edit_auc": _interval(-0.03, 0.01),
        },
    }
    assert (
        assess(
            closed_loop,
            writeback,
            candidate="qwen",
            baseline="uncertainty_gate",
        )["promote"]
        is False
    )
