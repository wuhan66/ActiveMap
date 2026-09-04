from scripts.assess_agent_dpo_paired import assess


def _interval(low: float, high: float) -> dict[str, float]:
    return {"delta": (low + high) / 2.0, "ci95_low": low, "ci95_high": high}


def _comparisons() -> tuple[dict, dict]:
    actions = {
        "paired_delta": {
            "acquire_recall": _interval(-0.02, 0.04),
            "macro_f1": _interval(0.01, 0.05),
        }
    }
    rollouts = {
        "paired_delta": {
            "joint_utility_auc": _interval(-0.005, 0.02),
            "quality_cost_utility_auc": _interval(0.01, 0.03),
            "terminal_accuracy": _interval(-0.005, 0.03),
            "false_edit_rate": _interval(-0.02, 0.005),
            "missed_edit_rate": _interval(-0.02, 0.01),
            "mean_cost": _interval(-0.1, 0.02),
        }
    }
    return actions, rollouts


def test_paired_assessment_requires_noninferiority_and_positive_evidence() -> None:
    actions, rollouts = _comparisons()
    result = assess(actions, rollouts, {"promotion_passed": True})
    assert result["single_seed_intervention_passed"] is True
    assert result["benefit_evidence"]["macro_f1_improved"] is True
    assert result["protocol"]["paper_promotion_requires_three_seeds"] is True
    assert result["protocol"]["primary_utility_metric"] == "quality_cost_utility_auc"


def test_paired_assessment_rejects_unsafe_false_edit_gain() -> None:
    actions, rollouts = _comparisons()
    rollouts["paired_delta"]["false_edit_rate"] = _interval(0.02, 0.04)
    result = assess(actions, rollouts, {"promotion_passed": True})
    assert result["single_seed_intervention_passed"] is False
    assert "false_edit_noninferior" in result["failed_gates"]


def test_paired_assessment_uses_executable_writeback_as_primary_map_quality() -> None:
    actions, rollouts = _comparisons()
    writeback = {
        "paired_delta": {
            "raster_iou_auc": _interval(0.01, 0.03),
            "vector_delta_topology_valid_auc": _interval(-0.005, 0.01),
            "component_count_absolute_error_auc": _interval(-0.2, -0.05),
        }
    }

    result = assess(
        actions,
        rollouts,
        {"promotion_passed": True},
        writeback,
    )

    assert result["single_seed_intervention_passed"] is True
    assert result["protocol"]["primary_map_quality_metric"] == "raster_iou_auc"
    assert result["benefit_evidence"]["map_raster_iou_improved"] is True
