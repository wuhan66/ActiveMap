import pytest

from scripts.compare_closed_loop_reliability_gate import (
    compare,
    verify_controlled_difference,
)


def _summary(gated: bool, updater_hash: str):
    return {
        "schema_version": "active-catalog-closed-loop-evaluation-v1",
        "protocol": {
            "model_selected_state_transitions": True,
            "oracle_next_state_replay": False,
            "tool_mode": "selective",
            "selective_tool_calling": True,
            "tool_belief_reliability_gate": gated,
            "max_candidates": 16,
            "max_acquisitions": 2,
            "stochastic_policy_sampling": False,
            "temperature": 1.0,
            "top_p": 1.0,
        },
        "metrics": {"tool_call_episode_rate": 0.25},
        "sources": {
            "model": "model",
            "adapter": "adapter",
            "states": "states",
            "episodes": "episodes",
            "tool_gate": {"sha256": "fixed-gate", "summary_sha256": "fixed-summary"},
            "tool_belief_checkpoint": {"sha256": updater_hash},
        },
    }


def _row(aoi: str, correct: bool, utility: float, false_edit: bool = False):
    return {
        "aoi_id": aoi,
        "terminal_correct": correct,
        "false_edit": false_edit,
        "missed_edit": False,
        "acquisitions": 1,
        "steps": 3,
        "spent_cost": 1.0,
        "quality_gain": utility + 1.0,
        "quality_cost_utility": utility,
        "model_action_count": 1,
        "valid_action_count": 1,
        "fallback_count": 0,
    }


def test_reliability_ablation_requires_the_same_tool_gate():
    ungated = _summary(False, "u")
    gated = _summary(True, "g")
    gated["sources"]["tool_gate"]["sha256"] = "different"
    with pytest.raises(ValueError, match="fixed source mismatch"):
        verify_controlled_difference(ungated, gated)


def test_reliability_ablation_reports_paired_promotion():
    keys = [("e1", 1.0), ("e2", 1.0)]
    ungated_rows = {
        keys[0]: _row("a", True, 0.0),
        keys[1]: _row("b", True, 0.0),
    }
    gated_rows = {
        keys[0]: _row("a", True, 0.2),
        keys[1]: _row("b", True, 0.3),
    }
    report = compare(
        _summary(False, "u"),
        _summary(True, "g"),
        ungated_rows,
        gated_rows,
        repetitions=100,
        seed=7,
    )
    assert report["promotion_checks"]["passed"] is True
    assert report["fixed_tool_gate_sha256"] == "fixed-gate"
