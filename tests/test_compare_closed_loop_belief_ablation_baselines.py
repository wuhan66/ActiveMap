import copy

import pytest

from scripts.compare_closed_loop_belief_ablation_baselines import (
    verify_controlled_difference,
)


def _summary(*, belief_mode: str, reliability_gate: bool, checkpoint: str) -> dict:
    return {
        "schema_version": "active-catalog-closed-loop-baselines-v1",
        "sample_count": 6,
        "split": "val",
        "protocol": {
            "same_max_acquisitions": 2,
            "tool_mode": "selective",
            "max_tool_calls": 4,
            "explicit_geospatial_tool_calls": True,
            "selective_tool_calling": True,
            "map_relative_semantic_tool": False,
            "semantic_selective_calling": False,
            "inputs": {
                "states": {"sha256": "states"},
                "episodes": {"sha256": "episodes"},
            },
            "learned_selectors": {"edit_utility": {"sha256": "selector"}},
            "post_tool_action_adapter": {"sha256": "adapter"},
            "evaluated_policies": ["edit_utility"],
            "deterministic_sample_seed": 17,
            "belief_mode": belief_mode,
            "tool_belief_checkpoint": {
                "sha256": checkpoint,
                "reliability_gate": reliability_gate,
            },
            "tool_gate": {"sha256": "tool-gate"},
            "quality_gain_semantics": "executable_terminal_score",
            "test_assets_read": False,
        },
    }


def test_verifies_ungated_controlled_difference() -> None:
    gated = _summary(
        belief_mode="recurrent",
        reliability_gate=True,
        checkpoint="gated",
    )
    ungated = _summary(
        belief_mode="recurrent",
        reliability_gate=False,
        checkpoint="ungated",
    )

    report = verify_controlled_difference(ungated, gated, ablation="ungated")

    assert report["controlled_difference"] == "tool_belief_reliability_gate"


def test_verifies_frozen_prior_controlled_difference() -> None:
    gated = _summary(
        belief_mode="recurrent",
        reliability_gate=True,
        checkpoint="gated",
    )
    frozen = _summary(
        belief_mode="frozen_prior",
        reliability_gate=True,
        checkpoint="gated",
    )

    report = verify_controlled_difference(frozen, gated, ablation="frozen_prior")

    assert report["controlled_difference"] == "recurrent_belief_revision"


def test_rejects_hidden_selector_change() -> None:
    gated = _summary(
        belief_mode="recurrent",
        reliability_gate=True,
        checkpoint="gated",
    )
    ungated = _summary(
        belief_mode="recurrent",
        reliability_gate=False,
        checkpoint="ungated",
    )
    ungated = copy.deepcopy(ungated)
    ungated["protocol"]["learned_selectors"]["edit_utility"]["sha256"] = "other"

    with pytest.raises(ValueError, match="fixed protocol mismatch"):
        verify_controlled_difference(ungated, gated, ablation="ungated")
