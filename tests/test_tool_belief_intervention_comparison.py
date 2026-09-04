from pathlib import Path

import pytest

from scripts.compare_tool_belief_interventions import compare


def _report(*, macro_f1: float, utility: float, passed: bool) -> dict:
    identity = {
        "accuracy": 0.5,
        "macro_f1": 0.4,
        "false_edit_rate": 0.1,
        "missed_edit_rate": 0.2,
    }
    learned = {
        "accuracy": 0.7,
        "macro_f1": macro_f1,
        "false_edit_rate": 0.08,
        "missed_edit_rate": 0.15,
        "mean_joint_utility": utility,
    }
    recurrent = {"macro_f1": macro_f1 - 0.01, "mean_joint_utility": utility - 0.1}
    paired = {"macro_f1": macro_f1 - 0.02, "mean_joint_utility": utility - 0.2}
    return {
        "protocol": {
            "schema_version": "tool-belief-intervention-eval-v1",
            "split": "val",
            "episode_count": 138,
            "one_step_count": 414,
            "tools_per_episode": 6,
        },
        "summaries": {
            "identity_one_step": identity,
            "learned_one_step": learned,
            "learned_temporal_3": recurrent,
            "learned_quality_temporal_3": paired,
            "learned_quality_3": recurrent,
            "learned_no_current_one_step": learned,
        },
        "quality_noop_delta_max": {
            "probability_l1": 0.01,
            "confidence_absolute": 0.02,
            "geometry_mae": 0.01,
        },
        "gates": {"passed": passed, "checks": {"quality": passed}},
    }


def test_comparison_selects_only_passing_candidate_by_joint_utility() -> None:
    candidates = {
        "full": (Path("full.json"), _report(macro_f1=0.6, utility=0.4, passed=True)),
        "no_teacher": (
            Path("no_teacher.json"),
            _report(macro_f1=0.7, utility=0.5, passed=False),
        ),
        "no_operation": (
            Path("no_operation.json"),
            _report(macro_f1=0.58, utility=0.45, passed=True),
        ),
    }

    result = compare(candidates, reference="full")

    assert result["selection"]["selected"] == "no_operation"
    assert result["selection"]["eligible_count"] == 2
    assert result["deltas_from_reference"]["no_teacher"]["one_step_macro_f1"] == pytest.approx(
        0.1
    )


def test_comparison_rejects_protocol_mismatch() -> None:
    mismatched = _report(macro_f1=0.6, utility=0.4, passed=True)
    mismatched["protocol"]["episode_count"] = 137

    with pytest.raises(ValueError, match="protocol mismatch"):
        compare(
            {
                "full": (Path("full.json"), _report(macro_f1=0.6, utility=0.4, passed=True)),
                "bad": (Path("bad.json"), mismatched),
            },
            reference="full",
        )
