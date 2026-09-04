from __future__ import annotations

import pytest

from scripts.evaluate_terminal_only_safe_commit_crossfit import (
    crossfit_seed,
    summarize,
)


def _row(sample: int, aoi: int, target: str, uncertainty: float) -> dict:
    predicted = "ADD"
    false_edit = target == "KEEP"
    return {
        "sample_id": f"s{sample}",
        "aoi_id": f"aoi{aoi}",
        "target_edit": target,
        "predicted_edit": predicted,
        "terminal_correct": target == predicted,
        "false_edit": false_edit,
        "missed_edit": False,
        "wrong_edit": False,
        "spent_cost": 0.2,
        "budget": 1.0,
        "tool_calls": 1,
        "acquisitions": 1,
        "quality_gain": 0.1,
        "quality_cost_utility": 0.05,
        "episode_utility_v2_proxy_balanced": -1.0 if false_edit else 0.8,
        "episode_utility_v2_proxy_safety": -1.0 if false_edit else 0.8,
        "episode_utility_v2_proxy_cost_aware": -1.0 if false_edit else 0.8,
        "events": [
            {
                "observable_state": {
                    "direct_draft": {"confidence": 1.0 - uncertainty},
                    "belief": {
                        "uncertainty": uncertainty,
                        "edit_probabilities": [0.05, 0.8, 0.1, 0.05],
                    },
                }
            }
        ],
    }


@pytest.mark.parametrize("gate_model", ["stump", "logistic"])
def test_aoi_crossfit_changes_only_terminal_decision(gate_model: str) -> None:
    candidate = {}
    reference = {}
    sample = 0
    for aoi in range(3):
        for target, uncertainty in (("ADD", 0.1), ("KEEP", 0.9)):
            row = _row(sample, aoi, target, uncertainty)
            candidate[row["sample_id"]] = row
            base = dict(row)
            base.update(
                predicted_edit="KEEP",
                terminal_correct=target == "KEEP",
                false_edit=False,
                missed_edit=target != "KEEP",
            )
            reference[row["sample_id"]] = base
            sample += 1

    safe_rows, gates = crossfit_seed(candidate, reference, gate_model=gate_model)
    metrics = summarize(safe_rows)

    assert metrics["terminal_accuracy"] == 1.0
    assert metrics["false_edit_rate"] == 0.0
    assert metrics["mean_cost"] == pytest.approx(0.2)
    assert all(row["tool_calls"] == 1 for row in safe_rows)
    assert all(row["acquisitions"] == 1 for row in safe_rows)
    assert gates["final_frozen_test_gate"]["fit_constraints_satisfied"] is True
    assert gates["final_frozen_test_gate"]["uses_ground_truth_at_inference"] is False
