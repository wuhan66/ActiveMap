from __future__ import annotations

from activemap.features import EVIDENCE_DIM, HYPOTHESIS_DIM, STATE_DIM
from activemap.selector_records import SelectorSample
from scripts.convert_sn7_rollouts_counterfactual_aligned import align_rollout


def sample() -> SelectorSample:
    return SelectorSample.model_validate(
        {
            "sample_id": "sample",
            "split": "val",
            "edit_type": "RESHAPE",
            "hypothesis_features": [0.0] * HYPOTHESIS_DIM,
            "state_features": [0.0] * STATE_DIM,
            "evidence_ids": ["candidate"],
            "evidence_features": [[0.0] * EVIDENCE_DIM],
            "evidence_costs": [0.2],
            "false_edit_risks": [0.1],
            "oracle_utilities": [0.4],
            "metadata": {
                "source_episode": "episode",
                "budget": 3.0,
                "oracle_step": 0,
                "gt_edit": "RESHAPE",
                "initial_evidence_id": "initial",
                "operation_update_threshold": 0.5,
                "evidence_predictions": {
                    "initial": {
                        "edit_probabilities": [0.8, 0.1, 0.05, 0.05],
                        "confidence": 0.8,
                        "geometry_delta": [0.0] * 8,
                    },
                    "candidate": {
                        "edit_probabilities": [0.1, 0.1, 0.1, 0.7],
                        "confidence": 0.9,
                        "geometry_delta": [0.0] * 8,
                    },
                },
                "executable_outcomes": {
                    "initial": {"predicted_operation": "KEEP"},
                    "candidate": {"predicted_operation": "RESHAPE"},
                },
            },
        }
    )


def rollout(*, acquired: bool = True) -> dict:
    return {
        "schema_version": "source",
        "source_episode": "episode",
        "split": "val",
        "budget": 3.0,
        "target": "COMMIT:RESHAPE",
        "prediction": "REJECT",
        "selected_evidence_ids": ["initial", "candidate"] if acquired else ["initial"],
        "initial_evidence_id": "initial",
        "selected_extra_evidence_id": "candidate" if acquired else None,
        "selected_extra_evidence": acquired,
        "test_assets_read": False,
    }


def test_alignment_executes_the_acquired_candidate_and_registered_action() -> None:
    row = align_rollout(sample(), rollout())

    assert row["selected_evidence_ids"] == ["candidate"]
    assert row["source_selected_evidence_ids"] == ["initial", "candidate"]
    assert row["prediction"] == "COMMIT:RESHAPE"
    assert row["writeback_evidence_mode"] == "counterfactual_aligned"


def test_alignment_retains_initial_evidence_without_acquisition() -> None:
    row = align_rollout(sample(), rollout(acquired=False))

    assert row["selected_evidence_ids"] == ["initial"]
    assert row["prediction"] == "REJECT"
