from __future__ import annotations

import numpy as np

from activemap.features import EVIDENCE_DIM, HYPOTHESIS_DIM, STATE_DIM
from activemap.selector_records import SelectorSample
from scripts.build_sn7_v5_matched_rollouts import (
    EvidenceValueActionScorer,
    build_policy_rollout,
)


class FixedScorer:
    def __init__(self, scores: list[float]) -> None:
        self.scores = np.asarray(scores, dtype=np.float64)

    def action_scores(self, sample: SelectorSample) -> np.ndarray:
        assert sample.evidence_ids == ["candidate"]
        return self.scores


def sample() -> SelectorSample:
    return SelectorSample.model_validate(
        {
            "sample_id": "episode-a__b1p5__s0",
            "split": "val",
            "edit_type": "ADD",
            "hypothesis_features": [0.0] * HYPOTHESIS_DIM,
            "state_features": [0.0] * STATE_DIM,
            "evidence_ids": ["candidate"],
            "evidence_features": [[0.0] * EVIDENCE_DIM],
            "evidence_costs": [0.3],
            "false_edit_risks": [0.1],
            "oracle_utilities": [0.2],
            "metadata": {
                "source_episode": "episode-a",
                "aoi_id": "aoi-a",
                "budget": 1.5,
                "oracle_step": 0,
                "gt_edit": "ADD",
                "selected_evidence_ids": ["anchor"],
                "evidence_predictions": {
                    "anchor": {
                        "edit_probabilities": [0.1, 0.8, 0.05, 0.05],
                        "confidence": 0.6,
                        "geometry_delta": [0.0] * 8,
                    },
                    "candidate": {
                        "edit_probabilities": [0.1, 0.0, 0.9, 0.0],
                        "confidence": 1.0,
                        "geometry_delta": [0.0] * 8,
                    },
                },
                "executable_outcomes": {
                    "anchor": {"predicted_operation": "ADD"},
                    "candidate": {"predicted_operation": "DELETE"},
                },
            },
        }
    )


def test_direct_policy_keeps_the_registered_initial_evidence():
    rollout = build_policy_rollout(sample(), policy="direct", scorer=None)

    assert rollout["selected_evidence_ids"] == ["anchor"]
    assert rollout["selected_extra_evidence"] is False
    assert rollout["spent_cost"] == 0.0
    assert rollout["prediction"] == "COMMIT:ADD"
    assert rollout["target"] == "COMMIT:ADD"


def test_selected_policy_only_acquires_when_above_its_stop_score():
    rollout = build_policy_rollout(sample(), policy="selected", scorer=FixedScorer([2.0, 0.0]))

    assert rollout["selected_evidence_ids"] == ["anchor", "candidate"]
    assert rollout["selected_extra_evidence_id"] == "candidate"
    assert rollout["selected_extra_evidence"] is True
    assert rollout["spent_cost"] == 0.3
    assert rollout["prediction"] == "COMMIT:DELETE"
    assert rollout["source_selected_evidence_ids"] == ["anchor", "candidate"]
    assert rollout["writeback_evidence_mode"] == "all"


def test_counterfactual_aligned_policy_executes_only_the_acquired_candidate():
    rollout = build_policy_rollout(
        sample(),
        policy="selected",
        scorer=FixedScorer([2.0, 0.0]),
        writeback_evidence_mode="counterfactual_aligned",
    )

    assert rollout["schema_version"] == "sn7-v5-matched-rollout-v2"
    assert rollout["selected_evidence_ids"] == ["candidate"]
    assert rollout["source_selected_evidence_ids"] == ["anchor", "candidate"]
    assert rollout["writeback_evidence_mode"] == "counterfactual_aligned"
    assert rollout["prediction"] == "COMMIT:DELETE"
    assert rollout["source_fused_prediction"] == "COMMIT:DELETE"


def test_counterfactual_aligned_stop_retains_the_initial_evidence():
    rollout = build_policy_rollout(
        sample(),
        policy="selected",
        scorer=FixedScorer([0.0, 2.0]),
        writeback_evidence_mode="counterfactual_aligned",
    )

    assert rollout["selected_evidence_ids"] == ["anchor"]
    assert rollout["source_selected_evidence_ids"] == ["anchor"]
    assert rollout["prediction"] == "COMMIT:ADD"


def test_counterfactual_aligned_uses_registered_operation_selector_prediction():
    state = sample()
    state.metadata["executable_outcomes"]["candidate"]["predicted_operation"] = "KEEP"
    rollout = build_policy_rollout(
        state,
        policy="selected",
        scorer=FixedScorer([2.0, 0.0]),
        writeback_evidence_mode="counterfactual_aligned",
    )

    assert rollout["prediction"] == "REJECT"
    assert rollout["standalone_belief_prediction"] == "COMMIT:DELETE"
    assert rollout["counterfactual_action_override"] is True


def test_selected_policy_respects_stop_and_falls_back_to_direct_evidence():
    rollout = build_policy_rollout(sample(), policy="selected", scorer=FixedScorer([0.0, 2.0]))

    assert rollout["selected_evidence_ids"] == ["anchor"]
    assert rollout["selected_extra_evidence"] is False
    assert rollout["spent_cost"] == 0.0


def test_forced_policy_acquires_the_same_top_candidate_even_when_stop_wins():
    rollout = build_policy_rollout(sample(), policy="forced", scorer=FixedScorer([0.0, 2.0]))

    assert rollout["selected_evidence_ids"] == ["anchor", "candidate"]
    assert rollout["selected_extra_evidence"] is True
    assert rollout["spent_cost"] == 0.3


def test_evidence_value_adapter_preserves_candidate_and_stop_scores():
    scorer = EvidenceValueActionScorer(
        lambda state: np.asarray([0.25, -0.05], dtype=np.float32)
    )

    scores = scorer.action_scores(sample())

    assert np.allclose(scores, [0.25, -0.05])
