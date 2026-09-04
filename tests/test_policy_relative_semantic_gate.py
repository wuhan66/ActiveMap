import numpy as np

from activemap.agent.active_catalog_tool_gate import (
    LinearProbabilityGate,
    policy_relative_semantic_gate_features,
)
from activemap.models import EditOperation
from activemap.selector_records import SelectorSample
from scripts.train_policy_relative_semantic_gate_multibackend import (
    consensus_beneficial_labels,
)


def test_observable_gate_features_are_fixed_and_finite():
    sample = SelectorSample(
        sample_id="sample",
        split="train",
        edit_type=EditOperation.KEEP,
        hypothesis_features=[0.0] * 16,
        state_features=[0.0] * 8,
        evidence_ids=["evidence"],
        evidence_features=[[0.1] * 13],
        evidence_costs=[1.0],
        false_edit_risks=[0.0],
        oracle_utilities=[0.0],
        stop_utility=0.0,
        false_edit_penalty_weight=0.35,
        metadata={},
    )
    features = policy_relative_semantic_gate_features(sample)
    assert features.shape == (53,)
    assert np.all(np.isfinite(features))


def test_linear_probability_gate_matches_manual_sigmoid():
    gate = LinearProbabilityGate(
        mean=[0.0] * 53,
        scale=[2.0] * 53,
        coefficient=[1.0] * 53,
        intercept=-1.0,
    )
    probability = gate.predict_proba(np.ones((1, 53)))[0, 1]
    expected = 1.0 / (1.0 + np.exp(-(53.0 / 2.0 - 1.0)))
    assert np.isclose(probability, expected)


def test_consensus_beneficial_labels_require_votes_and_positive_mean_gain():
    direct = np.array([0.5, 0.5, 0.5])
    semantic = np.array(
        [
            [0.8, 0.8, 0.8],
            [0.7, 0.7, 0.1],
            [0.4, 0.1, 0.1],
        ]
    )
    costs = np.full_like(semantic, 0.1)

    labels = consensus_beneficial_labels(
        direct,
        semantic,
        costs,
        minimum_votes=2,
    )

    assert labels.tolist() == [1, 0, 0]
