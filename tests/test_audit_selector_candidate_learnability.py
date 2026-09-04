import numpy as np
import pytest

from scripts.audit_selector_candidate_learnability import candidate_features


def _row():
    return {
        "edit_type": "ADD",
        "hypothesis_features": [0.1, 0.7, 0.1, 0.1] + [0.0] * 9 + [0.8, 0.0, 0.0],
        "state_features": [0.0] * 8,
        "evidence_ids": ["candidate"],
        "evidence_features": [[0.0] * 13],
        "evidence_costs": [1.0],
        "false_edit_risks": [0.2],
        "metadata": {
            "evidence_predictions": {
                "candidate": {
                    "edit_probabilities": [0.6, 0.2, 0.1, 0.1],
                    "gated_edit": "KEEP",
                    "confidence": 0.9,
                    "geometry_delta": [0.0] * 8,
                    "mask_features": [float(index) / 10.0 for index in range(10)],
                }
            },
            "mask_feature_contract": {
                "schema_version": "target-free-mask-features-v2",
                "feature_names": [f"feature-{index}" for index in range(10)],
                "target_free": True,
                "scope": "candidate_local_grid",
            },
        },
    }


def test_policy_relative_candidate_features_are_target_free_and_invariant():
    features = candidate_features(_row(), "policy-relative")
    assert features.shape == (1, 70)
    assert np.isfinite(features).all()
    assert features[0, 28:32].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert features[0, 32:36].tolist() == pytest.approx([0.5, -0.5, 0.0, 0.0])
    assert features[0, 43] == pytest.approx(0.1)
    assert features[0, 44] == 0.0
    assert features[0, 45] == pytest.approx(1.0)


def test_policy_relative_mask_features_append_target_free_spatial_summary():
    features = candidate_features(_row(), "policy-relative-mask")
    assert features.shape == (1, 80)
    assert features[0, 46:56].tolist() == pytest.approx(
        [float(index) / 10.0 for index in range(10)]
    )
    assert np.isfinite(features).all()


def test_policy_relative_mask_features_reject_unproven_provenance():
    row = _row()
    row["metadata"]["mask_feature_contract"]["target_free"] = False
    with pytest.raises(ValueError, match="target-free provenance"):
        candidate_features(row, "policy-relative-mask")
