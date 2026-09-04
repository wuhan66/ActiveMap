import pytest

from scripts.fit_active_catalog_terminal_arbiter import (
    apply_gate,
    fit_gate,
    observable_features,
)


def row(sample_id, prediction, target, confidence):
    false_edit = target == "KEEP" and prediction != "KEEP"
    missed_edit = target != "KEEP" and prediction == "KEEP"
    correct = prediction == target
    return {
        "sample_id": sample_id,
        "predicted_edit": prediction,
        "terminal_correct": correct,
        "false_edit": false_edit,
        "missed_edit": missed_edit,
        "wrong_edit": not correct and not false_edit and not missed_edit,
        "spent_cost": 0.0,
        "tool_calls": 0,
        "acquisitions": 0,
        "episode_utility_v2_proxy_balanced": float(correct) - 0.5 * false_edit,
        "episode_utility_v2_proxy_safety": float(correct) - false_edit,
        "events": [
            {
                "observable_state": {
                    "direct_draft": {"confidence": confidence},
                    "belief": {
                        "uncertainty": 0.2,
                        "edit_probabilities": [0.1, 0.2, 0.6, 0.1],
                    },
                }
            }
        ],
    }


def test_observable_features_include_probability_margin():
    features = observable_features(row("x", "DELETE", "DELETE", 0.8))
    assert features["belief_probability_margin"] == pytest.approx(0.4)
    assert features["delete_confidence"] == 0.8


def test_train_gate_can_accept_safe_candidate_and_reject_false_edit():
    pairs = [
        (row("a", "KEEP", "DELETE", 0.4), row("a", "DELETE", "DELETE", 0.9)),
        (row("b", "KEEP", "KEEP", 0.4), row("b", "DELETE", "KEEP", 0.3)),
    ]
    gate, metrics = fit_gate(pairs)
    selected, disagreements, accepted = apply_gate(
        pairs,
        feature=gate["feature"],
        direction=gate["direction"],
        threshold=gate["threshold"],
    )
    assert disagreements == 2
    assert accepted == 1
    assert metrics["terminal_accuracy"] == 1.0
    assert metrics["false_edit_rate"] == 0.0
    assert [item["predicted_edit"] for item in selected] == ["DELETE", "KEEP"]


def test_agreement_keeps_candidate_trajectory():
    baseline = row("a", "ADD", "ADD", 0.8)
    baseline["spent_cost"] = 1.0
    candidate = row("a", "ADD", "ADD", 0.8)
    selected, disagreements, accepted = apply_gate(
        [(baseline, candidate)],
        feature="direct_confidence",
        direction="ge",
        threshold=2.0,
    )
    assert disagreements == 0
    assert accepted == 0
    assert selected[0]["spent_cost"] == 0.0
