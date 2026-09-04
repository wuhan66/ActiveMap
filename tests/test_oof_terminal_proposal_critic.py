from scripts.train_oof_terminal_proposal_critic import (
    belief_features,
    terminal_operation,
)


def test_belief_feature_contract() -> None:
    result = belief_features(
        {
            "edit_probabilities": [0.1, 0.2, 0.3, 0.4],
            "confidence": 0.6,
            "geometry_delta": [0.0] * 8,
            "uncertainty": 0.8,
        }
    )
    assert len(result) == 14
    assert result[:4] == [0.1, 0.2, 0.3, 0.4]


def test_terminal_operation() -> None:
    assert terminal_operation({"action": "REJECT"}) == "KEEP"
    assert terminal_operation({"action": "COMMIT", "edit": "ADD"}) == "ADD"
    assert terminal_operation({"action": "ACQUIRE"}) is None
