from scripts.train_terminal_proposal_critic import (
    ACTIONS,
    action_index,
    action_score,
    encode,
)


def test_action_encoding() -> None:
    assert action_index("REJECT") == 0
    assert action_index("COMMIT:DELETE") == 2
    features = encode([0.25, 0.5], 0, 1)
    assert len(features) == 2 + 2 * len(ACTIONS)
    assert features[2:6] == [1.0, 0.0, 0.0, 0.0]
    assert features[6:10] == [0.0, 1.0, 0.0, 0.0]


def test_safety_weighted_action_score() -> None:
    assert action_score(1, 1) == 1.0
    assert action_score(1, 0) < action_score(0, 1)
