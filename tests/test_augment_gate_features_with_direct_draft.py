import numpy as np

from scripts.augment_gate_features_with_direct_draft import direct_draft_vector


def test_direct_draft_vector_encodes_observable_transition_without_target():
    vector = direct_draft_vector(
        {
            "direct_operation": "ADD",
            "belief_operation": "KEEP",
            "direct_terminal_valid": True,
            "tool_cost": 0.75,
        }
    )

    assert vector.shape == (29,)
    assert vector[:4].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert vector[4:8].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert vector[8 + 1] == 1.0
    assert vector[-5:].tolist() == [0.0, 1.0, 0.0, 1.0, 0.75]
    assert np.sum(vector[8:24]) == 1.0
