import numpy as np
import pytest

from scripts.train_tool_belief_stopping_policy import (
    POLICY_FEATURE_DIM,
    PolicyData,
    StoppingPolicy,
    calibrate_threshold,
    rollout_policy,
)

torch = pytest.importorskip("torch")


def _data() -> PolicyData:
    return PolicyData(
        sequence_ids=["one", "two"],
        features=np.zeros((2, 3, POLICY_FEATURE_DIM), dtype=np.float32),
        targets=np.asarray([1, 0]),
        predictions=np.asarray([[0, 1, 1, 1], [0, 1, 0, 0]]),
        confidences=np.full((2, 4), 0.8),
        costs=np.asarray([[0.0, 0.18, 0.36, 0.54], [0.0, 0.18, 0.36, 0.54]]),
        continue_targets=np.asarray([[1, 0, 0], [0, 0, 0]], dtype=np.float32),
        future_advantages=np.zeros((2, 3), dtype=np.float32),
    )


def test_stopping_policy_network_contract() -> None:
    model = StoppingPolicy(hidden_dim=16, dropout=0.0)
    logits, advantages = model(torch.zeros(5, POLICY_FEATURE_DIM))
    assert logits.shape == (5,)
    assert advantages.shape == (5,)


def test_rollout_and_train_only_threshold_calibration() -> None:
    data = _data()
    probabilities = np.asarray([[0.9, 0.1, 0.1], [0.1, 0.1, 0.1]])

    metrics, stages = rollout_policy(data, probabilities, 0.5, name="learned")
    threshold, calibrated = calibrate_threshold(data, probabilities, safety_margin=1.0)

    assert stages.tolist() == [1, 0]
    assert metrics["mean_joint_utility"] == pytest.approx(0.91)
    assert calibrated["mean_joint_utility"] == pytest.approx(0.91)
    assert 0.1 < threshold <= 0.9
