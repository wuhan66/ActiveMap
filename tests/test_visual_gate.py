import numpy as np
import pytest
import torch

from activemap.agent.visual_gate import (
    NumpyMLPGate,
    binary_call_metrics,
    pool_last_prompt_state,
    pool_prompt_state,
    unique_pre_tool_rows,
)


def test_unique_pre_tool_rows_deduplicates_balanced_training_records():
    rows = [
        {"stage": "PRE_TOOL", "example_id": "a", "oracle_use_tool": True},
        {"stage": "POST_TOOL", "example_id": "a", "oracle_use_tool": True},
        {"stage": "PRE_TOOL", "example_id": "a", "oracle_use_tool": True},
        {"stage": "PRE_TOOL", "example_id": "b", "oracle_use_tool": False},
    ]
    assert [row["example_id"] for row in unique_pre_tool_rows(rows)] == ["a", "b"]


def test_unique_pre_tool_rows_rejects_conflicting_duplicate_labels():
    rows = [
        {"stage": "PRE_TOOL", "example_id": "a", "oracle_use_tool": True},
        {"stage": "PRE_TOOL", "example_id": "a", "oracle_use_tool": False},
    ]
    with pytest.raises(ValueError, match="labels disagree"):
        unique_pre_tool_rows(rows)


def test_pool_last_prompt_state_respects_right_padding():
    hidden = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
    pooled = pool_last_prompt_state(hidden, mask)
    assert torch.equal(pooled[0], hidden[0, 1])
    assert torch.equal(pooled[1], hidden[1, 2])


def test_pool_prompt_state_concatenates_last_and_masked_mean():
    hidden = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    pooled = pool_prompt_state(hidden, mask, mode="last_mean")
    assert pooled.tolist() == [[3.0, 5.0, 2.0, 4.0]]
    assert torch.equal(pool_prompt_state(hidden, mask, mode="last"), hidden[:, 1])


def test_binary_call_metrics_reports_sparse_precision_and_recall():
    metrics = binary_call_metrics(
        np.asarray([1, 1, 0, 0]), np.asarray([0.9, 0.2, 0.8, 0.1]), 0.5
    )
    assert metrics["predicted_calls"] == 2
    assert metrics["true_calls"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f0_5"] == 0.5


def test_numpy_mlp_gate_exports_stable_probabilities():
    gate = NumpyMLPGate(
        mean=np.asarray([1.0, 1.0]),
        scale=np.asarray([1.0, 2.0]),
        weight1=np.asarray([[1.0], [1.0]]),
        bias1=np.asarray([0.0]),
        weight2=np.asarray([1.0]),
        bias2=0.0,
    )

    probabilities = gate.predict_proba(np.asarray([[1.0, 1.0], [2.0, 3.0]]))

    assert probabilities.shape == (2, 2)
    assert probabilities[0, 1] == pytest.approx(0.5)
    assert probabilities[1, 1] > probabilities[0, 1]
    assert np.allclose(probabilities.sum(axis=1), 1.0)
