# ruff: noqa: E402

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from scripts.train_tool_belief_decision_head import (
    FEATURE_DIM,
    DecisionTrajectoryDataset,
    HierarchicalDecisionHead,
    _calibrate,
)


def _rows(split: str) -> list[dict[str, object]]:
    rows = []
    examples = (
        ("keep", "KEEP", "KEEP", [0.8, 0.1, 0.05, 0.05]),
        ("add", "ADD", "KEEP", [0.3, 0.6, 0.05, 0.05]),
    )
    for sequence_id, target, baseline, probabilities in examples:
        for step in range(1, 4):
            rows.append(
                {
                    "sequence_id": sequence_id,
                    "split": split,
                    "step": step,
                    "target": target,
                    "baseline": baseline,
                    "baseline_probabilities": [0.7, 0.1, 0.1, 0.1],
                    "baseline_recommended_edit": baseline,
                    "paired": max(
                        zip(("KEEP", "ADD", "DELETE", "RESHAPE"), probabilities, strict=False),
                        key=lambda item: item[1],
                    )[0],
                    "paired_probabilities": probabilities,
                    "paired_confidence": 0.9,
                }
            )
    return rows


def test_decision_dataset_and_network_contract(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in _rows("train")), encoding="utf-8"
    )
    dataset = DecisionTrajectoryDataset(path, "train")
    model = HierarchicalDecisionHead(hidden_dim=16, dropout=0.0)

    update_logit, operation_logit = model(dataset.features)

    assert dataset.features.shape == (2, FEATURE_DIM)
    assert update_logit.shape == (2,)
    assert operation_logit.shape == (2, 3)


def test_prefix_dataset_masks_unobserved_future_steps(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in _rows("train")), encoding="utf-8"
    )
    dataset = DecisionTrajectoryDataset(path, "train", stages=(0, 1, 2, 3))

    assert dataset.features.shape == (8, FEATURE_DIM)
    assert dataset.stages.tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
    assert torch.count_nonzero(dataset.features[0, 9:]) == 0
    assert torch.count_nonzero(dataset.features[1, 9:14]) > 0
    assert torch.count_nonzero(dataset.features[1, 14:]) == 0
    assert torch.count_nonzero(dataset.features[3, 9:]) > 0


def test_hierarchical_threshold_calibration_obeys_false_edit_constraint() -> None:
    result = _calibrate(
        target=torch.tensor([0, 0, 1, 1]).numpy(),
        baseline=torch.tensor([0, 0, 0, 0]).numpy(),
        update_probability=torch.tensor([0.1, 0.2, 0.7, 0.8]).numpy(),
        update_operation=torch.tensor([1, 1, 1, 1]).numpy(),
        safety_margin=0.0,
    )

    assert result is not None
    assert result["metrics"]["accuracy"] == 1.0
    assert result["metrics"]["false_edit_rate"] == 0.0
