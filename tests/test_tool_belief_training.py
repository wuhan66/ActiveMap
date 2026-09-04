import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from activemap.agent.records import AgentBelief
from activemap.agent.tool_belief_data import ToolBeliefExample
from activemap.geo_tools.records import GeoToolName, GeoToolResult
from activemap.models import EditOperation
from scripts.train_tool_belief import (
    LossWeights,
    ToolBeliefDataset,
    _classification_metrics,
    _expected_calibration_error,
    _losses,
)


def _belief(probabilities: list[float], confidence: float) -> AgentBelief:
    return AgentBelief(
        edit_probabilities=probabilities,
        confidence=confidence,
        uncertainty=0.5,
        geometry_delta=[0.0] * 8,
    )


class _GeometryResidual(nn.Module):
    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        count = inputs.shape[0]
        return (
            torch.zeros((count, 4)),
            torch.zeros(count),
            torch.ones((count, 8)),
        )


def test_training_uses_teacher_confidence_and_penalizes_noop_geometry(tmp_path: Path) -> None:
    prior = _belief([0.7, 0.1, 0.1, 0.1], 0.83)
    target = _belief([0.5, 0.3, 0.1, 0.1], 0.61)
    rows = [
        ToolBeliefExample(
            record_id="quality",
            episode_id="episode",
            split="train",
            evidence_id="evidence",
            prior_belief=prior,
            tool_result=GeoToolResult(
                call_id="quality",
                tool=GeoToolName.IMAGE_QUALITY,
                success=True,
                outputs={"sharpness": 0.2},
                cost=0.03,
            ),
            target_belief=prior,
            gt_edit=EditOperation.ADD,
            metadata={"target_kind": "observational_noop"},
        ),
        ToolBeliefExample(
            record_id="change",
            episode_id="episode",
            split="train",
            evidence_id="evidence",
            prior_belief=prior,
            tool_result=GeoToolResult(
                call_id="change",
                tool=GeoToolName.TEMPORAL_CHANGE,
                success=True,
                outputs={"changed_fraction": 0.4},
                cost=0.15,
            ),
            target_belief=target,
            gt_edit=EditOperation.ADD,
            metadata={"target_kind": "teacher_belief_update"},
        ),
    ]
    path = tmp_path / "train.jsonl"
    path.write_text(
        "".join(json.dumps(row.model_dump(mode="json")) + "\n" for row in rows),
        encoding="utf-8",
    )

    dataset = ToolBeliefDataset(path)
    assert dataset.target_confidence.tolist() == pytest.approx([0.83, 0.61])
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    total, components, _ = _losses(
        batch,
        _GeometryResidual(),
        torch.ones(4),
        max_logit_delta=3.0,
        geometry_scale=0.25,
        loss_weights=LossWeights(
            teacher_kl=0.0,
            operation=0.0,
            calibration=0.0,
            geometry=0.0,
            noop_probability_drift=0.0,
            noop_confidence_drift=0.0,
            noop_geometry_drift=1.0,
        ),
    )

    assert float(components["noop_probability_drift"]) == pytest.approx(0.0, abs=1e-6)
    assert float(components["noop_confidence_drift"]) == pytest.approx(0.0, abs=1e-6)
    assert float(components["noop_geometry_drift"]) > 0.0
    assert float(total) == pytest.approx(float(components["noop_geometry_drift"]))


def test_classwise_metrics_expose_collapsed_operation() -> None:
    target = torch.tensor([0, 1, 2, 3]).numpy()
    prediction = torch.tensor([0, 0, 0, 0]).numpy()

    metrics = _classification_metrics(target, prediction)

    assert metrics["keep_recall"] == 1.0
    assert metrics["add_recall"] == 0.0
    assert metrics["delete_recall"] == 0.0
    assert metrics["reshape_recall"] == 0.0
    assert _expected_calibration_error(
        torch.tensor([1.0, 1.0, 1.0, 1.0]).numpy(), prediction == target
    ) == pytest.approx(0.75)
