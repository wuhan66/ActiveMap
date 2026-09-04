import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.train_tool_belief_spatial_decision_head import (  # noqa: E402
    SpatialDecisionDataset,
    SpatialHierarchicalDecisionHead,
    causal_spatial_stack,
)


def test_causal_spatial_stack_zeros_future_evidence() -> None:
    artifacts = [torch.full((8, 8), float(index + 1)) for index in range(3)]
    stack = causal_spatial_stack(artifacts, 1, spatial_size=8)
    assert torch.equal(stack[0], artifacts[0])
    assert torch.count_nonzero(stack[1:]) == 0


def test_spatial_decision_head_shapes_and_ablation() -> None:
    model = SpatialHierarchicalDecisionHead(
        hidden_dim=32, spatial_base_channels=8, dropout=0.0
    )
    update, operation = model(torch.rand(2, 24), torch.rand(2, 3, 64, 64))
    assert update.shape == (2,)
    assert operation.shape == (2, 3)

    no_spatial = SpatialHierarchicalDecisionHead(
        hidden_dim=32, spatial_base_channels=8, dropout=0.0, use_spatial=False
    )
    no_spatial.eval()
    belief = torch.rand(2, 24)
    first = no_spatial(belief, torch.zeros(2, 3, 64, 64))
    second = no_spatial(belief, torch.ones(2, 3, 64, 64))
    assert torch.allclose(first[0], second[0])
    assert torch.allclose(first[1], second[1])


def test_gated_residual_starts_from_spatial_invariant_baseline() -> None:
    model = SpatialHierarchicalDecisionHead(
        hidden_dim=32,
        spatial_base_channels=8,
        dropout=0.0,
        fusion_mode="gated_residual",
    )
    model.eval()
    belief = torch.rand(2, 24)
    first = model(belief, torch.zeros(2, 3, 64, 64))
    second = model(belief, torch.ones(2, 3, 64, 64))
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert torch.count_nonzero(model.spatial_residual.weight) == 0


def test_gated_residual_no_spatial_is_strictly_invariant() -> None:
    model = SpatialHierarchicalDecisionHead(
        hidden_dim=32,
        spatial_base_channels=8,
        dropout=0.0,
        use_spatial=False,
        fusion_mode="gated_residual",
    )
    with torch.no_grad():
        model.spatial_residual.weight.fill_(1.0)
        model.spatial_residual.bias.fill_(1.0)
    model.eval()
    belief = torch.rand(2, 24)
    first = model(belief, torch.zeros(2, 3, 64, 64))
    second = model(belief, torch.ones(2, 3, 64, 64))
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_spatial_dataset_maps_artifacts_and_preserves_causality(tmp_path: Path) -> None:
    details = tmp_path / "details.jsonl"
    records = tmp_path / "records.jsonl"
    detail_rows = []
    record_rows = []
    for step in range(1, 4):
        detail_rows.append(
            {
                "sequence_id": "sequence-1",
                "episode_id": "task-1",
                "evidence_id": f"evidence-{step}",
                "step": step,
                "target": "ADD",
                "baseline": "KEEP",
                "baseline_probabilities": [0.8, 0.1, 0.05, 0.05],
                "baseline_recommended_edit": None,
                "paired": "ADD",
                "paired_probabilities": [0.1, 0.7, 0.1, 0.1],
                "paired_confidence": 0.7,
            }
        )
        call_id = f"call-{step}"
        record_rows.append(
            {
                "episode_id": "task-1",
                "evidence_id": f"evidence-{step}",
                "tool_result": {"tool": "TEMPORAL_CHANGE", "call_id": call_id},
            }
        )
        np.save(
            tmp_path / f"{call_id}_change.npy",
            np.full((8, 8), step / 3, dtype=np.float32),
        )
    details.write_text(
        "".join(json.dumps(row) + "\n" for row in detail_rows), encoding="utf-8"
    )
    records.write_text(
        "".join(json.dumps(row) + "\n" for row in record_rows), encoding="utf-8"
    )
    dataset = SpatialDecisionDataset(
        details, records, tmp_path, "train", spatial_size=8
    )
    assert len(dataset) == 4
    stage_two = dataset[2][1]
    assert torch.allclose(stage_two[0], torch.full((8, 8), 1 / 3))
    assert torch.allclose(stage_two[1], torch.full((8, 8), 2 / 3))
    assert torch.count_nonzero(stage_two[2]) == 0
