import numpy as np
import torch

from scripts.evaluate_tool_belief_spatial_decision_head import (
    _model_from_checkpoint,
    _stage_metrics,
)
from scripts.train_tool_belief_spatial_decision_head import SpatialHierarchicalDecisionHead


def test_stage_metrics_keeps_causal_stages_separate() -> None:
    values = {
        "target": np.asarray([0, 1, 0, 1]),
        "prediction": np.asarray([0, 1, 1, 1]),
        "update_probability": np.asarray([0.1, 0.9, 0.8, 0.9]),
        "stage": np.asarray([0, 0, 1, 1]),
    }
    metrics = _stage_metrics(values, "test")
    assert set(metrics) == {"0", "1"}
    assert metrics["0"]["accuracy"] == 1.0
    assert metrics["1"]["accuracy"] == 0.5


def test_checkpoint_loader_preserves_no_spatial_ablation() -> None:
    source = SpatialHierarchicalDecisionHead(
        hidden_dim=16, spatial_base_channels=8, dropout=0.0, use_spatial=False
    )
    checkpoint = {
        "config": {
            "hidden_dim": 16,
            "spatial_base_channels": 8,
            "dropout": 0.0,
            "use_spatial": False,
        },
        "model_state_dict": source.state_dict(),
    }
    model, use_spatial = _model_from_checkpoint(checkpoint, torch.device("cpu"))
    model.eval()
    belief = torch.randn(3, 24)
    first_spatial = torch.randn(3, 3, 64, 64)
    second_spatial = torch.randn(3, 3, 64, 64)
    with torch.inference_mode():
        first = model(belief, first_spatial)
        second = model(belief, second_spatial)
    assert use_spatial is False
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_checkpoint_loader_restores_gated_residual_fusion() -> None:
    source = SpatialHierarchicalDecisionHead(
        hidden_dim=16,
        spatial_base_channels=8,
        dropout=0.0,
        fusion_mode="gated_residual",
        gate_bias=-1.5,
    )
    checkpoint = {
        "config": {
            "hidden_dim": 16,
            "spatial_base_channels": 8,
            "dropout": 0.0,
            "use_spatial": True,
            "fusion_mode": "gated_residual",
            "gate_bias": -1.5,
        },
        "model_state_dict": source.state_dict(),
    }
    model, use_spatial = _model_from_checkpoint(checkpoint, torch.device("cpu"))
    assert use_spatial is True
    assert model.fusion_mode == "gated_residual"
    assert torch.equal(model.spatial_gate.bias, source.spatial_gate.bias)
