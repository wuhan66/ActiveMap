from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from activemap.evaluation.operation_selector import (
    calibrate_operation_predictions,
    export_updater_operation_baseline,
    gated_operation_predictions,
)
from activemap.inference import OperationSelectorPredictor
from activemap.nn.operation_selector import (
    OperationSelector,
    OperationSelectorConfig,
    operation_selector_inputs,
)
from activemap.nn.updater import PriorConditionedUNet, UpdaterConfig
from activemap.training.operation_selector import (
    effective_class_weights,
    focal_cross_entropy,
    operation_metrics,
)


def test_operation_selector_uses_spatial_arrangement() -> None:
    torch.manual_seed(7)
    model = OperationSelector(
        OperationSelectorConfig(
            context_dim=12,
            spatial_size=32,
            base_channels=8,
            hidden_dim=16,
            dropout=0.0,
        )
    ).eval()
    evidence = torch.zeros(2, 3, 32, 32)
    evidence[:, 2, 8:24, 8:24] = 1.0
    evidence[0, 0, 2:10, 2:10] = 1.0
    evidence[1, 0, 22:30, 22:30] = 1.0
    context = torch.zeros(2, 12)
    logits = model(evidence, context)
    assert logits.shape == (2, 4)
    assert not torch.allclose(logits[0], logits[1])

    no_spatial = OperationSelector(
        OperationSelectorConfig(
            context_dim=12,
            spatial_size=32,
            base_channels=8,
            hidden_dim=16,
            dropout=0.0,
            use_spatial=False,
        )
    ).eval()
    ablated = no_spatial(evidence, context)
    assert torch.allclose(ablated[0], ablated[1])


def test_operation_selector_inputs_use_predictions_without_target_mask() -> None:
    outputs = {
        "temporal_change_logits": torch.randn(2, 2, 16, 16),
        "shared_features": torch.randn(2, 8),
        "change_descriptor": torch.randn(2, 12),
        "geometry_delta": torch.randn(2, 8),
        "confidence_logits": torch.randn(2),
        "edit_logits": torch.randn(2, 4),
    }
    spatial, context = operation_selector_inputs(outputs, torch.zeros(2, 1, 16, 16), spatial_size=8)
    assert spatial.shape == (2, 3, 8, 8)
    assert context.shape == (2, 8 + 12 + 8 + 1 + 4 + 6)
    assert torch.all((0.0 <= spatial) & (spatial <= 1.0))


def test_effective_weights_raise_for_missing_class_and_upweight_rare_class() -> None:
    targets = torch.tensor([0] * 10 + [1] * 5 + [2] * 2 + [3])
    weights = effective_class_weights(targets, classes=4, beta=0.99)
    assert weights.shape == (4,)
    assert weights[3] > weights[2] > weights[1] > weights[0]
    assert torch.isclose(weights.mean(), torch.tensor(1.0))
    with pytest.raises(ValueError, match="missing operation classes"):
        effective_class_weights(torch.tensor([0, 1, 2]), classes=4, beta=0.99)


def test_operation_loss_and_metrics_reward_all_four_classes() -> None:
    targets = torch.tensor([0, 1, 2, 3])
    logits = torch.eye(4) * 8.0
    loss = focal_cross_entropy(
        logits,
        targets,
        class_weights=torch.ones(4),
        gamma=1.5,
        label_smoothing=0.0,
    )
    metrics = operation_metrics(logits, targets)
    assert loss < 1e-6
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["f1_reshape"] == pytest.approx(1.0)
    assert metrics["false_edit_rate"] == pytest.approx(0.0)
    assert metrics["missed_update_rate"] == pytest.approx(0.0)


def test_operation_gate_and_calibration_enforce_false_edit_limit(tmp_path) -> None:
    probabilities = torch.tensor(
        [
            [0.90, 0.05, 0.03, 0.02],
            [0.40, 0.50, 0.05, 0.05],
            [0.20, 0.70, 0.05, 0.05],
            [0.15, 0.10, 0.70, 0.05],
        ]
    )
    assert gated_operation_predictions(probabilities, 0.5).tolist() == [0, 1, 1, 2]
    records = []
    names = ["KEEP", "ADD", "DELETE", "RESHAPE"]
    targets = ["KEEP", "KEEP", "ADD", "DELETE"]
    for index, target in enumerate(targets):
        records.append(
            {
                "sample_id": str(index),
                "target": target,
                "probabilities": {
                    name: float(probabilities[index, class_index])
                    for class_index, name in enumerate(names)
                },
            }
        )
    prediction_path = tmp_path / "predictions.jsonl"
    prediction_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    summary = calibrate_operation_predictions(
        prediction_path, tmp_path / "calibration.json", max_false_edit=0.0
    )
    assert summary["constraint_satisfied"] is True
    assert summary["selected"]["false_edit_rate"] == 0.0
    assert summary["test_evaluation"] is None


def test_frozen_updater_baseline_uses_cached_edit_logits(tmp_path) -> None:
    base_channels = 4
    context = torch.zeros(2, base_channels * 12 + 19)
    edit_start = base_channels * 12 + 9
    context[0, edit_start : edit_start + 4] = torch.tensor([5.0, 0.0, 0.0, 0.0])
    context[1, edit_start : edit_start + 4] = torch.tensor([0.0, 5.0, 0.0, 0.0])
    cache = {
        "updater_checkpoint": "updater.pt",
        "updater_model_config": {"base_channels": base_channels},
        "splits": {
            "val": {
                "spatial": torch.zeros(2, 3, 8, 8),
                "context": context,
                "targets": torch.tensor([0, 1]),
                "sample_ids": ["keep", "add"],
            }
        },
    }
    cache_path = tmp_path / "cache.pt"
    torch.save(cache, cache_path)
    summary = export_updater_operation_baseline(cache_path, tmp_path / "baseline")
    assert summary["metrics"]["macro_f1"] == pytest.approx(0.5)
    assert summary["metrics"]["accuracy"] == pytest.approx(1.0)
    assert summary["test_evaluation"] is None


def test_combined_operation_predictor_emits_agent_belief(tmp_path) -> None:
    updater_config = UpdaterConfig(
        base_channels=4,
        dropout=0.0,
        hierarchical_edit=True,
        vector_change_encoder=True,
        vector_change_to_edit_head=True,
        temporal_change_head=True,
    )
    updater = PriorConditionedUNet(updater_config)
    updater_path = tmp_path / "updater.pt"
    torch.save(
        {"state_dict": updater.state_dict(), "model_config": updater_config.as_dict()},
        updater_path,
    )
    selector_config = OperationSelectorConfig(
        context_dim=4 * 12 + 19,
        spatial_size=32,
        base_channels=8,
        hidden_dim=16,
        dropout=0.0,
    )
    selector = OperationSelector(selector_config)
    selector_path = tmp_path / "selector.pt"
    torch.save(
        {
            "state_dict": selector.state_dict(),
            "model_config": selector_config.as_dict(),
            "updater_checkpoint": str(updater_path),
        },
        selector_path,
    )
    predictor = OperationSelectorPredictor(selector_path, update_threshold=0.6, device="cpu")
    prediction = predictor.predict(
        np.zeros((64, 64, 3), dtype=np.uint8),
        np.zeros((64, 64), dtype=np.float32),
    )
    assert np.asarray(prediction["edit_probabilities"]).shape == (4,)
    assert np.asarray(prediction["mask_probability"]).shape == (64, 64)
    assert prediction["gated_edit"] in {"KEEP", "ADD", "DELETE", "RESHAPE"}
