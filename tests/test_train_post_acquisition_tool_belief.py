import torch

from activemap.agent.tool_belief_model import (
    BELIEF_FEATURE_DIM,
    ToolPairBeliefResidualNetwork,
    apply_tool_belief_residual,
)
from activemap.agent.tool_features import TOOL_RESULT_FEATURE_DIM
from scripts.train_post_acquisition_tool_belief import (
    LossWeights,
    _threshold_predictions,
    post_acquisition_losses,
    promotion_gate,
)


def _batch():
    belief = torch.tensor(
        [[0.7, 0.1, 0.1, 0.1, 0.7, 0.5, *([0.0] * 8)]], dtype=torch.float32
    )
    quality = torch.ones((1, TOOL_RESULT_FEATURE_DIM), dtype=torch.float32)
    temporal = torch.full((1, TOOL_RESULT_FEATURE_DIM), 0.5, dtype=torch.float32)
    target_p = torch.tensor([[0.05, 0.85, 0.05, 0.05]], dtype=torch.float32)
    target_c = torch.tensor([0.85], dtype=torch.float32)
    target_g = torch.zeros((1, 8), dtype=torch.float32)
    gt = torch.tensor([1], dtype=torch.long)
    return belief, quality, temporal, target_p, target_c, target_g, gt


def test_post_acquisition_loss_supports_tool_ablation():
    model = ToolPairBeliefResidualNetwork(hidden_dim=16, dropout=0.0)
    class_weights = torch.ones(4)
    full_loss, components, full = post_acquisition_losses(
        _batch(),
        model,
        class_weights,
        max_logit_delta=1.5,
        geometry_scale=0.15,
        weights=LossWeights(),
    )
    zero_loss, _, zero = post_acquisition_losses(
        _batch(),
        model,
        class_weights,
        max_logit_delta=1.5,
        geometry_scale=0.15,
        weights=LossWeights(),
        zero_tool_features=True,
    )
    assert model.backbone[0].in_features == BELIEF_FEATURE_DIM + 2 * TOOL_RESULT_FEATURE_DIM
    assert torch.isfinite(full_loss)
    assert torch.isfinite(zero_loss)
    assert not torch.allclose(full[0], zero[0])
    assert components["tool_contrastive"] >= 0.0


def test_post_acquisition_loss_exposes_reliability_for_audit() -> None:
    model = ToolPairBeliefResidualNetwork(
        hidden_dim=16, dropout=0.0, reliability_gate=True, gate_bias=-1.5
    )
    _, _, prediction = post_acquisition_losses(
        _batch(),
        model,
        torch.ones(4),
        max_logit_delta=1.5,
        geometry_scale=0.15,
        weights=LossWeights(),
    )

    assert len(prediction) == 4
    assert torch.all((prediction[3] >= 0.0) & (prediction[3] <= 1.0))


def test_promotion_gate_requires_safety_quality_and_tool_dependence():
    metrics = {
        "baseline_argmax_macro_f1": 0.60,
        "baseline_argmax_false_edit_rate": 0.02,
        "baseline_deployed_macro_f1": 0.61,
        "baseline_deployed_false_edit_rate": 0.015,
        "zero_tool_macro_f1": 0.61,
        "full_macro_f1": 0.64,
        "full_false_edit_rate": 0.02,
        "baseline_deployed_expected_calibration_error": 0.08,
        "full_expected_calibration_error": 0.10,
    }
    assert promotion_gate(
        metrics, safety_margin=0.01, min_quality_delta=0.02, min_tool_delta=0.02
    )["passed"]
    metrics["zero_tool_macro_f1"] = 0.635
    assert not promotion_gate(
        metrics, safety_margin=0.01, min_quality_delta=0.02, min_tool_delta=0.02
    )["passed"]


def test_threshold_predictions_suppresses_unsafe_updates():
    probabilities = torch.tensor(
        [[0.40, 0.50, 0.05, 0.05], [0.20, 0.70, 0.05, 0.05]]
    )
    prediction = _threshold_predictions(probabilities, torch.tensor([0.69, 0.69]))
    assert prediction.tolist() == [0, 1]


def test_reliability_gate_bounds_evidence_effect() -> None:
    belief = _batch()[0]
    edit = torch.ones((1, 4))
    confidence = torch.ones(1)
    geometry = torch.ones((1, 8))
    ungated = apply_tool_belief_residual(
        belief, (edit, confidence, geometry), max_logit_delta=1.5, geometry_scale=0.15
    )
    gated = apply_tool_belief_residual(
        belief,
        (edit, confidence, geometry, torch.tensor([0.0])),
        max_logit_delta=1.5,
        geometry_scale=0.15,
    )

    assert torch.allclose(gated[0], belief[:, :4])
    assert torch.allclose(gated[1], belief[:, 4])
    assert torch.allclose(gated[2], belief[:, 6:14])
    assert not torch.allclose(ungated[1], gated[1])
