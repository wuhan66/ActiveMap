import hashlib

import numpy as np
import pytest

import activemap.integrations.prior_conditioned_sam_road as prior_module
from activemap.integrations.prior_conditioned_sam_road import (
    PriorConditionedChangeHead,
    PriorConditionedSAMRoadPredictor,
    change_targets,
    prior_conditioned_change_loss,
)
from activemap.models import EditOperation

torch = pytest.importorskip("torch")


def test_change_targets_separate_add_and_remove() -> None:
    prior = torch.tensor([[[1.0, 1.0], [0.0, 0.0]]])
    target = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])

    current, add, remove = change_targets(target, prior)

    assert torch.equal(current, target)
    assert torch.equal(add, torch.tensor([[[0.0, 0.0], [1.0, 0.0]]]))
    assert torch.equal(remove, torch.tensor([[[0.0, 1.0], [0.0, 0.0]]]))


def test_prior_conditioned_loss_is_finite_and_backpropagates() -> None:
    logits = torch.zeros((2, 3, 4, 4), requires_grad=True)
    operation_logits = torch.zeros((2, 4), requires_grad=True)
    prior = torch.zeros((2, 4, 4))
    target = prior.clone()
    target[1, 1:3, 1:3] = 1
    valid = torch.ones_like(target)
    loss, components = prior_conditioned_change_loss(
        logits,
        operation_logits,
        torch.tensor([0, 1]),
        target,
        prior,
        valid,
        positive_weights=(2.0, 4.0, 4.0),
        operation_weights=torch.ones(4),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert bool(torch.isfinite(logits.grad).all())
    assert operation_logits.grad is not None
    assert components["unchanged_safety"].item() > 0


def test_change_head_starts_from_source_road_and_low_independent_change() -> None:
    source_decoder = torch.nn.Conv2d(2, 2, kernel_size=1)
    head = PriorConditionedChangeHead.build(source_decoder)
    embedding = torch.randn(1, 2, 4, 4)
    prior = torch.zeros(1, 1, 4, 4)

    output = head(embedding, prior)
    source_road = source_decoder(embedding)[:, 1]

    assert torch.allclose(output["change_logits"][:, 0], source_road)
    assert float(torch.sigmoid(output["change_logits"][:, 1:]).max()) < 0.02
    assert output["operation_logits"].shape == (1, 4)


def test_current_difference_parameterization_is_available_for_ablation() -> None:
    source_decoder = torch.nn.Conv2d(2, 2, kernel_size=1)
    head = PriorConditionedChangeHead.build(
        source_decoder,
        parameterization="current_difference",
    )
    embedding = torch.randn(1, 2, 4, 4)
    prior = torch.zeros(1, 1, 4, 4)

    output = head(embedding, prior)
    source_road = source_decoder(embedding)[:, 1]

    assert torch.allclose(output["change_logits"][:, 1], source_road)
    assert torch.allclose(output["change_logits"][:, 2], -source_road - 8.0)


def test_spatial_pyramid_operation_head_preserves_output_contract() -> None:
    source_decoder = torch.nn.Conv2d(2, 2, kernel_size=1)
    head = PriorConditionedChangeHead.build(
        source_decoder,
        operation_head="spatial_pyramid",
    )

    output = head(torch.randn(2, 2, 8, 8), torch.zeros(2, 1, 8, 8))

    assert output["change_logits"].shape == (2, 3, 8, 8)
    assert output["operation_logits"].shape == (2, 4)
    assert sum(p.numel() for p in head.operation_classifier.parameters()) > 80_000


def test_positive_only_change_dice_ignores_empty_change_channels() -> None:
    logits = torch.zeros((2, 3, 4, 4), requires_grad=True)
    operation_logits = torch.zeros((2, 4), requires_grad=True)
    target = torch.zeros((2, 4, 4))
    prior = torch.zeros_like(target)
    valid = torch.ones_like(target)

    loss, components = prior_conditioned_change_loss(
        logits,
        operation_logits,
        torch.tensor([0, 0]),
        target,
        prior,
        valid,
        positive_weights=(2.0, 4.0, 4.0),
        operation_weights=torch.ones(4),
        positive_only_change_dice=True,
    )
    loss.backward()

    assert components["add_dice"].item() == 0.0
    assert components["remove_dice"].item() == 0.0
    assert bool(torch.isfinite(logits.grad).all())


def test_operation_decision_applies_calibrated_commit_gate() -> None:
    raw, gated = PriorConditionedSAMRoadPredictor.operation_decision(
        np.asarray([0.20, 0.55, 0.15, 0.10]), 0.60
    )
    assert raw == EditOperation.ADD
    assert gated == EditOperation.KEEP

    raw, gated = PriorConditionedSAMRoadPredictor.operation_decision(
        np.asarray([0.20, 0.65, 0.10, 0.05]), 0.60
    )
    assert raw == EditOperation.ADD
    assert gated == EditOperation.ADD


def test_operation_decision_rejects_invalid_probabilities() -> None:
    with pytest.raises(ValueError, match="four finite"):
        PriorConditionedSAMRoadPredictor.operation_decision(
            np.asarray([0.5, np.nan, 0.5]), 0.6
        )


def test_prior_conditioned_predictor_emits_explicit_change_maps(
    tmp_path, monkeypatch
) -> None:
    class FakeSAMModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.image_encoder = torch.nn.Conv2d(3, 2, kernel_size=1)
            self.map_decoder = torch.nn.Conv2d(2, 2, kernel_size=1)
            self.register_buffer("pixel_mean", torch.zeros((1, 3, 1, 1)))
            self.register_buffer("pixel_std", torch.ones((1, 3, 1, 1)))
            with torch.no_grad():
                self.image_encoder.weight.zero_()
                self.image_encoder.bias.zero_()
                self.map_decoder.weight.zero_()
                self.map_decoder.bias.zero_()

    class FakeSAMRoadPredictor:
        def __init__(self, *args, device="cpu", upstream_commit=None, **kwargs):
            self._torch = torch
            self.model = FakeSAMModel().eval()
            self.device = torch.device(device)
            self.patch_size = 4
            self.upstream_commit = upstream_commit or "fake-commit"

        @staticmethod
        def _rgb_255(image):
            return np.asarray(image, dtype=np.float32)

    source = tmp_path / "source.ckpt"
    source.write_bytes(b"source-model")
    fake_model = FakeSAMModel()
    head = PriorConditionedChangeHead.build(fake_model.map_decoder)
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
        head.change_projection.bias.copy_(torch.tensor([0.0, 2.0, -2.0]))
        head.operation_classifier[-1].bias.copy_(
            torch.tensor([0.0, 3.0, 0.0, 0.0])
        )
    change = tmp_path / "change.pt"
    torch.save(
        {
            "schema_version": "muno21-prior-conditioned-head-v2",
            "operation_point": {"commit_threshold": 0.6},
            "source_checkpoint_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "head_state_dict": head.state_dict(),
        },
        change,
    )
    monkeypatch.setattr(prior_module, "SAMRoadPredictor", FakeSAMRoadPredictor)
    predictor = PriorConditionedSAMRoadPredictor(
        tmp_path,
        tmp_path / "config.yml",
        source,
        tmp_path / "sam.pth",
        change,
        device="cpu",
    )

    result = predictor.predict(
        np.ones((3, 6, 5), dtype=np.float32),
        np.zeros((6, 5), dtype=np.float32),
    )

    assert result["mask_probability"].shape == (6, 5)
    assert result["add_probability"].shape == (6, 5)
    assert result["remove_probability"].shape == (6, 5)
    assert np.allclose(result["add_probability"], torch.sigmoid(torch.tensor(2.0)))
    assert result["predicted_edit"] == "ADD"
    assert result["gated_edit"] == "ADD"
    assert sum(result["edit_probabilities"]) == pytest.approx(1.0)
