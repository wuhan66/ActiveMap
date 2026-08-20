from pathlib import Path

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from activemap.evaluation.updater_calibration import select_operating_point  # noqa: E402
from activemap.evaluation.updater_temporal_calibration import (  # noqa: E402
    select_temporal_change_thresholds,
)
from activemap.inference import UpdaterPredictor, _image_channels_first  # noqa: E402
from activemap.models import EditOperation  # noqa: E402
from activemap.nn.updater import (  # noqa: E402
    PriorConditionedUNet,
    UpdaterConfig,
    combine_confidence_targets,
    dice_loss,
    hierarchical_edit_predictions,
    operation_probabilities,
    segmentation_evidence_features,
    soft_cldice_loss,
    temporal_change_evidence_features,
    updater_loss,
)
from activemap.synthetic_updater import generate_updater_smoke_dataset  # noqa: E402
from activemap.training.updater import (  # noqa: E402
    _batch_metrics,
    _filter_samples_by_edit,
    _quality_score,
    _safety_eligible,
    _training_sample_weights,
    initialize_updater_weights,
    run_updater_epoch,
    set_updater_trainable_scope,
    train_updater,
    updater_config_from_payload,
)
from activemap.training.updater_data import (  # noqa: E402
    UpdaterDataset,
    transform_geometry_delta,
)
from activemap.updater_records import UpdaterSample  # noqa: E402


def test_product_confidence_requires_joint_segmentation_and_edit_quality() -> None:
    segmentation = torch.tensor([1.0, 0.8])
    edit = torch.tensor([0.05, 0.5])
    mean_target = combine_confidence_targets(segmentation, edit, mode="mean")
    product_target = combine_confidence_targets(segmentation, edit, mode="product")
    assert torch.all(product_target <= mean_target)
    assert product_target[0].item() == pytest.approx(0.05)
    assert mean_target[0].item() == pytest.approx(0.525)
    with pytest.raises(ValueError, match="confidence_target_mode"):
        combine_confidence_targets(segmentation, edit, mode="invalid")


def test_safety_checkpoint_gate_enforces_configured_limits() -> None:
    metrics = {"false_edit_rate": 0.06, "missed_edit_rate": 0.10}
    eligible, limits = _safety_eligible(
        metrics,
        {"safety_false_edit_limit": 0.05, "safety_missed_edit_limit": 0.15},
    )
    assert eligible is False
    assert limits == {"false_edit_rate": 0.05, "missed_edit_rate": 0.15}

    eligible, _ = _safety_eligible(
        {**metrics, "false_edit_rate": 0.05},
        {"safety_false_edit_limit": 0.05},
    )
    assert eligible is True


def test_safety_checkpoint_gate_rejects_invalid_limit() -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        _safety_eligible(
            {"false_edit_rate": 0.0, "missed_edit_rate": 0.0},
            {"safety_false_edit_limit": 1.1},
        )


def test_updater_dataset_resizes_mixed_crop_shapes(tmp_path: Path) -> None:
    np.save(tmp_path / "image.npy", np.zeros((3, 16, 16), dtype=np.float32))
    np.save(tmp_path / "mask.npy", np.zeros((1, 16, 16), dtype=np.float32))
    sample = UpdaterSample(
        sample_id="resize",
        split="train",
        image_path=str(tmp_path / "image.npy"),
        prior_mask_path=str(tmp_path / "mask.npy"),
        target_mask_path=str(tmp_path / "mask.npy"),
        edit_type=EditOperation.KEEP,
        geometry_delta=[0.0] * 8,
        dataset_name="spacenet7",
    )
    batch = UpdaterDataset([sample], input_size=32)[0]
    assert batch["image"].shape == (3, 32, 32)
    assert batch["prior_mask"].shape == (1, 32, 32)
    assert batch["dataset_name"] == "spacenet7"


def test_temporal_pair_dataset_concatenates_old_then_current_rgb(tmp_path: Path) -> None:
    np.save(tmp_path / "old.npy", np.full((3, 16, 16), 0.25, dtype=np.float32))
    np.save(tmp_path / "current.npy", np.full((3, 16, 16), 0.75, dtype=np.float32))
    np.save(tmp_path / "mask.npy", np.zeros((1, 16, 16), dtype=np.float32))
    sample = UpdaterSample(
        sample_id="temporal",
        split="train",
        image_path=str(tmp_path / "current.npy"),
        prior_image_path=str(tmp_path / "old.npy"),
        prior_mask_path=str(tmp_path / "mask.npy"),
        target_mask_path=str(tmp_path / "mask.npy"),
        edit_type=EditOperation.KEEP,
        geometry_delta=[0.0] * 8,
    )
    batch = UpdaterDataset([sample], temporal_pair_input=True)[0]
    image = batch["image"]
    assert image.shape == (6, 16, 16)
    assert torch.allclose(image[:3], torch.full((3, 16, 16), 0.25))
    assert torch.allclose(image[3:], torch.full((3, 16, 16), 0.75))

    model = PriorConditionedUNet(
        UpdaterConfig(
            image_channels=6,
            temporal_pair_input=True,
            base_channels=8,
            dropout=0.0,
        )
    )
    assert model(image[None], batch["prior_mask"][None])["segmentation_logits"].shape == (
        1,
        1,
        16,
        16,
    )
    with pytest.raises(ValueError, match="six image channels"):
        PriorConditionedUNet(UpdaterConfig(temporal_pair_input=True))


def test_updater_predictor_accepts_six_channel_temporal_pairs(tmp_path: Path) -> None:
    model = PriorConditionedUNet(
        UpdaterConfig(
            image_channels=6,
            temporal_pair_input=True,
            base_channels=8,
            dropout=0.0,
        )
    )
    checkpoint = tmp_path / "temporal.pt"
    torch.save(
        {"state_dict": model.state_dict(), "model_config": model.config.__dict__}, checkpoint
    )

    predictor = UpdaterPredictor(checkpoint, device="cpu")
    output = predictor.predict(np.zeros((6, 16, 16), dtype=np.float32), np.zeros((16, 16)))

    assert output["mask_probability"].shape == (16, 16)
    assert _image_channels_first(
        np.zeros((16, 16, 6), dtype=np.float32), expected_channels=6
    ).shape == (6, 16, 16)


def test_encoder_only_stage_initialization_preserves_new_heads(tmp_path: Path) -> None:
    source = PriorConditionedUNet(UpdaterConfig(base_channels=8, dropout=0.0))
    target = PriorConditionedUNet(UpdaterConfig(base_channels=8, dropout=0.0))
    checkpoint = tmp_path / "source.pt"
    torch.save({"state_dict": source.state_dict()}, checkpoint)
    original_head = target.edit_head.weight.detach().clone()

    summary = initialize_updater_weights(
        target,
        checkpoint,
        scope="encoder",
        device=torch.device("cpu"),
    )

    assert summary["scope"] == "encoder"
    assert torch.equal(target.encoder1.layers[0].weight, source.encoder1.layers[0].weight)
    assert torch.equal(target.edit_head.weight, original_head)


def test_segmentation_stage_initialization_loads_decoder_but_preserves_heads(
    tmp_path: Path,
) -> None:
    source = PriorConditionedUNet(UpdaterConfig(base_channels=8, dropout=0.0))
    target = PriorConditionedUNet(
        UpdaterConfig(base_channels=8, dropout=0.0, hierarchical_edit=True)
    )
    checkpoint = tmp_path / "source.pt"
    torch.save({"state_dict": source.state_dict()}, checkpoint)
    original_edit_head = target.edit_head.weight.detach().clone()
    original_presence_head = target.presence_head.weight.detach().clone()

    summary = initialize_updater_weights(
        target,
        checkpoint,
        scope="segmentation",
        device=torch.device("cpu"),
    )

    assert summary["scope"] == "segmentation"
    assert torch.equal(target.encoder1.layers[0].weight, source.encoder1.layers[0].weight)
    assert torch.equal(target.decoder1.layers[0].weight, source.decoder1.layers[0].weight)
    assert torch.equal(target.segmentation_head.weight, source.segmentation_head.weight)
    assert torch.equal(target.edit_head.weight, original_edit_head)
    assert torch.equal(target.presence_head.weight, original_presence_head)


def test_edit_aware_sampling_upweights_delete() -> None:
    samples = [
        UpdaterSample(
            sample_id=operation.value,
            split="train",
            image_path="image.npy",
            prior_mask_path="prior.npy",
            target_mask_path="target.npy",
            edit_type=operation,
            geometry_delta=[0.0] * 8,
            dataset_name="spacenet7",
        )
        for operation in EditOperation
    ]
    weights = _training_sample_weights(
        samples,
        dataset_balance_power=0.0,
        edit_sampling_weights={"DELETE": 3.0},
    )
    assert weights is not None
    assert weights[2] == 3.0
    assert weights[0] == 1.0


def test_source_group_sampling_equalizes_total_scenario_weight() -> None:
    samples = [
        UpdaterSample(
            sample_id=f"large-{index}",
            aoi_id="city",
            split="train",
            image_path="image.npy",
            prior_mask_path="prior.npy",
            target_mask_path="target.npy",
            edit_type=EditOperation.ADD,
            geometry_delta=[0.0] * 8,
            dataset_name="muno21",
            source_metadata={"annotation_index": 1},
        )
        for index in range(4)
    ]
    samples.append(
        UpdaterSample(
            sample_id="small-0",
            aoi_id="city",
            split="train",
            image_path="image.npy",
            prior_mask_path="prior.npy",
            target_mask_path="target.npy",
            edit_type=EditOperation.ADD,
            geometry_delta=[0.0] * 8,
            dataset_name="muno21",
            source_metadata={"annotation_index": 2},
        )
    )
    weights = _training_sample_weights(
        samples,
        dataset_balance_power=0.0,
        edit_sampling_weights={},
        source_group_balance_power=1.0,
    )

    assert weights is not None
    assert sum(weights[:4]) == pytest.approx(weights[4])


def test_allowed_edits_filter_removes_duplicate_segmentation_operations() -> None:
    samples = [
        UpdaterSample(
            sample_id=operation.value,
            split="train",
            image_path="image.npy",
            prior_mask_path="prior.npy",
            target_mask_path="target.npy",
            edit_type=operation,
            geometry_delta=[0.0] * 8,
        )
        for operation in (EditOperation.KEEP, EditOperation.ADD, EditOperation.RESHAPE)
    ]

    filtered = _filter_samples_by_edit(samples, ["keep"])

    assert [sample.edit_type for sample in filtered] == [EditOperation.KEEP]
    with pytest.raises(ValueError, match="unknown data.allowed_edits"):
        _filter_samples_by_edit(samples, ["MOVE"])
    with pytest.raises(ValueError, match="non-empty list"):
        _filter_samples_by_edit(samples, [])


def test_updater_forward_and_loss() -> None:
    model = PriorConditionedUNet(UpdaterConfig(base_channels=8, dropout=0.0))
    image = torch.rand(2, 3, 32, 32)
    prior = torch.rand(2, 1, 32, 32)
    outputs = model(image, prior)
    assert outputs["segmentation_logits"].shape == (2, 1, 32, 32)
    assert outputs["edit_logits"].shape == (2, 4)
    assert outputs["geometry_delta"].shape == (2, 8)
    loss, components = updater_loss(
        outputs,
        target_mask=torch.rand(2, 1, 32, 32),
        valid_mask=torch.ones(2, 1, 32, 32),
        edit_target=torch.tensor([0, 3]),
        geometry_target=torch.zeros(2, 8),
    )
    assert torch.isfinite(loss)
    assert set(components) == {
        "segmentation_bce",
        "segmentation_dice",
        "segmentation_focal",
        "segmentation_cldice",
        "segmentation",
        "edit",
        "presence",
        "change",
        "geometry",
        "false_edit",
        "missed_edit",
        "confidence",
        "temporal_change_bce",
        "temporal_change_dice",
        "temporal_change_focal",
        "temporal_change",
        "temporal_add",
        "temporal_remove",
    }


def test_cldice_penalizes_a_broken_road_more_than_a_connected_road() -> None:
    target = torch.zeros(1, 1, 32, 32)
    target[:, :, 15:18, 3:29] = 1.0
    connected = torch.where(target > 0.5, torch.tensor(8.0), torch.tensor(-8.0))
    broken = connected.clone()
    broken[:, :, 15:18, 15:18] = -8.0
    valid = torch.ones_like(target)

    connected_loss = soft_cldice_loss(connected, target, valid, iterations=5)
    broken_loss = soft_cldice_loss(broken, target, valid, iterations=5)

    assert connected_loss < broken_loss


def test_cldice_backpropagates_to_segmentation_logits() -> None:
    logits = torch.randn(1, 1, 16, 16, requires_grad=True)
    target = torch.zeros_like(logits)
    target[:, :, 7:10, 2:14] = 1.0
    loss = soft_cldice_loss(logits, target, torch.ones_like(target), iterations=3)
    loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) > 0


def test_segmentation_metrics_separate_foreground_and_empty_scenes() -> None:
    model = PriorConditionedUNet(UpdaterConfig(base_channels=4, dropout=0.0))
    prior = torch.zeros(2, 1, 16, 16)
    outputs = model(torch.zeros(2, 3, 16, 16), prior)
    logits = torch.full((2, 1, 16, 16), -20.0)
    logits[0, 0, 1:5, 1:5] = 20.0
    logits[1, 0, 0, 0] = 20.0
    outputs["segmentation_logits"] = logits
    target = torch.zeros(2, 1, 16, 16)
    target[0, 0, 1:5, 1:5] = 1.0
    metrics = _batch_metrics(
        outputs,
        {
            "target_mask": target,
            "valid_mask": torch.ones_like(target),
            "prior_mask": prior,
            "edit_target": torch.zeros(2, dtype=torch.long),
        },
    )

    assert metrics["iou"] == pytest.approx(0.5)
    assert metrics["foreground_iou"] == pytest.approx(1.0)
    assert metrics["empty_scene_false_positive_fraction"] == pytest.approx(1 / 256)


def test_batch_metrics_measure_added_and_removed_change_iou() -> None:
    prior = torch.zeros(2, 1, 8, 8)
    prior[:, :, 1:3, 1:3] = 1.0
    target = prior.clone()
    target[0, :, 5:7, 5:7] = 1.0
    target[1, :, 1:3, 1:3] = 0.0
    logits = torch.where(target > 0.5, torch.tensor(20.0), torch.tensor(-20.0))
    outputs = {
        "segmentation_logits": logits,
        "edit_logits": torch.zeros(2, 4),
        "confidence_logits": torch.zeros(2),
    }
    metrics = _batch_metrics(
        outputs,
        {
            "target_mask": target,
            "valid_mask": torch.ones_like(target),
            "prior_mask": prior,
            "edit_target": torch.tensor([1, 2]),
            "supervision_type": ["full_scene_temporal", "full_scene_temporal"],
        },
    )
    assert metrics["added_change_iou_sum"] == pytest.approx(1.0)
    assert metrics["added_change_iou_count"] == pytest.approx(1.0)
    assert metrics["removed_change_iou_sum"] == pytest.approx(1.0)
    assert metrics["removed_change_iou_count"] == pytest.approx(1.0)


def test_batch_metrics_measure_change_iou_for_real_temporal_crops() -> None:
    prior = torch.zeros(1, 1, 8, 8)
    target = prior.clone()
    target[:, :, 5:7, 5:7] = 1.0
    logits = torch.where(target > 0.5, torch.tensor(20.0), torch.tensor(-20.0))
    outputs = {
        "segmentation_logits": logits,
        "edit_logits": torch.zeros(1, 4),
        "confidence_logits": torch.zeros(1),
    }

    metrics = _batch_metrics(
        outputs,
        {
            "target_mask": target,
            "valid_mask": torch.ones_like(target),
            "prior_mask": prior,
            "edit_target": torch.tensor([1]),
            "supervision_type": ["real_temporal"],
        },
    )

    assert metrics["added_change_iou_sum"] == pytest.approx(1.0)
    assert metrics["added_change_iou_count"] == pytest.approx(1.0)


def test_temporal_change_loss_accepts_real_temporal_supervision() -> None:
    prior = torch.zeros(1, 1, 8, 8)
    target = prior.clone()
    target[:, :, 4:6, 4:6] = 1.0
    add_logits = torch.full_like(prior, -10.0)
    add_logits[:, :, 4:6, 4:6] = 10.0
    remove_logits = torch.full_like(prior, -10.0)
    outputs = {
        "segmentation_logits": add_logits,
        "temporal_change_logits": torch.cat((add_logits, remove_logits), dim=1),
        "edit_logits": torch.zeros(1, 4),
        "geometry_delta": torch.zeros(1, 8),
        "confidence_logits": torch.zeros(1),
    }

    _, components = updater_loss(
        outputs,
        target_mask=target,
        valid_mask=torch.ones_like(target),
        edit_target=torch.tensor([1]),
        geometry_target=torch.zeros(1, 8),
        prior_mask=prior,
        full_scene_mask=torch.tensor([False]),
        temporal_supervision_mask=torch.tensor([True]),
        temporal_change_weight=1.0,
    )

    assert components["temporal_change"] < 0.01


def test_temporal_quality_score_uses_change_iou_instead_of_full_raster_iou() -> None:
    metrics = {
        "iou": 0.99,
        "added_change_iou": 0.20,
        "added_change_iou_count": 3.0,
        "removed_change_iou": 0.40,
        "removed_change_iou_count": 2.0,
        "edit_accuracy": 0.80,
        "delete_recall": 0.50,
    }
    score, mode = _quality_score(
        metrics,
        {"quality_score_mode": "temporal_change", "quality_delete_recall_weight": 0.2},
    )
    assert mode == "temporal_change"
    assert score == pytest.approx(0.30 + 0.80 + 0.10)


def test_edit_quality_score_penalizes_false_and_missed_updates() -> None:
    metrics = {
        "iou": 0.99,
        "edit_accuracy": 0.80,
        "delete_recall": 0.50,
        "false_edit_rate": 0.10,
        "missed_edit_rate": 0.20,
    }
    score, mode = _quality_score(
        metrics,
        {
            "quality_score_mode": "edit",
            "quality_delete_recall_weight": 0.2,
            "quality_false_edit_weight": 0.5,
            "quality_missed_edit_weight": 0.5,
        },
    )
    assert mode == "edit"
    assert score == pytest.approx(0.80 + 0.10 - 0.05 - 0.10)


def test_hierarchical_edit_probabilities_factor_presence_and_change() -> None:
    prior = torch.ones(3, 1, 8, 8)
    outputs = {
        "edit_logits": torch.zeros(3, 4),
        "presence_logits": torch.tensor([8.0, -8.0, 8.0]),
        "change_logits": torch.tensor([-8.0, 0.0, 8.0]),
    }
    predicted = torch.argmax(operation_probabilities(outputs, prior), dim=-1)
    assert predicted.tolist() == [0, 2, 3]


def test_hierarchical_updater_can_remove_auxiliary_edit_classifier() -> None:
    model = PriorConditionedUNet(
        UpdaterConfig(
            base_channels=4,
            dropout=0.0,
            hierarchical_edit=True,
            auxiliary_edit_head=False,
        )
    )
    image = torch.rand(2, 3, 16, 16)
    prior = torch.zeros(2, 1, 16, 16)
    prior[0] = 1.0
    outputs = model(image, prior)

    assert not hasattr(model, "edit_head")
    assert "edit_logits" not in outputs
    probabilities = operation_probabilities(outputs, prior)
    assert probabilities.shape == (2, 4)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(2), atol=1e-6)

    total, components = updater_loss(
        outputs,
        target_mask=torch.rand(2, 1, 16, 16),
        valid_mask=torch.ones(2, 1, 16, 16),
        edit_target=torch.tensor([3, 1]),
        geometry_target=torch.zeros(2, 8),
        edit_weight=1.0,
        presence_weight=1.0,
        change_weight=1.0,
        prior_mask=prior,
    )
    total.backward()
    assert torch.isfinite(total)
    assert components["edit"].item() == 0.0
    assert model.presence_head.weight.grad is not None


def test_edit_specific_geometry_head_routes_supervision_by_operation() -> None:
    model = PriorConditionedUNet(
        UpdaterConfig(
            base_channels=4,
            dropout=0.0,
            geometry_head_mode="edit_specific",
        )
    )
    outputs = model(torch.rand(2, 3, 16, 16), torch.zeros(2, 1, 16, 16))
    assert outputs["geometry_delta_by_edit"].shape == (2, 4, 8)
    assert outputs["geometry_delta"].shape == (2, 8)

    total, _ = updater_loss(
        outputs,
        target_mask=torch.rand(2, 1, 16, 16),
        valid_mask=torch.ones(2, 1, 16, 16),
        edit_target=torch.tensor([1, 3]),
        geometry_target=torch.ones(2, 8),
        geometry_weight=1.0,
    )
    total.backward()
    assert model.geometry_head.weight.grad is not None
    gradient = model.geometry_head.weight.grad.reshape(4, 8, -1)
    assert torch.count_nonzero(gradient[1]) > 0
    assert torch.count_nonzero(gradient[3]) > 0
    assert torch.count_nonzero(gradient[0]) == 0
    assert torch.count_nonzero(gradient[2]) == 0


def test_structural_updater_ablation_options_are_validated() -> None:
    with pytest.raises(ValueError, match="hierarchical_edit"):
        PriorConditionedUNet(UpdaterConfig(auxiliary_edit_head=False))
    with pytest.raises(ValueError, match="geometry_head_mode"):
        PriorConditionedUNet(UpdaterConfig(geometry_head_mode="invalid"))


def test_calibrated_hierarchical_threshold_controls_delete() -> None:
    prior = torch.ones(2, 1, 8, 8)
    outputs = {
        "edit_logits": torch.zeros(2, 4),
        "presence_logits": torch.logit(torch.tensor([0.4, 0.6])),
        "change_logits": torch.logit(torch.tensor([0.2, 0.8])),
    }
    predicted = hierarchical_edit_predictions(
        outputs,
        prior,
        presence_threshold=0.5,
        change_threshold=0.5,
    )
    assert predicted.tolist() == [2, 3]


def test_calibration_respects_false_edit_constraint() -> None:
    target = np.asarray([0, 0, 0, 2, 2, 2, 3, 3])
    auxiliary = target.copy()
    has_prior = np.ones_like(target, dtype=bool)
    presence = np.asarray([0.9, 0.8, 0.7, 0.2, 0.4, 0.6, 0.8, 0.9])
    change = np.asarray([0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.8, 0.9])
    selected, _ = select_operating_point(
        target,
        auxiliary,
        has_prior,
        presence,
        change,
        max_false_edit=0.0,
        grid_steps=9,
    )
    assert selected["false_edit_rate"] == 0.0
    assert selected["delete_recall"] > 0.0


def test_temporal_change_calibration_selects_safe_independent_thresholds() -> None:
    target = np.zeros((3, 1, 1, 4), dtype=np.float32)
    prior = np.zeros_like(target)
    valid = np.ones_like(target)
    target[0, 0, 0, 0] = 1.0
    target[1, 0, 0, 1] = 1.0
    add = np.asarray(
        [
            [[[0.8, 0.2, 0.2, 0.2]]],
            [[[0.2, 0.6, 0.2, 0.2]]],
            [[[0.4, 0.4, 0.4, 0.4]]],
        ],
        dtype=np.float32,
    )

    prior[:, :, :, 2:] = 1.0
    target[:, :, :, 2:] = prior[:, :, :, 2:]
    target[0, 0, 0, 2] = 0.0
    target[1, 0, 0, 3] = 0.0
    remove = np.asarray(
        [
            [[[0.0, 0.0, 0.8, 0.2]]],
            [[[0.0, 0.0, 0.2, 0.6]]],
            [[[0.0, 0.0, 0.4, 0.4]]],
        ],
        dtype=np.float32,
    )
    result = select_temporal_change_thresholds(
        add,
        remove,
        target,
        prior,
        valid,
        max_stable_false_positive=0.0,
        grid_steps=19,
    )
    assert result["constraint_satisfied"] is True
    assert 0.4 < result["selected"]["add_threshold"] <= 0.6
    assert 0.4 < result["selected"]["remove_threshold"] <= 0.6
    assert result["selected"]["harmonic_mean_positive_iou"] == pytest.approx(1.0)


def test_hierarchical_heads_receive_conditional_losses() -> None:
    model = PriorConditionedUNet(
        UpdaterConfig(base_channels=4, dropout=0.0, hierarchical_edit=True)
    )
    prior = torch.ones(3, 1, 16, 16)
    outputs = model(torch.rand(3, 3, 16, 16), prior)
    total, components = updater_loss(
        outputs,
        target_mask=torch.rand(3, 1, 16, 16),
        valid_mask=torch.ones(3, 1, 16, 16),
        edit_target=torch.tensor([0, 2, 3]),
        geometry_target=torch.zeros(3, 8),
        presence_weight=1.0,
        change_weight=1.0,
        prior_mask=prior,
    )
    total.backward()
    assert components["presence"].item() > 0.0
    assert components["change"].item() > 0.0
    assert model.presence_head.weight.grad is not None
    assert model.change_head.weight.grad is not None


def test_prior_guided_roi_branch_receives_hierarchical_gradients() -> None:
    model = PriorConditionedUNet(
        UpdaterConfig(
            base_channels=4,
            dropout=0.0,
            hierarchical_edit=True,
            prior_guided_roi=True,
        )
    )
    image = torch.rand(3, 3, 32, 32)
    prior = torch.zeros(3, 1, 32, 32)
    prior[:, :, 8:24, 8:24] = 1.0
    outputs = model(image, prior)
    total, _ = updater_loss(
        outputs,
        target_mask=torch.rand(3, 1, 32, 32),
        valid_mask=torch.ones(3, 1, 32, 32),
        edit_target=torch.tensor([0, 2, 3]),
        geometry_target=torch.zeros(3, 8),
        presence_weight=1.0,
        change_weight=1.0,
        prior_mask=prior,
    )
    total.backward()
    projection = model.roi_projection[0]
    assert isinstance(projection, torch.nn.Linear)
    assert projection.weight.grad is not None
    assert torch.count_nonzero(projection.weight.grad) > 0


def test_prior_guided_roi_keeps_legacy_head_shapes_for_compatible_init(tmp_path: Path) -> None:
    source = PriorConditionedUNet(
        UpdaterConfig(base_channels=4, dropout=0.0, hierarchical_edit=True)
    )
    target = PriorConditionedUNet(
        UpdaterConfig(
            base_channels=4,
            dropout=0.0,
            hierarchical_edit=True,
            prior_guided_roi=True,
        )
    )
    checkpoint = tmp_path / "hierarchical.pt"
    torch.save({"state_dict": source.state_dict()}, checkpoint)
    summary = initialize_updater_weights(
        target,
        checkpoint,
        scope="compatible",
        device=torch.device("cpu"),
    )
    assert torch.equal(target.presence_head.weight, source.presence_head.weight)
    assert torch.equal(target.change_head.weight, source.change_head.weight)
    assert summary["loaded_tensor_count"] == len(source.state_dict())
    source.eval()
    target.eval()
    image = torch.rand(2, 3, 32, 32)
    prior = torch.zeros(2, 1, 32, 32)
    prior[:, :, 8:24, 8:24] = 1.0
    source_outputs = source(image, prior)
    target_outputs = target(image, prior)
    assert torch.equal(source_outputs["presence_logits"], target_outputs["presence_logits"])
    assert torch.equal(source_outputs["change_logits"], target_outputs["change_logits"])


def test_roi_warmup_freezes_inherited_model_then_restores_it() -> None:
    model = PriorConditionedUNet(
        UpdaterConfig(base_channels=4, hierarchical_edit=True, prior_guided_roi=True)
    )
    roi_count = set_updater_trainable_scope(model, roi_only=True)
    assert roi_count == model.roi_projection[0].weight.numel()
    assert model.roi_projection[0].weight.requires_grad
    assert not model.presence_head.weight.requires_grad
    all_count = set_updater_trainable_scope(model, roi_only=False)
    assert all_count == model.parameter_count()
    assert model.presence_head.weight.requires_grad


def test_segmentation_evidence_measures_prior_occupancy_and_context() -> None:
    prior = torch.zeros(2, 1, 32, 32)
    prior[:, :, 8:24, 8:24] = 1.0
    logits = torch.full_like(prior, -5.0)
    logits[0, :, 8:24, 8:24] = 5.0
    logits[1, :, :8, :] = 5.0
    evidence = segmentation_evidence_features(logits, prior)
    assert evidence.shape == (2, 4)
    assert evidence[0, 0] > 0.99
    assert evidence[0, 1] < 0.01
    assert evidence[0, 2] > evidence[1, 2]
    assert evidence[0, 3] > 0.98
    assert evidence[1, 0] < 0.01


def test_segmentation_evidence_is_zero_initialized_compatible_residual(
    tmp_path: Path,
) -> None:
    source = PriorConditionedUNet(
        UpdaterConfig(base_channels=4, dropout=0.0, hierarchical_edit=True)
    )
    target = PriorConditionedUNet(
        UpdaterConfig(
            base_channels=4,
            dropout=0.0,
            hierarchical_edit=True,
            segmentation_evidence=True,
        )
    )
    checkpoint = tmp_path / "hierarchical.pt"
    torch.save({"state_dict": source.state_dict()}, checkpoint)
    initialize_updater_weights(target, checkpoint, scope="compatible", device=torch.device("cpu"))
    source.eval()
    target.eval()
    image = torch.rand(2, 3, 32, 32)
    prior = torch.zeros(2, 1, 32, 32)
    prior[:, :, 8:24, 8:24] = 1.0
    source_outputs = source(image, prior)
    target_outputs = target(image, prior)
    assert torch.equal(source_outputs["presence_logits"], target_outputs["presence_logits"])
    assert torch.equal(source_outputs["change_logits"], target_outputs["change_logits"])
    assert torch.count_nonzero(target.segmentation_evidence_head.weight) == 0


def test_segmentation_evidence_head_receives_hierarchical_gradients() -> None:
    model = PriorConditionedUNet(
        UpdaterConfig(
            base_channels=4,
            dropout=0.0,
            hierarchical_edit=True,
            segmentation_evidence=True,
        )
    )
    prior = torch.zeros(3, 1, 32, 32)
    prior[:, :, 8:24, 8:24] = 1.0
    outputs = model(torch.rand(3, 3, 32, 32), prior)
    total, _ = updater_loss(
        outputs,
        target_mask=torch.rand(3, 1, 32, 32),
        valid_mask=torch.ones(3, 1, 32, 32),
        edit_target=torch.tensor([0, 2, 3]),
        geometry_target=torch.zeros(3, 8),
        presence_weight=1.0,
        change_weight=1.0,
        prior_mask=prior,
    )
    total.backward()
    gradient = model.segmentation_evidence_head.weight.grad
    assert gradient is not None
    assert torch.count_nonzero(gradient) > 0


def test_evidence_warmup_and_config_parser_cover_new_architecture_flags() -> None:
    config = updater_config_from_payload(
        {
            "base_channels": 4,
            "hierarchical_edit": True,
            "prior_guided_roi": False,
            "segmentation_evidence": True,
        }
    )
    assert config.segmentation_evidence is True
    assert config.prior_guided_roi is False
    model = PriorConditionedUNet(config)
    evidence_count = set_updater_trainable_scope(model, evidence_only=True)
    assert evidence_count == model.segmentation_evidence_head.weight.numel()
    assert model.segmentation_evidence_head.weight.requires_grad
    assert not model.presence_head.weight.requires_grad
    set_updater_trainable_scope(model)
    assert model.presence_head.weight.requires_grad


def test_vector_change_encoder_is_hierarchical_and_prior_conditioned() -> None:
    with pytest.raises(ValueError, match="hierarchical_edit"):
        PriorConditionedUNet(UpdaterConfig(base_channels=4, vector_change_encoder=True))
    with pytest.raises(ValueError, match="use_prior"):
        PriorConditionedUNet(
            UpdaterConfig(
                base_channels=4,
                use_prior=False,
                hierarchical_edit=True,
                vector_change_encoder=True,
            )
        )


def test_vector_change_encoder_receives_hierarchical_gradients() -> None:
    model = PriorConditionedUNet(
        UpdaterConfig(
            base_channels=4,
            dropout=0.0,
            hierarchical_edit=True,
            vector_change_encoder=True,
        )
    )
    image = torch.rand(3, 3, 32, 32)
    prior = torch.zeros(3, 1, 32, 32)
    prior[:, :, 8:24, 8:24] = 1.0
    outputs = model(image, prior)
    assert outputs["change_descriptor"].shape == (3, 32)
    assert outputs["change_residual"].shape == (3, 16)
    total, _ = updater_loss(
        outputs,
        target_mask=torch.rand(3, 1, 32, 32),
        valid_mask=torch.ones(3, 1, 32, 32),
        edit_target=torch.tensor([0, 2, 3]),
        geometry_target=torch.zeros(3, 8),
        presence_weight=1.0,
        change_weight=1.0,
        prior_mask=prior,
    )
    total.backward()
    image_gradient = model.change_image_encoder[0].layers[0].weight.grad
    prior_gradient = model.change_prior_encoder[0].layers[0].weight.grad
    fusion_gradient = model.change_fusion[0].weight.grad
    assert image_gradient is not None and torch.count_nonzero(image_gradient) > 0
    assert prior_gradient is not None and torch.count_nonzero(prior_gradient) > 0
    assert fusion_gradient is not None and torch.count_nonzero(fusion_gradient) > 0


def test_vector_change_encoder_config_is_persisted_by_parser() -> None:
    config = updater_config_from_payload(
        {
            "base_channels": 4,
            "hierarchical_edit": True,
            "vector_change_encoder": True,
            "vector_change_to_edit_head": True,
            "temporal_change_head": True,
            "temporal_change_to_edit_head": True,
            "temporal_spatial_edit_head": True,
        }
    )
    assert config.vector_change_encoder is True
    assert config.vector_change_to_edit_head is True
    assert config.temporal_change_head is True
    assert config.temporal_change_to_edit_head is True
    assert config.temporal_spatial_edit_head is True
    model = PriorConditionedUNet(config)
    assert model.config.as_dict()["vector_change_encoder"] is True
    assert model.config.as_dict()["vector_change_to_edit_head"] is True
    assert model.config.as_dict()["temporal_change_head"] is True
    assert model.config.as_dict()["temporal_change_to_edit_head"] is True
    assert model.config.as_dict()["temporal_spatial_edit_head"] is True


def test_temporal_change_head_reconstructs_target_logits_from_prior() -> None:
    model = PriorConditionedUNet(
        UpdaterConfig(base_channels=4, dropout=0.0, temporal_change_head=True)
    )
    torch.nn.init.zeros_(model.temporal_change_head.weight)
    with torch.no_grad():
        model.temporal_change_head.bias.copy_(torch.tensor([2.0, 3.0]))
    prior = torch.zeros(1, 1, 16, 16)
    prior[:, :, 4:12, 4:12] = 1.0
    outputs = model(torch.zeros(1, 3, 16, 16), prior)
    assert outputs["temporal_change_logits"].shape == (1, 2, 16, 16)
    assert torch.all(outputs["segmentation_logits"][prior < 0.5] == 2.0)
    assert torch.all(outputs["segmentation_logits"][prior >= 0.5] == -3.0)


def test_temporal_change_evidence_distinguishes_add_and_remove() -> None:
    prior = torch.zeros(2, 1, 8, 8)
    prior[:, :, :, 4:] = 1.0
    logits = torch.full((2, 2, 8, 8), -8.0)
    logits[0, 0, :, :4] = 8.0
    logits[1, 1, :, 4:] = 8.0
    evidence = temporal_change_evidence_features(logits, prior)
    assert evidence.shape == (2, 6)
    assert evidence[0, 0] > 0.99
    assert evidence[0, 3] < 0.01
    assert evidence[1, 0] < 0.01
    assert evidence[1, 3] > 0.99


def test_temporal_change_edit_evidence_head_receives_classification_gradient() -> None:
    model = PriorConditionedUNet(
        UpdaterConfig(
            base_channels=4,
            dropout=0.0,
            temporal_change_head=True,
            temporal_change_to_edit_head=True,
        )
    )
    assert torch.count_nonzero(model.temporal_change_evidence_head.weight) == 0
    image = torch.rand(3, 3, 16, 16)
    prior = torch.zeros(3, 1, 16, 16)
    prior[1:, :, 4:12, 4:12] = 1.0
    outputs = model(image, prior)
    loss = torch.nn.functional.cross_entropy(outputs["edit_logits"], torch.tensor([1, 2, 3]))
    loss.backward()
    gradient = model.temporal_change_evidence_head.weight.grad
    assert gradient is not None
    assert torch.count_nonzero(gradient) > 0


def test_temporal_change_edit_evidence_supports_frozen_warmup() -> None:
    model = PriorConditionedUNet(
        UpdaterConfig(
            base_channels=4,
            temporal_change_head=True,
            temporal_change_to_edit_head=True,
        )
    )
    count = set_updater_trainable_scope(model, evidence_only=True)
    assert count == model.temporal_change_evidence_head.weight.numel()
    assert model.temporal_change_evidence_head.weight.requires_grad
    assert not model.edit_head.weight.requires_grad


def test_temporal_spatial_edit_head_preserves_spatial_evidence_and_supports_warmup() -> None:
    with pytest.raises(ValueError, match="temporal_change_head"):
        PriorConditionedUNet(UpdaterConfig(base_channels=4, temporal_spatial_edit_head=True))

    model = PriorConditionedUNet(
        UpdaterConfig(
            base_channels=4,
            dropout=0.0,
            temporal_change_head=True,
            temporal_spatial_edit_head=True,
        )
    )
    final_layer = model.temporal_spatial_edit_encoder[-1]
    assert torch.count_nonzero(final_layer.weight) == 0

    image = torch.rand(3, 3, 16, 16)
    prior = torch.zeros(3, 1, 16, 16)
    prior[1:, :, 4:12, 4:12] = 1.0
    outputs = model(image, prior)
    assert outputs["temporal_spatial_input"].shape == (3, 3, 16, 16)
    loss = torch.nn.functional.cross_entropy(outputs["edit_logits"], torch.tensor([1, 2, 3]))
    loss.backward()
    assert final_layer.weight.grad is not None
    assert torch.count_nonzero(final_layer.weight.grad) > 0

    count = set_updater_trainable_scope(model, evidence_only=True)
    expected = sum(
        parameter.numel() for parameter in model.temporal_spatial_edit_encoder.parameters()
    )
    assert count == expected
    assert final_layer.weight.requires_grad
    assert not model.edit_head.weight.requires_grad


def test_frozen_warmup_keeps_backbone_in_evaluation_mode() -> None:
    model = PriorConditionedUNet(
        UpdaterConfig(
            base_channels=4,
            temporal_change_head=True,
            temporal_change_to_edit_head=True,
        )
    )
    set_updater_trainable_scope(model, evidence_only=True)
    sample = {
        "image": torch.rand(2, 3, 16, 16),
        "prior_mask": torch.zeros(2, 1, 16, 16),
        "target_mask": torch.zeros(2, 1, 16, 16),
        "valid_mask": torch.ones(2, 1, 16, 16),
        "edit_target": torch.tensor([0, 1]),
        "geometry_target": torch.zeros(2, 8),
        "supervision_type": ["full_scene_temporal", "full_scene_temporal"],
    }
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.1
    )
    before = model.encoder1.layers[1].running_mean.clone()
    run_updater_epoch(
        model,
        [sample],
        device=torch.device("cpu"),
        optimizer=optimizer,
        loss_settings={
            "segmentation": 0.0,
            "segmentation_bce": 1.0,
            "segmentation_dice": 1.0,
            "segmentation_focal": 0.0,
            "focal_gamma": 2.0,
            "focal_alpha": 0.75,
            "edit": 1.0,
            "edit_class_weights": None,
            "edit_label_smoothing": 0.0,
            "geometry": 0.0,
            "geometry_beta": 0.1,
            "false_edit": 0.0,
            "missed_edit": 0.0,
            "confidence": 0.0,
            "confidence_target_mode": "mean",
            "presence": 0.0,
            "change": 0.0,
            "temporal_change": 0.0,
            "temporal_change_bce": 1.0,
            "temporal_change_dice": 1.0,
            "temporal_change_focal": 0.0,
            "temporal_change_focal_alpha": 0.9,
        },
        grad_clip=1.0,
        show_progress=False,
        frozen_backbone=True,
    )
    assert model.training is False
    assert torch.equal(model.encoder1.layers[1].running_mean, before)


def test_explicit_temporal_channels_penalize_copying_prior() -> None:
    prior = torch.zeros(1, 1, 16, 16)
    prior[:, :, 3:7, 2:6] = 1.0
    target = prior.clone()
    target[:, :, 3:7, 2:6] = 0.0
    target[:, :, 9:13, 10:14] = 1.0
    valid = torch.ones_like(target)

    def outputs_for(
        add_logits: torch.Tensor, remove_logits: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        temporal_logits = torch.cat((add_logits, remove_logits), dim=1)
        reconstructed = torch.where(prior >= 0.5, -remove_logits, add_logits)
        return {
            "segmentation_logits": reconstructed,
            "temporal_change_logits": temporal_logits,
            "edit_logits": torch.zeros(1, 4),
            "geometry_delta": torch.zeros(1, 8),
            "confidence_logits": torch.zeros(1),
        }

    correct_add = torch.full_like(prior, -10.0)
    correct_add[:, :, 9:13, 10:14] = 10.0
    correct_remove = torch.full_like(prior, -10.0)
    correct_remove[:, :, 3:7, 2:6] = 10.0
    copied_add = torch.full_like(prior, -10.0)
    copied_remove = torch.full_like(prior, -10.0)
    kwargs = {
        "target_mask": target,
        "valid_mask": valid,
        "edit_target": torch.tensor([3]),
        "geometry_target": torch.zeros(1, 8),
        "segmentation_weight": 0.0,
        "edit_weight": 0.0,
        "geometry_weight": 0.0,
        "false_edit_weight": 0.0,
        "missed_edit_weight": 0.0,
        "confidence_weight": 0.0,
        "prior_mask": prior,
        "full_scene_mask": torch.tensor([True]),
        "temporal_change_weight": 1.0,
    }
    correct_loss, correct_components = updater_loss(
        outputs_for(correct_add, correct_remove), **kwargs
    )
    copied_loss, _ = updater_loss(outputs_for(copied_add, copied_remove), **kwargs)
    assert correct_loss < copied_loss
    assert correct_components["temporal_add"] < 0.01
    assert correct_components["temporal_remove"] < 0.01


def test_temporal_change_loss_penalizes_copying_prior() -> None:
    model = PriorConditionedUNet(UpdaterConfig(base_channels=4, dropout=0.0))
    image = torch.zeros(1, 3, 16, 16)
    prior = torch.zeros(1, 1, 16, 16)
    prior[:, :, 4:8, 2:6] = 1.0
    target = prior.clone()
    target[:, :, 4:8, 10:14] = 1.0
    valid = torch.ones_like(target)
    outputs = model(image, prior)
    copied_outputs = dict(outputs)
    correct_outputs = dict(outputs)
    copied_outputs["segmentation_logits"] = torch.where(
        prior > 0.5, torch.full_like(prior, 12.0), torch.full_like(prior, -12.0)
    )
    correct_outputs["segmentation_logits"] = torch.where(
        target > 0.5, torch.full_like(target, 12.0), torch.full_like(target, -12.0)
    )
    kwargs = {
        "target_mask": target,
        "valid_mask": valid,
        "edit_target": torch.tensor([1]),
        "geometry_target": torch.zeros(1, 8),
        "segmentation_weight": 0.0,
        "edit_weight": 0.0,
        "geometry_weight": 0.0,
        "false_edit_weight": 0.0,
        "missed_edit_weight": 0.0,
        "confidence_weight": 0.0,
        "prior_mask": prior,
        "full_scene_mask": torch.tensor([True]),
        "temporal_change_weight": 1.0,
    }

    copied_loss, copied_components = updater_loss(copied_outputs, **kwargs)
    correct_loss, correct_components = updater_loss(correct_outputs, **kwargs)

    assert correct_loss < copied_loss
    assert correct_components["temporal_change"] < copied_components["temporal_change"]


def test_compatible_initialization_adds_hierarchical_heads(tmp_path: Path) -> None:
    source = PriorConditionedUNet(UpdaterConfig(base_channels=4, dropout=0.0))
    target = PriorConditionedUNet(
        UpdaterConfig(base_channels=4, dropout=0.0, hierarchical_edit=True)
    )
    checkpoint = tmp_path / "flat.pt"
    torch.save({"state_dict": source.state_dict()}, checkpoint)
    original_presence = target.presence_head.weight.detach().clone()
    summary = initialize_updater_weights(
        target,
        checkpoint,
        scope="compatible",
        device=torch.device("cpu"),
    )
    assert summary["scope"] == "compatible"
    assert torch.equal(target.encoder1.layers[0].weight, source.encoder1.layers[0].weight)
    assert torch.equal(target.presence_head.weight, original_presence)


def test_empty_target_dice_penalizes_foreground_probability() -> None:
    target = torch.zeros(1, 1, 8, 8)
    valid = torch.ones_like(target)
    low_foreground = dice_loss(torch.full_like(target, -5.0), target, valid)
    high_foreground = dice_loss(torch.full_like(target, 5.0), target, valid)
    assert low_foreground < high_foreground


def test_updater_loss_is_finite_when_batch_contains_only_keep() -> None:
    outputs = {
        "segmentation_logits": torch.randn(2, 1, 8, 8, requires_grad=True),
        "edit_logits": torch.randn(2, 4, requires_grad=True),
        "geometry_delta": torch.randn(2, 8, requires_grad=True),
        "confidence_logits": torch.randn(2, requires_grad=True),
    }
    total, components = updater_loss(
        outputs,
        target_mask=torch.zeros(2, 1, 8, 8),
        valid_mask=torch.ones(2, 1, 8, 8),
        edit_target=torch.zeros(2, dtype=torch.long),
        geometry_target=torch.zeros(2, 8),
    )
    total.backward()
    assert torch.isfinite(total)
    assert components["geometry"].item() == 0.0
    assert components["missed_edit"].item() == 0.0


def test_no_prior_ablation_is_capacity_matched_and_ignores_prior() -> None:
    full = PriorConditionedUNet(UpdaterConfig(base_channels=4, dropout=0.0, use_prior=True))
    ablated = PriorConditionedUNet(UpdaterConfig(base_channels=4, dropout=0.0, use_prior=False))
    assert full.parameter_count() == ablated.parameter_count()
    ablated.eval()
    image = torch.rand(1, 3, 16, 16)
    first = ablated(image, torch.zeros(1, 1, 16, 16))["segmentation_logits"]
    second = ablated(image, torch.ones(1, 1, 16, 16))["segmentation_logits"]
    assert torch.allclose(first, second)


def test_geometry_delta_follows_paired_crop_augmentation() -> None:
    geometry = torch.tensor([0.2, 0.3, 0.6, 0.8, 0.1, 0.1, -0.1, 0.2])
    transformed = transform_geometry_delta(
        geometry,
        horizontal_flip=True,
        vertical_flip=False,
        rotations=0,
    )
    expected_target = torch.tensor([0.4, 0.3, 0.8, 0.8])
    expected_prior = torch.tensor([0.3, 0.2, 0.9, 0.6])
    assert torch.allclose(transformed[:4], expected_target)
    assert torch.allclose(transformed[4:], expected_target - expected_prior)


def test_empty_geometry_stays_empty_after_augmentation() -> None:
    transformed = transform_geometry_delta(
        torch.zeros(8),
        horizontal_flip=True,
        vertical_flip=True,
        rotations=1,
    )
    assert torch.equal(transformed, torch.zeros(8))


def test_geometry_delta_follows_counterclockwise_rotation() -> None:
    geometry = torch.tensor([0.2, 0.3, 0.6, 0.8, 0.1, 0.1, -0.1, 0.2])
    transformed = transform_geometry_delta(
        geometry,
        horizontal_flip=False,
        vertical_flip=False,
        rotations=1,
    )
    expected_target = torch.tensor([0.3, 0.4, 0.8, 0.8])
    expected_prior = torch.tensor([0.2, 0.3, 0.6, 0.9])
    assert torch.allclose(transformed[:4], expected_target)
    assert torch.allclose(transformed[4:], expected_target - expected_prior)


def test_updater_training_smoke(tmp_path: Path) -> None:
    manifest = generate_updater_smoke_dataset(
        tmp_path / "data", sample_count=40, image_size=16, seed=5
    )
    output_dir = tmp_path / "run"
    archive_dir = tmp_path / "code_outputs" / "updater" / "smoke"
    config = {
        "seed": 5,
        "data": {"samples": str(manifest)},
        "model": {
            "base_channels": 4,
            "dropout": 0.0,
            "hierarchical_edit": True,
            "prior_guided_roi": True,
        },
        "training": {
            "device": "cpu",
            "epochs": 1,
            "patience": 1,
            "batch_size": 8,
            "gradient_accumulation_steps": 2,
            "num_workers": 0,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "roi_warmup_epochs": 1,
        },
        "loss": {
            "segmentation": 1.0,
            "edit": 1.0,
            "geometry": 0.5,
            "false_edit": 0.5,
            "confidence": 0.2,
        },
        "monitoring": {"progress_bar": False},
        "output_dir": str(output_dir),
        "archive_dir": str(archive_dir),
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    summary = train_updater(config_path)
    assert summary["epochs_completed"] == 1
    assert summary["history"][0]["trainable_scope"] == "roi_only"
    checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    assert checkpoint["model_config"]["prior_guided_roi"] is True
    assert (output_dir / "best.pt").is_file()
    assert (output_dir / "best_val_loss.pt").is_file()
    assert (output_dir / "best_quality.pt").is_file()
    assert (output_dir / "best_tradeoff.pt").is_file()
    assert (output_dir / "best_safety.pt").is_file()
    assert (output_dir / "last.pt").is_file()
    assert (output_dir / "history.jsonl").is_file()
    assert (output_dir / "history.csv").is_file()
    assert (output_dir / "state.json").is_file()
    assert (output_dir / "visualizations" / "epoch_0001.png").is_file()
    assert (output_dir / "curves" / "training_curves.png").is_file()
    assert (output_dir / "provenance" / "run_provenance.json").is_file()
    assert (output_dir / "provenance" / "updater_samples.jsonl").is_file()
    assert (archive_dir / "best_val_loss.pt").is_file()
    assert (archive_dir / "metrics.json").is_file()
    assert (archive_dir / "ARCHIVED_FROM.txt").read_text(encoding="utf-8").strip() == str(
        output_dir.resolve()
    )
    history_header = (output_dir / "history.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "learning_rate" in history_header
    assert "train/loss" in history_header
    assert "val/loss" in history_header
    predictor = UpdaterPredictor(output_dir / "best.pt", device="cpu")
    result = predictor.predict(torch.rand(3, 16, 16).numpy(), torch.zeros(1, 16, 16).numpy())
    assert result["mask_probability"].shape == (16, 16)
    assert result["edit_probabilities"].shape == (4,)
