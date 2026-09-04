from types import SimpleNamespace

import numpy as np
import pytest
from affine import Affine

from activemap.agent.writeback import (
    EvidenceMaskPrediction,
    apply_typed_mask_edit,
    effective_operation_from_masks,
    evaluate_typed_writeback,
    regularize_typed_delta,
)
from activemap.models import EditOperation
from scripts.evaluate_agent_map_writeback import _temporal_pair_input


def test_temporal_writeback_uses_the_pinned_six_channel_contract():
    temporal_updater = SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(temporal_pair_input=True, image_channels=6)
        )
    )
    single_frame_updater = SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(temporal_pair_input=False, image_channels=3)
        )
    )

    assert _temporal_pair_input(temporal_updater) is True
    assert _temporal_pair_input(single_frame_updater) is False


def test_temporal_writeback_rejects_an_invalid_channel_contract():
    invalid_updater = SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(temporal_pair_input=True, image_channels=3)
        )
    )

    with pytest.raises(ValueError, match="six-channel"):
        _temporal_pair_input(invalid_updater)


def test_typed_mask_edit_enforces_operation_monotonicity():
    prior = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    target = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    added = apply_typed_mask_edit(prior, target, EditOperation.ADD)
    deleted = apply_typed_mask_edit(prior, target, EditOperation.DELETE)

    assert np.all(added >= prior)
    assert np.all(deleted <= prior)


def test_vector_delta_replay_matches_committed_mask():
    prior = np.zeros((8, 8), dtype=np.float32)
    target = prior.copy()
    target[2:6, 3:5] = 1.0
    result = evaluate_typed_writeback(
        [EvidenceMaskPrediction("e1", target, 0.9)],
        operation=EditOperation.ADD,
        prior=prior,
        target=target,
        valid=np.ones_like(prior),
        transform=Affine.identity(),
    )

    assert result["raster_iou"] == 1.0
    assert result["added_polygon_iou"] == 1.0
    assert result["vector_replay_iou"] == 1.0
    assert result["vector_delta_topology_valid"] is True


def test_confidence_weighted_fusion_uses_all_selected_evidence():
    prior = np.zeros((4, 4), dtype=np.float32)
    strong = np.ones((4, 4), dtype=np.float32)
    weak = np.zeros((4, 4), dtype=np.float32)
    result = evaluate_typed_writeback(
        [
            EvidenceMaskPrediction("strong", strong, 0.9),
            EvidenceMaskPrediction("weak", weak, 0.1),
        ],
        operation=EditOperation.ADD,
        prior=prior,
        target=strong,
        valid=np.ones_like(prior),
        transform=Affine.identity(),
    )

    assert result["selected_evidence_ids"] == ["strong", "weak"]
    assert result["raster_iou"] == 1.0


def test_counterfactual_aligned_reshape_avoids_harmful_initial_mask_fusion():
    prior = np.asarray([[0.0, 1.0]], dtype=np.float32)
    target = np.asarray([[1.0, 0.0]], dtype=np.float32)
    valid = np.ones_like(prior)
    initial = EvidenceMaskPrediction(
        evidence_id="initial",
        target_probability=prior.copy(),
        confidence=1.0,
    )
    acquired = EvidenceMaskPrediction(
        evidence_id="acquired",
        target_probability=target.copy(),
        confidence=1.0,
    )

    fused = evaluate_typed_writeback(
        [initial, acquired],
        operation=EditOperation.RESHAPE,
        prior=prior,
        target=target,
        valid=valid,
        transform=Affine.identity(),
    )
    aligned = evaluate_typed_writeback(
        [acquired],
        operation=EditOperation.RESHAPE,
        prior=prior,
        target=target,
        valid=valid,
        transform=Affine.identity(),
    )

    assert fused["raster_iou_gain"] < aligned["raster_iou_gain"]
    assert aligned["raster_iou_gain"] == pytest.approx(1.0)
    assert aligned["selected_evidence_ids"] == ["acquired"]


def test_max_confidence_fusion_routes_one_selected_evidence():
    prior = np.zeros((4, 4), dtype=np.float32)
    strong = np.ones((4, 4), dtype=np.float32)
    weak = np.zeros((4, 4), dtype=np.float32)
    result = evaluate_typed_writeback(
        [
            EvidenceMaskPrediction("weak", weak, 0.4),
            EvidenceMaskPrediction("strong", strong, 0.8),
        ],
        operation=EditOperation.ADD,
        prior=prior,
        target=strong,
        valid=np.ones_like(prior),
        transform=Affine.identity(),
        evidence_fusion="max_confidence",
    )

    assert result["fusion_weights"] == [0.0, 1.0]
    assert result["fused_confidence"] == 0.8
    assert result["evidence_fusion"] == "max_confidence"
    assert result["raster_iou"] == 1.0


def test_writeback_can_retain_masks_for_official_graph_export():
    prior = np.zeros((4, 4), dtype=np.float32)
    target = prior.copy()
    target[1:3, 1:3] = 1.0
    result = evaluate_typed_writeback(
        [EvidenceMaskPrediction("e1", target, 1.0)],
        operation=EditOperation.ADD,
        prior=prior,
        target=target,
        valid=np.ones_like(prior),
        transform=Affine.identity(),
        return_artifacts=True,
    )

    assert np.array_equal(result["_committed_mask"], target)
    assert np.array_equal(result["_target_mask"], target)


def test_delta_regularization_removes_speckle_without_changing_prior():
    prior = np.zeros((8, 8), dtype=np.float32)
    prior[1:4, 1:4] = 1.0
    raw = prior.copy()
    raw[5:7, 5:7] = 1.0
    raw[0, 7] = 1.0

    regularized, diagnostics = regularize_typed_delta(
        raw,
        prior,
        EditOperation.ADD,
        min_component_pixels=2,
    )

    assert np.all(regularized >= prior)
    assert regularized[5:7, 5:7].all()
    assert not regularized[0, 7]
    assert diagnostics["raw_add_component_count"] == 2
    assert diagnostics["retained_add_component_count"] == 1


def test_delta_regularization_preserves_largest_when_threshold_removes_all():
    prior = np.zeros((8, 8), dtype=np.float32)
    raw = prior.copy()
    raw[1:3, 1:3] = 1.0
    raw[6, 6] = 1.0

    regularized, diagnostics = regularize_typed_delta(
        raw,
        prior,
        EditOperation.ADD,
        min_component_pixels=16,
        preserve_largest=True,
    )

    assert regularized[1:3, 1:3].all()
    assert not regularized[6, 6]
    assert diagnostics["retained_add_component_count"] == 1


def test_writeback_reports_validation_selected_delta_filter():
    prior = np.zeros((8, 8), dtype=np.float32)
    target = prior.copy()
    target[2:6, 2:6] = 1.0
    noisy = target.copy()
    noisy[0, 7] = 1.0

    result = evaluate_typed_writeback(
        [EvidenceMaskPrediction("e1", noisy, 1.0)],
        operation=EditOperation.ADD,
        prior=prior,
        target=target,
        valid=np.ones_like(prior),
        transform=Affine.identity(),
        min_delta_component_pixels=2,
    )

    assert result["raster_iou"] == 1.0
    assert result["min_delta_component_pixels"] == 2
    assert result["raw_add_component_count"] == 2
    assert result["retained_add_component_count"] == 1


def test_delta_margin_preserves_uncertain_pixels_from_prior():
    prior = np.zeros((4, 4), dtype=np.float32)
    uncertain = np.full((4, 4), 0.6, dtype=np.float32)
    result = evaluate_typed_writeback(
        [EvidenceMaskPrediction("e1", uncertain, 0.9)],
        operation=EditOperation.ADD,
        prior=prior,
        target=prior,
        valid=np.ones_like(prior),
        transform=Affine.identity(),
        delta_margin=0.2,
    )

    assert result["effective_operation"] == "KEEP"
    assert result["writeback_changed"] is False
    assert result["raster_iou"] == 1.0


def test_writeback_reports_when_component_filter_removes_a_raw_edit():
    prior = np.zeros((4, 4), dtype=np.float32)
    target = prior.copy()
    target[1, 1] = 1.0

    result = evaluate_typed_writeback(
        [EvidenceMaskPrediction("e1", target, 1.0)],
        operation=EditOperation.ADD,
        prior=prior,
        target=target,
        valid=np.ones_like(prior),
        transform=Affine.identity(),
        min_delta_component_pixels=2,
    )

    assert result["raw_effective_operation"] == EditOperation.ADD.value
    assert result["raw_writeback_changed"] is True
    assert result["raw_added_pixels"] == 1
    assert result["raw_removed_pixels"] == 0
    assert result["effective_operation"] == EditOperation.KEEP.value
    assert result["regularization_removed_all_delta"] is True


def test_confidence_gate_can_veto_executable_edit():
    prior = np.zeros((4, 4), dtype=np.float32)
    target = np.ones((4, 4), dtype=np.float32)
    result = evaluate_typed_writeback(
        [EvidenceMaskPrediction("e1", target, 0.4)],
        operation=EditOperation.ADD,
        prior=prior,
        target=target,
        valid=np.ones_like(prior),
        transform=Affine.identity(),
        confidence_floor=0.5,
    )

    assert result["confidence_gate_passed"] is False
    assert result["effective_operation"] == "KEEP"


def test_effective_operation_is_derived_from_actual_delta_direction():
    prior = np.asarray([[1, 0], [1, 0]], dtype=np.float32)
    reshaped = np.asarray([[0, 1], [1, 0]], dtype=np.float32)

    assert (
        effective_operation_from_masks(prior, reshaped, np.ones_like(prior))
        == EditOperation.RESHAPE
    )
