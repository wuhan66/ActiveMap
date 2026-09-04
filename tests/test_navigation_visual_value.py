from __future__ import annotations

import numpy as np
import pytest

from activemap.data.navigation_map import NavigationEvidence, NavigationPose
from activemap.evaluation.navigation_rollout import NavigationState


def load_visual_value_module():
    torch = pytest.importorskip("torch")
    from activemap.nn.navigation_visual_value import (
        NavigationVisualValueConfig,
        NavigationVisualValueNet,
        candidate_features,
        occupancy_array,
        select_prior_anchored_candidate,
    )

    return (
        torch,
        NavigationVisualValueConfig,
        NavigationVisualValueNet,
        candidate_features,
        occupancy_array,
        select_prior_anchored_candidate,
    )


def test_visual_value_network_scores_one_candidate_per_observable_state() -> None:
    torch, config_class, model_class, _, _, _ = load_visual_value_module()
    model = model_class(config_class(image_size=32, base_channels=8, hidden_dim=16))
    scores = model(
        torch.zeros((3, 4, 32, 32)),
        torch.full((3, 1, 32, 32), 0.5),
        torch.zeros((3, 6)),
    )

    assert scores.shape == (3,)
    assert model.parameter_count() > 0


def test_candidate_features_use_only_map_state_and_declared_geometry() -> None:
    torch, _, _, candidate_features, _, _ = load_visual_value_module()
    state = NavigationState(
        committed_map=torch.full((8, 8), 0.5).numpy(),
        pose=NavigationPose(x=0.0, y=0.0, yaw=0.0),
        remaining_budget=2.0,
        acquired_evidence_ids=(),
        step=0,
        pixels_per_meter=1.0,
    )
    evidence = NavigationEvidence(
        evidence_id="candidate",
        modality="camera",
        path="occupancy.npy",
        cost=2.0,
        timestamp=1,
        pose=NavigationPose(x=1.0, y=0.0, yaw=0.5),
        footprint_radius_pixels=1,
    )

    features = candidate_features(state, evidence)

    assert features.shape == (6,)
    assert features[0] == 1.0
    assert features[2] == 1.0


def test_occupancy_array_is_writable_for_zero_copy_torch_conversion() -> None:
    torch, _, _, _, occupancy_array, _ = load_visual_value_module()
    committed_map = torch.full((8, 8), 0.5).numpy()

    occupancy = occupancy_array(committed_map, image_size=32)

    assert occupancy.shape == (1, 32, 32)
    assert occupancy.flags.writeable


def test_prior_anchor_requires_a_strict_visual_advantage() -> None:
    _, _, _, _, _, select_candidate = load_visual_value_module()
    candidates = tuple(
        NavigationEvidence(
            evidence_id=evidence_id,
            modality="camera",
            path=f"{evidence_id}.npy",
            cost=1.0,
            timestamp=index,
        )
        for index, evidence_id in enumerate(("baseline", "visual", "other"))
    )
    scores = np.asarray((0.40, 0.55, 0.10), dtype=np.float32)

    assert (
        select_candidate(
            candidates,
            scores,
            baseline_evidence_id="baseline",
            override_margin=0.10,
        )
        == "visual"
    )
    assert (
        select_candidate(
            candidates,
            scores,
            baseline_evidence_id="baseline",
            override_margin=0.20,
        )
        == "baseline"
    )
