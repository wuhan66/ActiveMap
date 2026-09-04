import numpy as np

from activemap.updater_records import UpdaterSample
from scripts.evaluate_muno21_model_segmentation_tool import (
    _group_id,
    grouped_bootstrap_delta,
    promotion_decision,
    score_masks,
)


def test_score_masks_reports_map_gain_and_component_improvement() -> None:
    prior = np.zeros((6, 6), dtype=bool)
    prior[1:3, 1:3] = True
    target = prior.copy()
    target[3:5, 3:5] = True
    prediction = target.copy()
    metrics = score_masks(prior, prediction, target, np.ones_like(prior))

    assert metrics["prediction_target_iou"] == 1.0
    assert metrics["iou_delta"] == 0.5
    assert metrics["change_iou"] == 1.0
    assert metrics["prediction_component_error"] == 0
    assert metrics["prior_component_error"] == 1


def test_grouped_bootstrap_is_deterministic_and_preserves_positive_delta() -> None:
    rows = [
        {"group_id": "a", "iou_delta": 0.1},
        {"group_id": "a", "iou_delta": 0.2},
        {"group_id": "b", "iou_delta": 0.3},
    ]
    first = grouped_bootstrap_delta(rows, seed=7, samples=500)
    second = grouped_bootstrap_delta(rows, seed=7, samples=500)

    assert first == second
    assert first["ci95_low"] > 0.0
    assert first["group_count"] == 2


def test_promotion_requires_every_frozen_gate() -> None:
    rows = []
    for edit in ("KEEP", "ADD", "DELETE", "RESHAPE"):
        rows.append(
            {
                "group_id": edit,
                "edit_type": edit,
                "success": True,
                "prior_target_iou": 0.7,
                "prediction_target_iou": 0.8,
                "iou_delta": 0.1,
                "change_iou": 0.5,
                "added_change_iou": 0.5,
                "removed_change_iou": 0.5,
                "predicted_change_fraction": 0.005,
                "target_change_fraction": 0.01,
                "prior_component_error": 2,
                "prediction_component_error": 1,
            }
        )
    ci = {"mean": 0.1, "ci95_low": 0.05, "ci95_high": 0.15}
    decision = promotion_decision(rows, ci, ci)
    assert decision["promoted_for_agent_registration"]

    rows[0]["predicted_change_fraction"] = 0.02
    rejected = promotion_decision(rows, ci, ci)
    assert not rejected["promoted_for_agent_registration"]
    assert not rejected["gates"]["keep_false_change_at_most_0_01"]


def test_group_id_binds_overlapping_patches_to_source_annotation() -> None:
    sample = UpdaterSample(
        sample_id="muno21-000093-p02-chicago-add",
        aoi_id="chicago",
        split="val",
        image_path="image.npy",
        prior_mask_path="prior.npy",
        target_mask_path="target.npy",
        edit_type="ADD",
        geometry_delta=[0.0] * 8,
        object_id="muno21-scenario-93-patch-2",
        source_metadata={"annotation_index": 93, "patch_index": 2},
    )

    assert _group_id(sample) == "chicago:annotation:93"
