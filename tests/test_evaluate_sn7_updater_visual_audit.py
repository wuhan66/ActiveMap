import numpy as np

from scripts.evaluate_sn7_updater_visual_audit import (
    aggregate_visual_rows,
    binary_iou,
    operation_slices,
    predicted_edit_from_change,
)


def test_predicted_edit_from_change_recovers_all_operations() -> None:
    valid = np.ones((4, 4), dtype=bool)
    prior = np.zeros((4, 4), dtype=bool)
    prior[1:3, 1:3] = True

    assert predicted_edit_from_change(prior, np.zeros_like(prior), valid) == "KEEP"
    added = np.zeros_like(prior)
    added[0, 0] = True
    assert predicted_edit_from_change(prior, added, valid) == "ADD"
    removed = np.zeros_like(prior)
    removed[1, 1] = True
    assert predicted_edit_from_change(prior, removed, valid) == "DELETE"
    assert predicted_edit_from_change(prior, added | removed, valid) == "RESHAPE"


def test_binary_iou_and_operation_slices() -> None:
    valid = np.ones((2, 2), dtype=bool)
    assert binary_iou(valid, valid, valid) == 1.0
    rows = [
        {
            "target_edit": "ADD",
            "committed_map_iou": 0.8,
            "map_iou_delta": 0.2,
            "change_iou": 0.6,
            "operation_correct": True,
        },
        {
            "target_edit": "ADD",
            "committed_map_iou": 0.6,
            "map_iou_delta": 0.0,
            "change_iou": 0.4,
            "operation_correct": False,
        },
    ]
    summary = operation_slices(rows)["ADD"]
    assert summary["sample_count"] == 2
    assert summary["committed_map_iou"] == 0.7
    assert summary["operation_accuracy"] == 0.5
    aggregate = aggregate_visual_rows(
        [
            {
                **rows[0],
                "prior_map_iou": 0.5,
                "false_edit": False,
                "missed_edit": False,
                "wrong_edit": False,
            }
        ]
    )
    assert aggregate["false_edit_rate"] is None
    assert aggregate["missed_edit_rate"] == 0.0
