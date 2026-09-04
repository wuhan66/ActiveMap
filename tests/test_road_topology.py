import numpy as np

from activemap.evaluation.road_topology import road_connectivity_metrics


def test_identical_road_masks_have_perfect_connectivity() -> None:
    target = np.zeros((16, 16), dtype=bool)
    target[8, 2:14] = True
    metrics = road_connectivity_metrics(target, target, tolerance_pixels=1)
    assert metrics["connectivity_f1"] == 1.0
    assert metrics["component_count_error"] == 0


def test_fragmentation_reduces_connectivity_recall() -> None:
    target = np.zeros((16, 16), dtype=bool)
    target[8, 2:14] = True
    prediction = target.copy()
    prediction[8, 7:10] = False
    metrics = road_connectivity_metrics(prediction, target, tolerance_pixels=0)
    assert metrics["connectivity_recall"] < 0.75
    assert metrics["connectivity_precision"] == 1.0
    assert metrics["component_count_error"] == 1


def test_false_bridge_reduces_connectivity_precision() -> None:
    target = np.zeros((16, 16), dtype=bool)
    target[5, 2:14] = True
    target[10, 2:14] = True
    prediction = target.copy()
    prediction[5:11, 8] = True
    metrics = road_connectivity_metrics(prediction, target, tolerance_pixels=0)
    assert metrics["connectivity_precision"] < 0.75
    assert metrics["connectivity_recall"] == 1.0
    assert metrics["component_count_error"] == -1
