from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load_acquisition_pilot():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "run_habitat_online_acquisition_pilot.py"
    )
    spec = importlib.util.spec_from_file_location("habitat_online_acquisition_pilot", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_acquisition_policies_obey_pre_observation_contract() -> None:
    module = load_acquisition_pilot()

    assert module.should_acquire("acquire_all", unknown_score=0, min_unknown_score=8)
    assert module.should_acquire("unknown_gate", unknown_score=8, min_unknown_score=8)
    assert not module.should_acquire("unknown_gate", unknown_score=7, min_unknown_score=8)
    assert module.should_acquire(
        "novelty_gate",
        unknown_score=99,
        min_unknown_score=8,
        novelty_score=8,
        min_novelty_score=8,
    )
    assert not module.should_acquire(
        "novelty_gate",
        unknown_score=99,
        min_unknown_score=8,
        novelty_score=7,
        min_novelty_score=8,
    )
    assert not module.should_acquire("stop_after_initial", unknown_score=99, min_unknown_score=8)
    with pytest.raises(ValueError, match="non-negative"):
        module.should_acquire("unknown_gate", unknown_score=8, min_unknown_score=-1)


def test_prospective_unknown_cells_excludes_already_committed_space() -> None:
    module = load_acquisition_pilot()
    committed = np.full((21, 21), 0.5, dtype=np.float32)
    committed[9:12, 9:12] = 0.0

    cells = module.prospective_unknown_cells(
        committed,
        x=0.0,
        z=0.0,
        yaw=0.0,
        x_min=-5.0,
        z_max=5.0,
        pixels_per_meter=2.0,
        max_range_m=2.0,
    )

    assert cells
    assert (10, 10) not in cells


def test_reference_map_quality_uses_only_reference_known_cells() -> None:
    module = load_acquisition_pilot()
    committed = np.array([[0.0, 0.5], [1.0, 0.0]], dtype=np.float32)
    reference = np.array([[0.0, 0.5], [1.0, 1.0]], dtype=np.float32)

    assert module.reference_map_quality(committed, reference) == pytest.approx(2.0 / 3.0)


def test_reference_map_metrics_separate_safety_and_coverage_errors() -> None:
    module = load_acquisition_pilot()
    committed = np.array([[1.0, 0.0], [0.5, 1.0]], dtype=np.float32)
    reference = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32)

    metrics = module.reference_map_metrics(committed, reference)

    assert metrics["final_reference_map_quality"] == pytest.approx(0.25)
    assert metrics["reference_known_coverage"] == pytest.approx(0.75)
    assert metrics["reference_unknown_rate"] == pytest.approx(0.25)
    assert metrics["occupied_iou"] == pytest.approx(1.0 / 3.0)
    assert metrics["free_iou"] == pytest.approx(0.0)
    assert metrics["balanced_iou"] == pytest.approx(1.0 / 6.0)
    assert metrics["false_free_rate"] == pytest.approx(0.5)
    assert metrics["false_obstacle_rate"] == pytest.approx(0.5)
