from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def load_training_module():
    pytest.importorskip("torch")
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_habitat_visual_value.py"
    spec = importlib.util.spec_from_file_location("habitat_visual_value_training", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_state_relative_targets_preserve_order_and_remove_state_scale() -> None:
    module = load_training_module()
    gains = [-0.5, 0.0, 1.0]

    targets = module.state_targets(gains, mode="state_relative_gain")

    assert np.isclose(targets.mean(), 0.0, atol=1e-6)
    assert np.isclose(targets.std(), 1.0, atol=1e-6)
    assert np.array_equal(np.argsort(targets), np.argsort(gains))


def test_absolute_gain_targets_are_unchanged() -> None:
    module = load_training_module()

    targets = module.state_targets([-0.5, 0.0, 1.0], mode="absolute_gain")

    assert np.allclose(targets, (-0.5, 0.0, 1.0))
