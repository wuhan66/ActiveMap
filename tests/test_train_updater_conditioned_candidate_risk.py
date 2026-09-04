import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "train_updater_conditioned_candidate_risk.py"
SPEC = importlib.util.spec_from_file_location("candidate_risk_training", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_calibration_freezes_recall_target_from_development_labels() -> None:
    labels = np.asarray([1, 1, 1, 0, 0, 0], dtype=np.float32)
    scores = np.asarray([0.9, 0.8, 0.4, 0.7, 0.2, 0.1], dtype=np.float32)
    result = MODULE.calibrate_threshold(labels, scores, target_false_edit_recall=2 / 3)
    assert result["risk_threshold"] == pytest.approx(0.8)
    assert result["false_edit_recall"] == 2 / 3
    assert result["development_false_edits"] == 3


def test_auroc_orders_risky_candidates_above_safe_candidates() -> None:
    assert MODULE._auroc(np.asarray([0, 0, 1, 1]), np.asarray([0.1, 0.3, 0.7, 0.9])) == 1.0
