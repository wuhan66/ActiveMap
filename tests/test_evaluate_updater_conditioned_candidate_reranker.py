import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_updater_conditioned_candidate_reranker import choose_candidate, calibrate


def test_choose_candidate_reorders_to_lower_risk_candidate():
    assert choose_candidate(np.array([2.0, 1.8]), 0.0, np.array([0.9, 0.1]), risk_penalty=1.0, max_risk=1.0) == 1


def test_choose_candidate_stops_when_every_candidate_exceeds_cap():
    assert choose_candidate(np.array([2.0, 1.8]), 0.0, np.array([0.9, 0.8]), risk_penalty=0.0, max_risk=0.5) is None


def test_calibration_uses_feasible_candidate_only():
    rows = [{"candidate_logits": [2.0, 1.8], "stop_logit": 0.0, "risks": [0.9, 0.1], "candidates": [{"utility": 1.0, "raster_iou": 0.2, "false_edit": 1.0, "missed_edit": 0.0, "cost": 1.0}, {"utility": 0.8, "raster_iou": 0.3, "false_edit": 0.0, "missed_edit": 0.0, "cost": 1.0}], "stop": {"utility": 0.0, "raster_iou": 0.1, "false_edit": 0.0, "missed_edit": 1.0, "cost": 0.0}}]
    result = calibrate(rows, stale={"utility": 0.0, "raster_iou": 0.1, "false_edit": 0.0})
    assert result["status"] == "complete"
    assert result["selected"]["metrics"]["false_edit"] == 0.0
