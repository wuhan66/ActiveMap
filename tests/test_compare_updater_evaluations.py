from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _load_script() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "compare_updater_evaluations.py"
    spec = importlib.util.spec_from_file_location("compare_updater_evaluations", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _evaluation(offset: float) -> dict[str, object]:
    return {
        "macro_f1": 0.8 + offset,
        "edit_accuracy": 0.9,
        "false_edit_rate": 0.04,
        "missed_update_rate": 0.1,
        "mean_raster_iou": 0.7,
        "mean_polygon_iou": 0.6,
        "topology_valid_rate": 0.99,
        "ece": 0.12 - offset,
        "brier": 0.1,
        "nll": 0.3,
        "aurc": 0.05 - offset,
        "per_edit": {"DELETE": {"f1": 0.4 + offset}},
    }


def test_compare_evaluations_reports_candidate_minus_reference() -> None:
    module = _load_script()
    report = module.compare_evaluations(_evaluation(0.0), _evaluation(0.02))
    deltas = report["candidate_minus_reference"]
    assert deltas["macro_f1"] == pytest.approx(0.02)
    assert deltas["delete_f1"] == pytest.approx(0.02)
    assert deltas["ece"] == pytest.approx(-0.02)
    assert deltas["aurc"] == pytest.approx(-0.02)
