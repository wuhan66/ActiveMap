from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.aggregate_tool_belief_spatial_seeds import aggregate


def _report(*, spatial: bool, f1: float) -> dict[str, Any]:
    final = {
        "macro_f1": f1,
        "false_edit_rate": 0.04,
        "missed_edit_rate": 0.30 if spatial else 0.32,
    }
    identity = {"macro_f1": 0.57, "false_edit_rate": 0.04, "missed_edit_rate": 0.41}
    return {
        "protocol": "spatial_hierarchical_decision_head_v1_independent_eval",
        "split": "val",
        "test_assets_read": False,
        "maximum_causal_future_mass": 0.0,
        "checkpoint_uses_spatial": spatial,
        "no_spatial_masking_invariance": True,
        "prediction_changes_when_masked": 2 if spatial else 0,
        "maximum_update_probability_change_when_masked": 0.2 if spatial else 0.0,
        "example_count": 552,
        "spatial_by_stage": {"3": final},
        "identity_by_stage": {"3": identity},
        "passed": True,
    }


def _write(path: Path, report: dict[str, Any]) -> Path:
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_aggregate_requires_positive_safe_paired_gain(tmp_path: Path) -> None:
    pairs = []
    for index, gain in enumerate((0.02, 0.03, 0.01), start=1):
        pairs.append(
            (
                index,
                _write(tmp_path / f"s{index}.json", _report(spatial=True, f1=0.61 + gain)),
                _write(tmp_path / f"n{index}.json", _report(spatial=False, f1=0.61)),
            )
        )
    result = aggregate(pairs, minimum_mean_gain=0.01, safety_margin=0.02)
    assert result["spatial_win_count"] == 3
    assert result["mean_macro_f1_gain"] == pytest.approx(0.02)
    assert result["passed"] is True


def test_aggregate_rejects_no_spatial_input_dependence(tmp_path: Path) -> None:
    no_spatial = _report(spatial=False, f1=0.60)
    no_spatial["prediction_changes_when_masked"] = 1
    pair = (
        1,
        _write(tmp_path / "spatial.json", _report(spatial=True, f1=0.62)),
        _write(tmp_path / "no_spatial.json", no_spatial),
    )
    with pytest.raises(ValueError, match="depend on spatial input"):
        aggregate([pair], minimum_mean_gain=0.01, safety_margin=0.02)
