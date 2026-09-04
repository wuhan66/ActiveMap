import json
from pathlib import Path

import pytest

from scripts.aggregate_tool_belief_decision_seeds import aggregate


def _report(macro_f1: float, *, test_assets_read: bool = False) -> dict:
    identity = {
        "accuracy": 0.70,
        "macro_f1": 0.50,
        "false_edit_rate": 0.05,
        "missed_edit_rate": 0.40,
        "expected_calibration_error": 0.20,
    }

    def metrics(value: float) -> dict:
        return {
            "accuracy": 0.72,
            "macro_f1": value,
            "false_edit_rate": 0.06,
            "missed_edit_rate": 0.39,
            "expected_calibration_error": 0.15,
        }

    return {
        "protocol": {
            "schema_version": "tool-belief-decision-head-eval-v1",
            "split": "val",
            "sequence_count": 10,
            "checkpoint_epoch": 3,
            "frozen_decision_threshold": 0.6,
            "test_assets_read": test_assets_read,
        },
        "summaries": {
            "identity": identity,
            "residual_argmax": metrics(0.49),
            "hierarchical": metrics(macro_f1),
            "no_quality": metrics(macro_f1 - 0.02),
            "no_anchor": metrics(macro_f1 - 0.10),
            "no_current": metrics(macro_f1 - 0.05),
        },
        "gates": {
            "thresholds": {
                "minimum_macro_f1_gain": 0.01,
                "max_false_edit_increase": 0.02,
                "max_missed_edit_increase": 0.02,
            },
            "checks": {},
            "passed": True,
        },
    }


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_aggregate_reports_sample_statistics(tmp_path: Path) -> None:
    runs = [
        ("1", _write(tmp_path / "one.json", _report(0.52))),
        ("2", _write(tmp_path / "two.json", _report(0.54))),
        ("3", _write(tmp_path / "three.json", _report(0.56))),
    ]

    report, rows = aggregate(runs)

    assert report["protocol"]["seed_count"] == 3
    assert report["method_statistics"]["hierarchical"]["macro_f1"]["mean"] == pytest.approx(0.54)
    assert report["method_statistics"]["hierarchical"]["macro_f1"]["sample_std"] == pytest.approx(
        0.02
    )
    assert report["hierarchical_minus_identity"]["macro_f1"]["mean"] == pytest.approx(0.04)
    assert report["ablation_contributions"]["no_anchor"]["mean"] == pytest.approx(0.10)
    assert report["gate_summary"]["all_passed"] is True
    assert len(rows) == 3


def test_aggregate_rejects_test_read_or_changed_baseline(tmp_path: Path) -> None:
    clean = _report(0.52)
    test_read = _report(0.54, test_assets_read=True)
    with pytest.raises(ValueError, match="test_assets_read"):
        aggregate(
            [
                ("1", _write(tmp_path / "clean.json", clean)),
                ("2", _write(tmp_path / "test.json", test_read)),
            ]
        )

    changed = _report(0.54)
    changed["summaries"]["identity"]["macro_f1"] = 0.51
    with pytest.raises(ValueError, match="identity macro_f1"):
        aggregate(
            [
                ("1", _write(tmp_path / "clean-again.json", clean)),
                ("2", _write(tmp_path / "changed.json", changed)),
            ]
        )
