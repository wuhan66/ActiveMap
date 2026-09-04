from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.compare_temporal_calibrations import compare


def _report(*, harmonic: float, samples: str = "/data/samples.jsonl") -> dict[str, Any]:
    return {
        "split": "val",
        "samples": samples,
        "sample_count": 140,
        "max_stable_false_positive": 0.005,
        "constraint_satisfied": True,
        "checkpoint": "/run/best.pt",
        "checkpoint_epoch": 10,
        "add_sweep": [{}, {}, {}],
        "remove_sweep": [{}, {}, {}],
        "selected": {
            "add_threshold": 0.3,
            "remove_threshold": 0.6,
            "add": {
                "mean_positive_iou": harmonic,
                "stable_false_positive_fraction": 0.001,
            },
            "remove": {
                "mean_positive_iou": harmonic,
                "stable_false_positive_fraction": 0.002,
            },
            "harmonic_mean_positive_iou": harmonic,
        },
    }


def _write(path: Path, report: dict[str, Any]) -> Path:
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_compare_reports_paired_calibrated_delta(tmp_path: Path) -> None:
    scratch = _write(tmp_path / "scratch.json", _report(harmonic=0.08))
    transfer = _write(tmp_path / "transfer.json", _report(harmonic=0.10))
    result = compare([("scratch", scratch), ("transfer", transfer)])
    assert result["all_constraints_satisfied"] is True
    assert result["deltas"]["transfer"]["delta_harmonic_iou"] == pytest.approx(0.02)
    assert result["test_assets_read"] is False


def test_compare_rejects_mixed_sample_protocols(tmp_path: Path) -> None:
    scratch = _write(tmp_path / "scratch.json", _report(harmonic=0.08))
    transfer = _write(
        tmp_path / "transfer.json",
        _report(harmonic=0.10, samples="/data/other.jsonl"),
    )
    with pytest.raises(ValueError, match="protocol differs"):
        compare([("scratch", scratch), ("transfer", transfer)])
