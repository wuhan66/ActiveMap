from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


def _load_script() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "compare_updater_runs.py"
    spec = importlib.util.spec_from_file_location("compare_updater_runs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paired_epoch_deltas_align_by_epoch() -> None:
    module = _load_script()
    reference: list[dict[str, Any]] = [
        {"epoch": 1, "val": {"loss": 2.0, "iou": 0.5}},
        {"epoch": 2, "val": {"loss": 1.0, "iou": 0.6}},
    ]
    candidate: list[dict[str, Any]] = [
        {"epoch": 2, "val": {"loss": 1.2, "iou": 0.65}}
    ]
    for records in (reference, candidate):
        for record in records:
            record["val"].update(
                {
                    "edit_accuracy": 0.8,
                    "false_edit_rate": 0.1,
                    "missed_edit_rate": 0.2,
                    "delete_recall": 0.3,
                    "loss_confidence": 0.4,
                    "added_change_iou": 0.2,
                    "removed_change_iou": 0.4,
                }
            )
    deltas = module.paired_epoch_deltas(reference, candidate)
    assert len(deltas) == 1
    assert deltas[0]["epoch"] == 2
    assert deltas[0]["delta_val_loss"] == pytest.approx(0.2)
    assert deltas[0]["delta_val_iou"] == pytest.approx(0.05)
    assert deltas[0]["delta_val_temporal_change_harmonic_iou"] == pytest.approx(0.0)


def test_summary_selects_best_temporal_harmonic_epoch() -> None:
    module = _load_script()
    records: list[dict[str, Any]] = [
        {
            "epoch": 1,
            "val": {
                "loss": 1.0,
                "iou": 0.8,
                "edit_accuracy": 0.7,
                "false_edit_rate": 0.1,
                "missed_edit_rate": 0.2,
                "delete_recall": 0.3,
                "added_change_iou": 0.1,
                "removed_change_iou": 0.3,
            },
        },
        {
            "epoch": 2,
            "val": {
                "loss": 1.2,
                "iou": 0.7,
                "edit_accuracy": 0.8,
                "false_edit_rate": 0.05,
                "missed_edit_rate": 0.1,
                "delete_recall": 0.4,
                "added_change_iou": 0.2,
                "removed_change_iou": 0.4,
            },
        },
    ]
    summary = module.summarize_history(records)
    assert summary["best_val_loss_epoch"] == 1
    assert summary["best_temporal_change_harmonic_iou_epoch"] == 2
    assert summary["best_temporal_change_harmonic_iou"] == pytest.approx(0.2666667)
