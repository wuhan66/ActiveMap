from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from activemap.evaluation.updater_promotion import (
    finalize_hierarchical_updater,
    select_promoted_checkpoint,
)


def _candidate(name: str, macro: float, delete: float, false_edit: float) -> dict[str, Any]:
    return {
        "checkpoint_name": name,
        "constraint_satisfied": True,
        "selected": {
            "macro_f1": macro,
            "delete_f1": delete,
            "false_edit_rate": false_edit,
        },
    }


def test_promotion_requires_both_metrics_and_safety() -> None:
    candidates = [
        _candidate("macro_only", 0.81, 0.35, 0.04),
        _candidate("delete_only", 0.78, 0.40, 0.04),
        _candidate("unsafe", 0.82, 0.41, 0.06),
    ]
    assert (
        select_promoted_checkpoint(
            candidates,
            baseline_macro_f1=0.79,
            baseline_delete_f1=0.36,
            max_false_edit=0.05,
        )
        is None
    )


def test_promotion_selects_highest_macro_then_delete_f1() -> None:
    candidates = [
        _candidate("first", 0.80, 0.39, 0.04),
        _candidate("winner", 0.81, 0.38, 0.05),
        _candidate("tie_loser", 0.81, 0.37, 0.03),
    ]
    promoted = select_promoted_checkpoint(
        candidates,
        baseline_macro_f1=0.79,
        baseline_delete_f1=0.36,
        max_false_edit=0.05,
    )
    assert promoted is not None
    assert promoted["checkpoint_name"] == "winner"


def test_validation_only_finalization_evaluates_validation_but_not_test(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for name in ("best_quality", "best_safety", "best_val_loss"):
        (tmp_path / f"{name}.pt").touch()

    def fake_calibration(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "constraint_satisfied": True,
            "selected": {
                "macro_f1": 0.81,
                "delete_f1": 0.40,
                "false_edit_rate": 0.04,
                "presence_threshold": 0.3,
                "change_threshold": 0.8,
            },
        }

    monkeypatch.setitem(
        sys.modules,
        "activemap.evaluation.updater_calibration",
        SimpleNamespace(calibrate_updater_hierarchy=fake_calibration),
    )
    evaluation_calls: list[dict[str, Any]] = []

    def fake_evaluation(*args: Any, **kwargs: Any) -> dict[str, Any]:
        evaluation_calls.append(kwargs)
        return {"split": kwargs["split"], "ece": 0.1, "aurc": 0.05}

    monkeypatch.setitem(
        sys.modules,
        "activemap.evaluation.updater",
        SimpleNamespace(evaluate_updater_checkpoint=fake_evaluation),
    )
    decision = finalize_hierarchical_updater(
        tmp_path,
        tmp_path / "samples.jsonl",
        baseline_macro_f1=0.79,
        baseline_delete_f1=0.36,
        evaluate_test=False,
    )
    assert decision["status"] == "promoted_validation_only"
    assert decision["test_enabled"] is False
    assert decision["promoted_checkpoint"] == "best_quality"
    assert decision["validation_evaluation"] == {
        "split": "val",
        "ece": 0.1,
        "aurc": 0.05,
    }
    assert decision["test_evaluation"] is None
    assert [call["split"] for call in evaluation_calls] == ["val"]
