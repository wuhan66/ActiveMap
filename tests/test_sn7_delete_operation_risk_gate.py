from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.screen_sn7_delete_operation_risk_gate import (
    FEATURE_NAMES,
    _auroc,
    collect_delete_rows,
    observable_features,
    score_gate,
    select_threshold,
)


def _row(**overrides: object) -> dict:
    base = {
        "fused_confidence": 0.8,
        "budget": 1.5,
        "spent_cost": 0.4,
        "semantic_tool_called": True,
        "fusion_weights": [0.4, 0.6],
        "predicted_component_count": 2,
        "prior_component_count": 1,
        "raw_add_component_count": 0,
        "raw_remove_component_count": 2,
        "retained_add_component_count": 0,
        "retained_remove_component_count": 1,
        "vector_replay_iou": 1.0,
        "topology_quality_before": 1.0,
        "topology_quality_after": 1.0,
        "vector_delta_topology_valid": True,
        "target": "COMMIT:DELETE",
        "raster_iou_gain": 0.2,
    }
    return {**base, **overrides}


def _writeback_row(**overrides: object) -> dict:
    return {
        **_row(),
        "task_id": "task-1",
        "aoi_id": "aoi-1",
        "effective_operation": "DELETE",
        "writeback_changed": True,
        "prior_raster_iou": 0.4,
        "raster_iou": 0.6,
        "split": "train",
        "test_assets_read": False,
        **overrides,
    }


def _write_jsonl(path: Path, row: dict) -> None:
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_observable_features_have_fixed_schema_without_target_leakage() -> None:
    values = observable_features(_row())
    assert len(values) == len(FEATURE_NAMES)
    assert values[0] == pytest.approx(0.8)
    assert values[4] == 2.0


def test_threshold_selection_respects_both_safety_caps() -> None:
    support = {
        "beneficial": np.asarray([True, True, False, False]),
        "harmful": np.asarray([False, False, True, True]),
        "target_keep": np.asarray([False, False, True, False]),
        "aoi_ids": np.asarray(["a", "a", "a", "a"], dtype=object),
    }
    scores = np.asarray([0.95, 0.75, 0.80, 0.20])
    selected = select_threshold(
        scores,
        support,
        maximum_harmful_accept_rate=0.0,
        maximum_false_edit_rate=0.0,
        threshold_count=101,
    )
    assert selected["threshold"] == pytest.approx(0.81)
    assert selected["beneficial_delete_recall"] == pytest.approx(0.5)
    assert selected["harmful_accept_rate"] == 0.0


def test_collect_delete_rows_streams_validated_records(tmp_path: Path) -> None:
    direct = tmp_path / "direct.jsonl"
    selected = tmp_path / "selected.jsonl"
    _write_jsonl(direct, _writeback_row())
    _write_jsonl(
        selected,
        _writeback_row(task_id="task-2", target="REJECT", raster_iou_gain=-0.1),
    )

    collected = collect_delete_rows(
        (("direct", direct), ("selected", selected)), split="train", gain_epsilon=1e-6
    )

    assert collected["features"].shape == (2, len(FEATURE_NAMES))
    assert collected["beneficial"].tolist() == [True, False]
    assert collected["harmful"].tolist() == [False, True]


def test_gate_score_and_auroc_are_well_defined() -> None:
    support = {
        "beneficial": np.asarray([True, False]),
        "harmful": np.asarray([False, True]),
        "target_keep": np.asarray([False, True]),
        "aoi_ids": np.asarray(["a", "b"], dtype=object),
    }
    score = score_gate(np.asarray([0.9, 0.1]), support, 0.5)
    assert score["beneficial_delete_recall"] == 1.0
    assert score["harmful_accept_rate"] == 0.0
    assert _auroc(np.asarray([0.9, 0.1]), np.asarray([True, False])) == 1.0
