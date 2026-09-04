from __future__ import annotations

import pytest

from scripts.audit_sn7_delete_confidence_separability import (
    _auc,
    audit,
    compact_delete_rows,
    parse_record,
)


def _row(task: str, *, target: str, confidence: float, gain: float) -> dict:
    return {
        "task_id": task,
        "aoi_id": "aoi-a",
        "budget": 1.5,
        "target": target,
        "effective_operation": "DELETE",
        "writeback_changed": True,
        "fused_confidence": confidence,
        "raster_iou_gain": gain,
    }


def test_auc_respects_ties() -> None:
    assert _auc([0.9, 0.8, 0.8], [True, False, False]) == 1.0
    assert _auc([0.5, 0.5], [True, False]) == 0.5
    assert _auc([0.1, 0.9], [True, False]) == 0.0


def test_delete_audit_reports_constrained_recall() -> None:
    policy = {
        ("good", 1.5): _row("good", target="COMMIT:DELETE", confidence=0.9, gain=0.3),
        ("weak", 1.5): _row("weak", target="COMMIT:DELETE", confidence=0.4, gain=0.2),
        ("bad", 1.5): _row("bad", target="REJECT", confidence=0.8, gain=-0.4),
    }
    result = audit(
        {"direct": policy},
        gain_epsilon=1e-6,
        maximum_harmful_accept_rate=0.0,
    )
    pooled = result["groups"]["pooled"]
    assert pooled["confidence_auroc_beneficial_vs_harmful"] == pytest.approx(0.5)
    assert pooled["best_operating_point"]["beneficial_delete_recall"] == 0.5
    assert pooled["best_operating_point"]["harmful_accept_rate"] == 0.0
    curve = {row["confidence_threshold"]: row for row in pooled["threshold_curve"]}
    assert curve[0.9]["beneficial_delete_recall"] == 0.5
    assert curve[0.8]["harmful_accept_rate"] == 1.0
    assert curve[0.0]["accept_rate"] == 1.0


def test_compaction_ignores_non_delete_proposals() -> None:
    keep = _row("keep", target="REJECT", confidence=0.9, gain=0.0)
    keep["effective_operation"] = "KEEP"
    delete = _row("delete", target="COMMIT:DELETE", confidence=0.8, gain=0.1)
    rows = compact_delete_rows(
        {"direct": {("keep", 1.5): keep, ("delete", 1.5): delete}},
        gain_epsilon=1e-6,
    )
    assert [row["task_id"] for row in rows] == ["delete"]


def test_parser_accepts_seed_specific_record_name() -> None:
    name, path = parse_record("seed20260817_direct=/tmp/writeback.jsonl")
    assert name == "seed20260817_direct"
    assert path.name == "writeback.jsonl"
