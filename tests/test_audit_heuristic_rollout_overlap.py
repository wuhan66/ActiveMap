import json

import pytest

from scripts.audit_heuristic_rollout_overlap import audit_rollout_overlap


def _write(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _row(task_id, acquired, *, prediction="COMMIT:ADD", budget=3.0):
    return {
        "task_id": task_id,
        "budget": budget,
        "prediction": prediction,
        "acquisitions": len(acquired),
        "selected_evidence_ids": ["initial", *acquired],
    }


def test_audit_detects_identical_acquisition_strategies(tmp_path):
    left, right = tmp_path / "left.jsonl", tmp_path / "right.jsonl"
    rows = [_row("task-a", ["e1", "e2"]), _row("task-b", ["e3"])]
    _write(left, rows)
    _write(right, rows)

    result = audit_rollout_overlap({"left": left, "right": right})

    pair = result["pairwise"][0]
    assert pair["acquisition_sequence_agreement"] == 1.0
    assert pair["acquisition_set_jaccard"] == 1.0
    assert result["warnings"][0]["type"] == "strategy_selection_collapse"


def test_audit_separates_common_initial_evidence_from_acquisitions(tmp_path):
    left, right = tmp_path / "left.jsonl", tmp_path / "right.jsonl"
    _write(left, [_row("task-a", ["e1", "e2"])])
    _write(right, [_row("task-a", ["e2", "e1"])])

    result = audit_rollout_overlap({"left": left, "right": right})

    pair = result["pairwise"][0]
    assert pair["acquisition_sequence_agreement"] == 0.0
    assert pair["acquisition_set_jaccard"] == 1.0
    assert result["warnings"] == []


def test_audit_rejects_unpaired_rollout_support(tmp_path):
    left, right = tmp_path / "left.jsonl", tmp_path / "right.jsonl"
    _write(left, [_row("task-a", ["e1"])])
    _write(right, [_row("task-b", ["e1"])])

    with pytest.raises(ValueError, match="keys differ"):
        audit_rollout_overlap({"left": left, "right": right})
