import json

import pytest

from scripts.validate_quality_rollout_pair import validate_pair


def _write(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _row(task_id="task-a", budget=1.5):
    return {
        "task_id": task_id,
        "budget": budget,
        "final_evidence_quality": 0.8,
        "evidence_quality_gain": 0.2,
        "quality_cost_utility": 0.1,
        "selected_evidence_ids": ["evidence-a"],
    }


def test_validate_pair_accepts_identical_keys_and_quality_fields(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write(baseline, [_row()])
    _write(candidate, [_row()])

    result = validate_pair(baseline, candidate)

    assert result["protocol"] == "quality-cost-v2"
    assert result["sample_count"] == 1


def test_validate_pair_rejects_legacy_rows(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write(baseline, [{"task_id": "task-a", "budget": 1.5}])
    _write(candidate, [_row()])

    with pytest.raises(ValueError, match="quality_cost_utility"):
        validate_pair(baseline, candidate)


def test_validate_pair_rejects_unmatched_protocol_keys(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write(baseline, [_row()])
    _write(candidate, [_row(task_id="task-b")])

    with pytest.raises(ValueError, match="keys differ"):
        validate_pair(baseline, candidate)
