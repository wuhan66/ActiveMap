import json
from pathlib import Path

import pytest

from scripts.filter_agent_writeback_support import filter_support


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row(task: str, budget: float, target: str = "KEEP") -> dict:
    return {
        "task_id": task,
        "budget": budget,
        "target": target,
        "aoi_id": "aoi-1",
        "split": "val",
        "test_assets_read": False,
    }


def test_filters_candidate_in_reference_order(tmp_path: Path) -> None:
    reference = tmp_path / "reference.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write(reference, [_row("b", 3.0), _row("a", 1.5)])
    _write(candidate, [_row("a", 1.5), _row("extra", 4.5), _row("b", 3.0)])

    rows, summary = filter_support(reference, candidate)

    assert [(row["task_id"], row["budget"]) for row in rows] == [
        ("b", 3.0),
        ("a", 1.5),
    ]
    assert summary["matched_count"] == 2
    assert summary["candidate_count"] == 3


def test_rejects_metadata_mismatch(tmp_path: Path) -> None:
    reference = tmp_path / "reference.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write(reference, [_row("a", 1.5)])
    _write(candidate, [_row("a", 1.5, target="ADD")])

    with pytest.raises(ValueError, match="metadata mismatch"):
        filter_support(reference, candidate)
