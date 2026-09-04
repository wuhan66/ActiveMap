import json
from pathlib import Path

from scripts.audit_active_catalog_trace_support import audit_support


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row(source_episode: str, budget: float, *, aoi_id: str = "aoi") -> dict:
    return {
        "source_episode": source_episode,
        "budget": budget,
        "aoi_id": aoi_id,
        "target_edit": "KEEP",
        "split": "val",
        "test_assets_read": False,
    }


def test_audit_support_reports_key_and_metadata_differences(tmp_path: Path) -> None:
    reference = tmp_path / "reference.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write(reference, [_row("a", 1.5), _row("b", 3.0)])
    _write(candidate, [_row("a", 1.5, aoi_id="other"), _row("c", 3.0)])

    result = audit_support(reference, candidate)

    assert result["identical_support"] is False
    assert result["reference_only"] == [["b", 3.0]]
    assert result["candidate_only"] == [["c", 3.0]]
    assert result["metadata_mismatches"][0]["fields"]["aoi_id"] == {
        "reference": "aoi",
        "candidate": "other",
    }
