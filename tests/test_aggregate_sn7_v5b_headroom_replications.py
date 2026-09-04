from __future__ import annotations

import json
from pathlib import Path

from scripts.aggregate_sn7_v5b_headroom_replications import validate_record


def _summary(*, delete_lower: float = 0.2) -> dict:
    return {
        "schema_version": "sn7-nonkeep-candidate-headroom-v1",
        "test_assets_read": False,
        "states": "/tmp/states.jsonl",
        "states_sha256": "a" * 64,
        "gate": {
            "passes_candidate_recovery_preflight": True,
            "passes_by_operation": {"ADD": True, "DELETE": True, "RESHAPE": True},
        },
        "slices": {
            "ADD": {
                "row_count": 20,
                "aoi_count": 4,
                "safe_map_headroom_ci95": [0.1, 0.3],
                "passes_preflight": True,
            },
            "DELETE": {
                "row_count": 20,
                "aoi_count": 4,
                "safe_map_headroom_ci95": [delete_lower, 0.3],
                "passes_preflight": True,
            },
            "RESHAPE": {
                "row_count": 20,
                "aoi_count": 4,
                "safe_map_headroom_ci95": [0.1, 0.3],
                "passes_preflight": True,
            },
        },
    }


def _receipt() -> dict:
    return {
        "checkpoint": "/tmp/best_quality.pt",
        "checkpoint_sha256": "b" * 64,
        "selection": "explicit queue input",
    }


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_accepts_independently_passing_validation_record(tmp_path: Path) -> None:
    result = validate_record(
        20260817,
        _write(tmp_path / "summary.json", _summary()),
        _write(tmp_path / "checkpoint.json", _receipt()),
    )

    assert result["passed"] is True
    assert result["errors"] == []
    assert set(result["operations"]) == {"ADD", "DELETE", "RESHAPE"}


def test_rejects_nonpositive_operation_lower_bound(tmp_path: Path) -> None:
    result = validate_record(
        20260817,
        _write(tmp_path / "summary.json", _summary(delete_lower=0.0)),
        _write(tmp_path / "checkpoint.json", _receipt()),
    )

    assert result["passed"] is False
    assert "DELETE lower headroom bound is not strictly positive" in result["errors"]
