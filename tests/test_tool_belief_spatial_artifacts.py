from pathlib import Path

import numpy as np

from scripts.audit_tool_belief_spatial_artifacts import audit_spatial_artifacts


def _record(call_id: str, *, split: str = "train") -> dict[str, object]:
    return {
        "split": split,
        "gt_edit": "ADD",
        "tool_result": {"call_id": call_id, "tool": "TEMPORAL_CHANGE"},
    }


def test_spatial_artifact_audit_accepts_exact_finite_mapping(tmp_path: Path) -> None:
    np.save(tmp_path / "call-1_change.npy", np.eye(8, dtype=np.float32))
    summary = audit_spatial_artifacts([_record("call-1")], tmp_path)
    assert summary["passed"] is True
    assert summary["shape_counts"] == {"(8, 8)": 1}
    assert summary["blank_count"] == 0


def test_spatial_artifact_audit_rejects_missing_and_test_records(tmp_path: Path) -> None:
    summary = audit_spatial_artifacts([_record("missing", split="test")], tmp_path)
    assert summary["passed"] is False
    assert summary["missing_count"] == 1
    assert summary["invalid_count"] == 1
