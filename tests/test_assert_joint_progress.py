import json
from pathlib import Path

import pytest

from scripts.assert_joint_progress import assert_joint_progress
from scripts.snapshot_cluster_progress import code_fingerprint


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    (project / "src").mkdir(parents=True)
    (project / "src" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    return project


def _ledger(path: Path, project: Path) -> Path:
    fingerprint, count = code_fingerprint(project)
    payload = {
        "schema_version": "activemap-unified-progress-v1",
        "joint_validation_allowed": True,
        "violations": [],
        "clusters": {
            "nts": {
                "role": "validation_debug",
                "code_fingerprint": fingerprint,
                "code_file_count": count,
                "registry_sha256": "registry",
                "test_access_allowed": False,
                "frozen_test_ledgers": [],
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_authorizes_only_exact_snapshotted_validation_code(tmp_path: Path) -> None:
    project = _project(tmp_path)
    ledger = _ledger(tmp_path / "unified.json", project)

    authorization = assert_joint_progress(
        ledger,
        project,
        cluster_id="nts",
        expected_role="validation_debug",
    )

    assert authorization["role"] == "validation_debug"
    assert len(authorization["ledger_sha256"]) == 64


def test_rejects_code_changed_after_snapshot(tmp_path: Path) -> None:
    project = _project(tmp_path)
    ledger = _ledger(tmp_path / "unified.json", project)
    (project / "src" / "module.py").write_text("VALUE = 2\n", encoding="utf-8")

    with pytest.raises(PermissionError, match="current code differs"):
        assert_joint_progress(
            ledger,
            project,
            cluster_id="nts",
            expected_role="validation_debug",
        )
