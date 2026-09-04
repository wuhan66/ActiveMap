import json
from pathlib import Path

import pytest

from scripts.merge_cluster_progress import merge
from scripts.snapshot_cluster_progress import code_fingerprint


def _snapshot(
    path: Path,
    *,
    cluster_id: str,
    role: str,
    code_hash: str = "code",
    registry_hash: str = "registry",
    ledgers: list[dict[str, str]] | None = None,
) -> Path:
    payload = {
        "schema_version": "activemap-cluster-progress-v1",
        "cluster_id": cluster_id,
        "role": role,
        "code_fingerprint": code_hash,
        "registry_sha256": registry_hash,
        "protocol_valid": True,
        "ready_for_frozen_test": role == "authoritative",
        "frozen_test_ledgers": ledgers or [],
        "test_access_allowed": role == "authoritative",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_joint_progress_requires_synced_code_and_single_test_authority(
    tmp_path: Path,
) -> None:
    snapshots = {
        "hdpi": _snapshot(
            tmp_path / "hdpi.json", cluster_id="hdpi", role="authoritative"
        ),
        "nts": _snapshot(
            tmp_path / "nts.json", cluster_id="nts", role="validation_debug"
        ),
    }

    report = merge(snapshots)

    assert report["joint_validation_allowed"] is True
    assert report["frozen_test_allowed"] is True
    assert report["authoritative_cluster"] == "hdpi"


def test_joint_progress_rejects_debug_test_ledger_and_code_drift(tmp_path: Path) -> None:
    snapshots = {
        "hdpi": _snapshot(
            tmp_path / "hdpi.json", cluster_id="hdpi", role="authoritative"
        ),
        "nts": _snapshot(
            tmp_path / "nts.json",
            cluster_id="nts",
            role="validation_debug",
            code_hash="old-code",
            ledgers=[{"path": "forbidden", "status": "started"}],
        ),
    }

    report = merge(snapshots)

    assert report["code_in_sync"] is False
    assert report["test_lineage_valid"] is False
    assert report["joint_validation_allowed"] is False
    assert report["frozen_test_allowed"] is False


def test_code_fingerprint_is_content_and_path_sensitive(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    source = tmp_path / "src" / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    first, count = code_fingerprint(tmp_path)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    second, second_count = code_fingerprint(tmp_path)

    assert count == second_count == 1
    assert first != second


def test_merge_requires_one_authoritative_cluster(tmp_path: Path) -> None:
    snapshots = {
        "a": _snapshot(tmp_path / "a.json", cluster_id="a", role="validation_debug"),
        "b": _snapshot(tmp_path / "b.json", cluster_id="b", role="validation_debug"),
    }
    with pytest.raises(ValueError, match="exactly one authoritative"):
        merge(snapshots)
