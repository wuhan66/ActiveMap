import hashlib
from pathlib import Path

import yaml

from scripts.audit_sn7_step0_frozen_registry_v2 import audit


def test_audit_requires_hashes_and_keeps_test_gate_closed(tmp_path: Path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"frozen")
    registry = {
        "schema_version": "sn7-step0-frozen-registry-v2",
        "shared_artifacts": [
            {
                "id": "artifact",
                "path": "${STORAGE_ROOT}/artifact.bin",
                "sha256": hashlib.sha256(b"frozen").hexdigest(),
            }
        ],
        "test_gate": {"ready": False, "blockers": ["test_manifest_missing"]},
    }
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump(registry), encoding="utf-8")
    result = audit(path, tmp_path)
    assert result["artifacts_ready"] is True
    assert result["ready_for_frozen_test"] is False
