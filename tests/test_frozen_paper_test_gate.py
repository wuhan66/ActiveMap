import json
import sys
from pathlib import Path

import pytest
import yaml

from scripts.run_frozen_paper_test import run_frozen_test


def _populate_required_artifacts(storage: Path) -> None:
    registry = yaml.safe_load(
        Path("configs/experiments/paper_registry.yaml").read_text(encoding="utf-8")
    )
    for artifact in registry["required_artifacts"]:
        path = Path(str(artifact["path"]).replace("${STORAGE_ROOT}", str(storage)))
        path.parent.mkdir(parents=True, exist_ok=True)
        if artifact["kind"] == "file":
            path.write_text("{}\n", encoding="utf-8")
            continue
        payload = {}
        cursor = payload
        parts = str(artifact["field"]).split(".")
        for part in parts[:-1]:
            cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = artifact["equals"]
        path.write_text(json.dumps(payload), encoding="utf-8")


def test_frozen_test_gate_is_one_shot(tmp_path: Path) -> None:
    storage = tmp_path / "storage"
    _populate_required_artifacts(storage)
    ledger = tmp_path / "ledger.json"
    marker = tmp_path / "marker.txt"
    registry = Path("configs/experiments/paper_registry.yaml")
    command = [
        sys.executable,
        "-c",
        (
            "from pathlib import Path; "
            "from scripts.frozen_test_access import assert_frozen_test_access; "
            "assert_frozen_test_access(); "
            f"Path({str(marker)!r}).write_text('ran')"
        ),
    ]
    assert run_frozen_test(registry, storage, ledger, command, purpose="unit test") == 0
    assert marker.read_text() == "ran"
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert payload["returncode"] == 0
    assert len(payload["authorization_sha256"]) == 64
    assert "ACTIVEMAP_FROZEN_TEST_TOKEN" not in payload
    expected_artifact_count = len(
        yaml.safe_load(registry.read_text(encoding="utf-8"))[
            "required_artifacts"
        ]
    )
    assert len(payload["artifact_sha256"]) == expected_artifact_count
    with pytest.raises(FileExistsError):
        run_frozen_test(registry, storage, ledger, command, purpose="second access")


def test_frozen_test_gate_does_not_create_ledger_when_blocked(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    with pytest.raises(ValueError, match="gate is closed"):
        run_frozen_test(
            Path("configs/experiments/paper_registry.yaml"),
            tmp_path / "empty",
            ledger,
            [sys.executable, "-c", "pass"],
            purpose="blocked",
        )
    assert not ledger.exists()


def test_frozen_test_access_cannot_reuse_completed_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.frozen_test_access import assert_frozen_test_access

    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps(
            {
                "schema_version": "activemap-frozen-test-access-v1",
                "status": "complete",
                "authorization_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ACTIVEMAP_FROZEN_TEST", "1")
    monkeypatch.setenv("ACTIVEMAP_FROZEN_TEST_LEDGER", str(ledger))
    monkeypatch.setenv("ACTIVEMAP_FROZEN_TEST_TOKEN", "forged")
    with pytest.raises(PermissionError, match="one-shot execution window"):
        assert_frozen_test_access()


def test_validation_debug_role_cannot_create_or_use_frozen_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.frozen_test_access import assert_frozen_test_access

    monkeypatch.setenv("ACTIVEMAP_DISABLE_FROZEN_TEST", "1")
    ledger = tmp_path / "ledger.json"
    with pytest.raises(PermissionError, match="disabled on this cluster role"):
        run_frozen_test(
            Path("configs/experiments/paper_registry.yaml"),
            tmp_path / "storage",
            ledger,
            [sys.executable, "-c", "pass"],
            purpose="forbidden debug test",
        )
    assert not ledger.exists()

    monkeypatch.setenv("ACTIVEMAP_FROZEN_TEST", "1")
    with pytest.raises(PermissionError, match="disabled on this cluster role"):
        assert_frozen_test_access()
