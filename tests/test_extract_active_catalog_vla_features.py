from pathlib import Path

import pytest

from scripts.extract_active_catalog_vla_features import model_identity


def test_model_identity_hashes_adapter_config_and_weights(tmp_path: Path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")

    identity = model_identity(str(adapter), adapter=True)

    assert identity["local_path"] is True
    assert len(identity["config_sha256"]) == 64
    assert len(identity["weights_sha256"]) == 64


def test_model_identity_rejects_adapter_without_weights(tmp_path: Path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="adapter weight"):
        model_identity(str(adapter), adapter=True)


def test_model_identity_preserves_remote_identifier():
    identity = model_identity("org/model", adapter=False)

    assert identity == {"identifier": "org/model", "local_path": False}
