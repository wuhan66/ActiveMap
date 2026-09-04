from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.promote_sparse_tool_sft_adapter import (
    promote_adapter,
    verify_adapter_promotion,
)


def test_promote_adapter_hashes_validation_selected_files(tmp_path: Path) -> None:
    run = tmp_path / "run"
    adapter = run / "checkpoints/checkpoint-20"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    decision = tmp_path / "decision.json"
    decision.write_text(
        json.dumps(
            {
                "protocol": {"test_assets_read": False},
                "selection_passed": True,
                "selected_checkpoint": "checkpoint-20",
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "promoted.json"
    promotion = promote_adapter(run, decision, output, seed=20260821)
    assert promotion["approved"] is True
    assert promotion["seed"] == 20260821
    assert len(promotion["adapter_artifacts"]) == 2
    assert verify_adapter_promotion(output)["approved"] is True
    with pytest.raises(FileExistsError):
        promote_adapter(run, decision, output, seed=20260821)
    (adapter / "adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        verify_adapter_promotion(output)


def test_promote_adapter_rejects_failed_validation_selection(tmp_path: Path) -> None:
    decision = tmp_path / "decision.json"
    decision.write_text(
        json.dumps(
            {
                "protocol": {"test_assets_read": False},
                "selection_passed": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no checkpoint"):
        promote_adapter(tmp_path, decision, tmp_path / "out.json", seed=1)
