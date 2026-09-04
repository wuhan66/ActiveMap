import json

import pytest

from scripts.train_semantic_vlm_sft import (
    action_sampling_weights,
    action_sequence_weights,
    adapter_initialization_manifest,
    record_loss_weights,
)


def test_adapter_initialization_manifest_hashes_config_and_weights(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 16}))
    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")

    manifest = adapter_initialization_manifest(tmp_path)

    assert manifest["weights"]["bytes"] == 7
    assert len(manifest["weights"]["sha256"]) == 64
    assert len(manifest["config"]["sha256"]) == 64


def test_adapter_initialization_manifest_rejects_incomplete_adapter(tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}")

    with pytest.raises(FileNotFoundError, match="incomplete"):
        adapter_initialization_manifest(tmp_path)


def _action_row(action):
    return {
        "messages": [
            {},
            {},
            {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({"stage": "SELECT", "selection": action}),
                    }
                ]
            },
        ]
    }


def test_action_sampling_weights_target_acquire_exposure():
    rows = [_action_row("ACQUIRE")] + [_action_row("STOP") for _ in range(9)]
    weights, summary = action_sampling_weights(rows, 0.25)
    assert weights[0] / sum(weights) == pytest.approx(0.25)
    assert summary["original_action_counts"] == {"ACQUIRE": 1, "STOP": 9}


def test_action_sequence_weights_only_upweight_acquire():
    rows = [_action_row("STOP"), _action_row("ACQUIRE"), _action_row("STOP")]
    assert action_sequence_weights(rows, 3.0) == [1.0, 3.0, 1.0]


def test_action_sequence_weights_reject_downweighting():
    with pytest.raises(ValueError, match="at least one"):
        action_sequence_weights([_action_row("ACQUIRE")], 0.5)


def test_record_loss_weights_are_train_only_metadata():
    rows = [{"crossfit_reliability_weight": 1.0}, {"crossfit_reliability_weight": 0.5}]
    weights, summary = record_loss_weights(rows, "crossfit_reliability_weight")
    assert weights == [1.0, 0.5]
    assert summary["mode"] == "loss_only"
    with pytest.raises(ValueError, match="missing"):
        record_loss_weights([{}], "crossfit_reliability_weight")
