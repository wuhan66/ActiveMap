import json

import pytest

from scripts.verify_sn7_active_catalog_gates import (
    verify_smoke,
    verify_token_audit,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def token_report(train=10, val=4):
    def split(records):
        return {
            "records": records,
            "actions": {"ACQUIRE": 1, "STOP": records - 1},
            "over_max_length_count": 0,
            "zero_supervised_labels": [],
        }

    return {
        "schema_version": "visual-sft-token-audit-v1",
        "passed": True,
        "test_assets_read": False,
        "max_length": 2048,
        "over_max_length_records": 0,
        "zero_supervised_label_records": 0,
        "splits": {"train": split(train), "val": split(val)},
    }


def test_verify_token_audit_accepts_complete_report(tmp_path):
    path = tmp_path / "tokens.json"
    write_json(path, token_report())
    result = verify_token_audit(path, expected_train=10, expected_val=4, max_length=2048)
    assert result["passed"] is True


def test_verify_token_audit_rejects_mismatched_record_count(tmp_path):
    path = tmp_path / "tokens.json"
    write_json(path, token_report(train=9))
    with pytest.raises(ValueError, match="train.records"):
        verify_token_audit(path, expected_train=10, expected_val=4, max_length=2048)


def test_verify_smoke_requires_completed_process_and_adapter(tmp_path):
    root = tmp_path / "smoke"
    audit = {
        "records": 2,
        "action_counts": {"ACQUIRE": 1, "STOP": 1},
        "invalid_action_records": 0,
    }
    write_json(
        root / "launch_manifest.json",
        {
            "test_assets_read": False,
            "schedule": [{"seed": 7, "gpu": 0}],
            "data_audit": {"train": audit, "val": audit},
        },
    )
    write_json(root / "seed7" / "run_state.json", {"status": "completed", "returncode": 0})
    write_json(root / "seed7" / "process_result.json", {"returncode": 0, "peak_memory_used_mib": 100})
    write_json(root / "seed7" / "final" / "adapter_config.json", {})
    (root / "seed7" / "history.jsonl").write_text('{"loss": 1.0}\n', encoding="utf-8")
    assert verify_smoke(root, 7)["passed"] is True


def test_verify_smoke_rejects_failed_process(tmp_path):
    root = tmp_path / "smoke"
    audit = {
        "records": 2,
        "action_counts": {"ACQUIRE": 1, "STOP": 1},
        "invalid_action_records": 0,
    }
    write_json(
        root / "launch_manifest.json",
        {
            "test_assets_read": False,
            "schedule": [{"seed": 7, "gpu": 0}],
            "data_audit": {"train": audit, "val": audit},
        },
    )
    write_json(root / "seed7" / "run_state.json", {"status": "failed", "returncode": 1})
    write_json(root / "seed7" / "process_result.json", {"returncode": 1})
    with pytest.raises(ValueError, match="run_state.status"):
        verify_smoke(root, 7)
