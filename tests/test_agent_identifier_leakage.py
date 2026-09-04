import json

import pytest

from scripts.audit_agent_identifier_leakage import audit


def _observation(evidence_id: str = "evidence-0123456789abcdef") -> dict:
    return {
        "task_id": "task-0123456789abcdef",
        "selected_evidence_ids": [],
        "candidates": [{"evidence_id": evidence_id}],
    }


def test_identifier_audit_supports_sft_and_preference_records(tmp_path) -> None:
    observation = json.dumps(_observation(), separators=(",", ":"))
    action = json.dumps({"action": "REJECT"}, separators=(",", ":"))
    rows = [
        {
            "messages": [
                {"role": "user", "content": observation},
                {"role": "assistant", "content": action},
            ]
        },
        {
            "prompt": f"system\n{observation}\nAction:",
            "chosen": action,
        },
    ]
    path = tmp_path / "records.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    assert audit(path) == {"records": 2, "failures": 0}


def test_identifier_audit_rejects_semantic_preference_id(tmp_path) -> None:
    observation = json.dumps(_observation("scene-add-2021"), separators=(",", ":"))
    row = {
        "prompt": f"system\n{observation}\nAction:",
        "chosen": json.dumps({"action": "REJECT"}),
    }
    path = tmp_path / "unsafe.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="identifier leakage audit failed"):
        audit(path)
