import json

import pytest

from scripts.build_active_catalog_vlm_preferences import build_preferences


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def sft_row(example_id="x", selection="ACQUIRE", evidence_id="e1"):
    action = {"stage": "SELECT", "selection": selection}
    if evidence_id is not None and selection == "ACQUIRE":
        action["evidence_id"] = evidence_id
    return {
        "example_id": example_id,
        "task_id": "task",
        "split": "train",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "system"}]},
            {"role": "user", "content": [{"type": "image", "image": "images/x.png"}]},
            {"role": "assistant", "content": [{"type": "text", "text": json.dumps(action)}]},
        ],
    }


def index_row(target="ACQUIRE", target_id="e1"):
    return {
        "example_id": "x",
        "split": "train",
        "target_selection": target,
        "target_evidence_id": target_id,
        "target_utility": 0.8,
        "stop_utility": 0.2,
        "candidates": [
            {"evidence_id": "e1", "utility": 0.8, "cost": 1.0},
            {"evidence_id": "e2", "utility": 0.6, "cost": 0.5},
        ],
        "model_visible": False,
        "test_assets_read": False,
    }


def test_builds_hard_evidence_and_stop_preferences(tmp_path):
    sft = tmp_path / "train.jsonl"
    index = tmp_path / "index.jsonl"
    output = tmp_path / "preferences.jsonl"
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "x.png").write_bytes(b"image-placeholder")
    write_jsonl(sft, [sft_row()])
    write_jsonl(index, [index_row()])
    summary = build_preferences(
        sft, index, output, expected_split="train", pairs_per_state=2
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert summary["preference_pairs"] == 2
    assert [row["rejected_action_key"] for row in rows] == ["ACQUIRE:e2", "STOP"]
    assert rows[0]["prompt"][0] == sft_row()["messages"][0]
    assert rows[0]["prompt"][1]["content"][0]["image"] == str(
        (tmp_path / "images" / "x.png").resolve()
    )
    assert all(row["model_visible_utility"] is False for row in rows)


def test_rejects_target_that_is_not_utility_optimal(tmp_path):
    sft = tmp_path / "train.jsonl"
    index = tmp_path / "index.jsonl"
    output = tmp_path / "preferences.jsonl"
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "x.png").write_bytes(b"image-placeholder")
    write_jsonl(sft, [sft_row()])
    row = index_row()
    row["candidates"][1]["utility"] = 0.9
    write_jsonl(index, [row])
    with pytest.raises(ValueError, match="not shortlist-optimal"):
        build_preferences(sft, index, output, expected_split="train")
