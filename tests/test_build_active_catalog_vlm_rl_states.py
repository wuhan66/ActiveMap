import json

from scripts.build_active_catalog_vlm_rl_states import build_states


def _write(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_builds_hidden_executable_action_set(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "x.png").write_bytes(b"image")
    action = {"stage": "SELECT", "selection": "ACQUIRE", "evidence_id": "e1"}
    sft = tmp_path / "train.jsonl"
    index = tmp_path / "index.jsonl"
    output = tmp_path / "rl.jsonl"
    _write(sft, [{
        "example_id": "x", "task_id": "task-a", "split": "train",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "system"}]},
            {"role": "user", "content": [{"type": "image", "image": "images/x.png"}]},
            {"role": "assistant", "content": [{"type": "text", "text": json.dumps(action)}]},
        ],
    }])
    _write(index, [{
        "example_id": "x", "split": "train", "target_utility": 0.8,
        "target_selection": "ACQUIRE", "target_evidence_id": "e1",
        "stop_utility": 0.2, "model_visible": False, "test_assets_read": False,
        "candidates": [
            {"evidence_id": "e1", "utility": 0.8, "cost": 1.0},
            {"evidence_id": "e2", "utility": 0.5, "cost": 0.5},
        ],
    }])
    summary = build_states(sft, index, output, expected_split="train")
    row = json.loads(output.read_text(encoding="utf-8"))
    assert summary["states"] == 1
    assert row["target_action_key"] == "ACQUIRE:e1"
    assert [item["key"] for item in row["actions"]] == ["STOP", "ACQUIRE:e1", "ACQUIRE:e2"]
    assert row["model_visible_utility"] is False
    assert row["prompt"][1]["content"][0]["image"] == str((images / "x.png").resolve())
