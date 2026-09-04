import hashlib
import json

import pytest
from PIL import Image

from scripts.audit_active_catalog_sft import audit


def _row(split, image="images/task.jpg", selection="ACQUIRE"):
    state = {
        "budget": {"remaining": 1.0},
        "selected_evidence_ids": ["evidence-old"],
        "candidate_evidence": [{"evidence_id": "evidence-new", "cost": 1.0}],
    }
    action = {"stage": "SELECT", "selection": selection}
    if selection == "ACQUIRE":
        action["evidence_id"] = "evidence-new"
    return {
        "example_id": f"{split}-example",
        "task_id": f"{split}-task",
        "split": split,
        "stage": "SELECT",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "system"}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": json.dumps(state)},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": json.dumps(action)}],
            },
        ],
    }


def _dataset(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (8, 8)).save(images / "task.jpg")
    Image.new("RGB", (8, 8)).save(images / "val.jpg")
    train = _row("train")
    val = _row("val", image="images/val.jpg", selection="STOP")
    (tmp_path / "train.jsonl").write_text(json.dumps(train) + "\n")
    (tmp_path / "val.jsonl").write_text(json.dumps(val) + "\n")
    evaluation_paths = {}
    for split, row in (("train", train), ("val", val)):
        action = json.loads(row["messages"][2]["content"][0]["text"])
        item = {
            "example_id": row["example_id"],
            "split": split,
            "target_selection": action["selection"],
            "target_evidence_id": action.get("evidence_id"),
            "candidates": [{"evidence_id": "evidence-new", "utility": 0.2, "cost": 1.0}],
            "model_visible": False,
            "test_assets_read": False,
        }
        path = tmp_path / f"{split}_evaluation_index.jsonl"
        path.write_text(json.dumps(item) + "\n")
        evaluation_paths[split] = path
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "active-catalog-sequential-selector-sft-v2",
                "test_assets_read": False,
                "prompt_target_metadata_exposed": False,
                "portable_relative_image_paths": True,
                "evaluation_indices": {
                    split: {
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "model_visible": False,
                    }
                    for split, path in evaluation_paths.items()
                },
            }
        )
    )
    return train, val


def test_audit_accepts_portable_executable_catalog(tmp_path):
    _dataset(tmp_path)
    report = audit(tmp_path)
    assert report["passed"] is True
    assert report["states"] == 2
    assert report["images"] == 2


def test_audit_rejects_prompt_target_leakage(tmp_path):
    train, _ = _dataset(tmp_path)
    state = json.loads(train["messages"][1]["content"][1]["text"])
    state["gt_edit"] = "ADD"
    train["messages"][1]["content"][1]["text"] = json.dumps(state)
    (tmp_path / "train.jsonl").write_text(json.dumps(train) + "\n")
    with pytest.raises(ValueError, match="target leakage"):
        audit(tmp_path)
