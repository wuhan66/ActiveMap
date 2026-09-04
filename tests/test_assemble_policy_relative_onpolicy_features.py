import json
from pathlib import Path

import numpy as np

from scripts.assemble_policy_relative_onpolicy_features import assemble_onpolicy_features


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _features(root: Path, example_id: str, task_id: str, split: str) -> None:
    root.mkdir()
    np.save(root / "features.npy", np.ones((1, 4), dtype=np.float16))
    _jsonl(
        root / "records.jsonl",
        [{"example_id": example_id, "task_id": task_id, "split": split}],
    )
    (root / "summary.json").write_text(
        json.dumps(
            {"adapter": "full-adapter", "test_assets_read": False}
        ),
        encoding="utf-8",
    )


def _branches(root: Path, example_id: str, task_id: str, split: str) -> None:
    root.mkdir()
    _jsonl(
        root / "traces.jsonl",
        [
            {
                "example_id": example_id,
                "task_id": task_id,
                "split": split,
                "policy_relative_advantage": 1.0,
                "policy_relative_use_tool": True,
                "static_use_tool": False,
                "direct_terminal_valid": True,
                "post_terminal_valid": True,
            }
        ],
    )
    (root / "summary.json").write_text(
        json.dumps({"test_assets_read": False}), encoding="utf-8"
    )


def test_onpolicy_features_join_same_adapter_states_and_advantages(tmp_path):
    train_features = tmp_path / "train-features"
    val_features = tmp_path / "val-features"
    train_branches = tmp_path / "train-branches"
    val_branches = tmp_path / "val-branches"
    _features(train_features, "train-e", "train-t", "train")
    _features(val_features, "val-e", "val-t", "val")
    _branches(train_branches, "train-e", "train-t", "train")
    _branches(val_branches, "val-e", "val-t", "val")

    result = assemble_onpolicy_features(
        train_branches,
        train_features,
        val_branches,
        val_features,
        tmp_path / "output",
    )

    assert result["label_source"] == "same-full-adapter-realized-advantage"
    assert result["train"]["positive_count"] == 1
    assert result["train"]["crossfit_protocol"] == (
        "current-deployment-policy-on-train-environments"
    )
