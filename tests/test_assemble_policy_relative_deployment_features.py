import json
from pathlib import Path

import numpy as np

from scripts.assemble_policy_relative_deployment_features import assemble_deployment_features


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _features(root: Path, row: dict, adapter: str) -> None:
    root.mkdir(parents=True)
    np.save(root / "features.npy", np.ones((1, 4), dtype=np.float16))
    _jsonl(root / "records.jsonl", [row])
    (root / "summary.json").write_text(
        json.dumps({"adapter": adapter, "test_assets_read": False}), encoding="utf-8"
    )


def _branches(root: Path, row: dict) -> None:
    root.mkdir(parents=True)
    _jsonl(root / "traces.jsonl", [row])
    (root / "summary.json").write_text(
        json.dumps({"test_assets_read": False}), encoding="utf-8"
    )


def test_deployment_features_keep_crossfit_labels_and_fold_provenance(tmp_path):
    folds = tmp_path / "folds"
    folds.mkdir()
    (folds / "summary.json").write_text(
        json.dumps({"folds": [{"fold": 0}], "test_assets_read": False}), encoding="utf-8"
    )
    _jsonl(folds / "outer_task_assignments.jsonl", [{"task_id": "task", "outer_fold": 0}])
    branch = {
        "example_id": "train-example",
        "task_id": "task",
        "split": "train",
        "policy_relative_advantage": 1.0,
        "policy_relative_use_tool": True,
        "static_use_tool": False,
        "direct_terminal_valid": True,
        "post_terminal_valid": True,
    }
    branch_root = tmp_path / "branches"
    _branches(branch_root / "fold0", branch)
    train_features = tmp_path / "train-features"
    _features(
        train_features,
        {"example_id": "train-example", "task_id": "task", "split": "train"},
        "deployment-adapter",
    )
    val_branch_root = tmp_path / "val-branches"
    val_branch = {**branch, "example_id": "val-example", "task_id": "val-task", "split": "val"}
    _branches(val_branch_root, val_branch)
    val_features = tmp_path / "val-features"
    _features(
        val_features,
        {"example_id": "val-example", "task_id": "val-task", "split": "val"},
        "deployment-adapter",
    )

    result = assemble_deployment_features(
        folds,
        branch_root,
        train_features,
        val_branch_root,
        val_features,
        tmp_path / "output",
    )

    assert result["train"]["positive_count"] == 1
    assert "full-deployment-adapter" in result["train"]["crossfit_protocol"]
    records = [
        json.loads(line)
        for line in (tmp_path / "output" / "train" / "records.jsonl").read_text().splitlines()
    ]
    assert records[0]["crossfit_fold"] == 0
