import json
from pathlib import Path

import numpy as np

from scripts.assemble_policy_relative_crossfit_features import assemble


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _feature_bundle(root: Path, rows: list[dict], adapter: str) -> None:
    root.mkdir(parents=True)
    np.save(root / "features.npy", np.arange(len(rows) * 4).reshape(len(rows), 4))
    _jsonl(root / "records.jsonl", rows)
    (root / "summary.json").write_text(
        json.dumps({"adapter": adapter, "test_assets_read": False}), encoding="utf-8"
    )


def _branch(root: Path, rows: list[dict]) -> None:
    root.mkdir(parents=True)
    _jsonl(root / "traces.jsonl", rows)
    (root / "summary.json").write_text(
        json.dumps({"test_assets_read": False}), encoding="utf-8"
    )


def test_assemble_joins_outer_fold_labels_without_duplicates(tmp_path):
    folds = tmp_path / "folds"
    folds.mkdir()
    (folds / "summary.json").write_text(
        json.dumps(
            {
                "folds": [{"fold": 0}, {"fold": 1}],
                "source_examples": 2,
                "test_assets_read": False,
            }
        ),
        encoding="utf-8",
    )
    _jsonl(
        folds / "outer_task_assignments.jsonl",
        [
            {"task_id": "train-0", "outer_fold": 0},
            {"task_id": "train-1", "outer_fold": 1},
        ],
    )
    features = tmp_path / "features"
    branches = tmp_path / "branches"
    for fold in (0, 1):
        example = f"example-{fold}"
        task = f"train-{fold}"
        _feature_bundle(
            features / f"fold{fold}",
            [{"example_id": example, "task_id": task, "split": "train"}],
            f"adapter-{fold}",
        )
        _branch(
            branches / f"fold{fold}",
            [
                {
                    "example_id": example,
                    "task_id": task,
                    "split": "train",
                    "policy_relative_advantage": 1.0 if fold == 0 else -0.5,
                    "policy_relative_use_tool": fold == 0,
                    "static_use_tool": False,
                    "direct_terminal_valid": True,
                    "post_terminal_valid": True,
                }
            ],
        )
    val_features = tmp_path / "val_features"
    val_branches = tmp_path / "val_branches"
    _feature_bundle(
        val_features,
        [{"example_id": "val-0", "task_id": "val-task", "split": "val"}],
        "full-adapter",
    )
    _branch(
        val_branches,
        [
            {
                "example_id": "val-0",
                "task_id": "val-task",
                "split": "val",
                "policy_relative_advantage": 0.25,
                "policy_relative_use_tool": True,
                "static_use_tool": False,
                "direct_terminal_valid": True,
                "post_terminal_valid": True,
            }
        ],
    )

    result = assemble(
        folds,
        branches,
        features,
        val_branches,
        val_features,
        tmp_path / "output",
    )

    assert result["train"]["sample_count"] == 2
    assert result["train"]["positive_count"] == 1
    assert result["val"]["positive_count"] == 1
    assert result["train"]["crossfit_protocol"] == "task-disjoint-outer-fold-adapter"
