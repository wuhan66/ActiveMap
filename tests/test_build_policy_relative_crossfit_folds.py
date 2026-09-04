import json
from pathlib import Path

from scripts.build_policy_relative_crossfit_folds import build_folds


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_build_folds_are_task_disjoint_and_cover_every_example(tmp_path):
    base = []
    balanced = []
    rollout = []
    for task_index in range(18):
        task_id = f"task-{task_index:02d}"
        for example_index in range(2):
            example_id = f"{task_id}-example-{example_index}"
            positive = task_index % 3 == 0 and example_index == 0
            pre = {
                "task_id": task_id,
                "example_id": example_id,
                "stage": "PRE_TOOL",
                "oracle_use_tool": positive,
                "split": "train",
            }
            post = {**pre, "stage": "POST_TOOL"}
            base.extend([pre, post] if positive else [pre])
            balanced.extend([pre, pre, pre, post] if positive else [pre])
            rollout.extend([pre, post])
    balanced_path = tmp_path / "balanced.jsonl"
    base_path = tmp_path / "base.jsonl"
    rollout_path = tmp_path / "rollout.jsonl"
    _write(balanced_path, balanced)
    _write(base_path, base)
    _write(rollout_path, rollout)

    result = build_folds(
        balanced_path,
        base_path,
        rollout_path,
        tmp_path / "folds",
        folds=3,
        seed=7,
        inner_val_fraction=0.2,
    )

    heldout_tasks = set()
    heldout_examples = set()
    for spec in result["folds"]:
        root = tmp_path / "folds" / f"fold{spec['fold']}"
        train = [json.loads(line) for line in (root / "train_sft.jsonl").read_text().splitlines()]
        val = [json.loads(line) for line in (root / "inner_val_sft.jsonl").read_text().splitlines()]
        holdout = [
            json.loads(line) for line in (root / "holdout_rollout.jsonl").read_text().splitlines()
        ]
        train_tasks = {row["task_id"] for row in train}
        val_tasks = {row["task_id"] for row in val}
        fold_holdout_tasks = {row["task_id"] for row in holdout}
        assert not train_tasks & val_tasks
        assert not train_tasks & fold_holdout_tasks
        assert not val_tasks & fold_holdout_tasks
        assert not heldout_tasks & fold_holdout_tasks
        heldout_tasks |= fold_holdout_tasks
        heldout_examples |= {row["example_id"] for row in holdout}
    assert len(heldout_tasks) == 18
    assert len(heldout_examples) == 36
    assert result["source_examples"] == 36
    assert result["test_assets_read"] is False
