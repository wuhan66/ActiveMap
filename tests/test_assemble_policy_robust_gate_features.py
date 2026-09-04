import json
from pathlib import Path

import numpy as np

from scripts.assemble_policy_robust_gate_features import assemble_policy_robust_features


def _bundle(root: Path, split: str, advantages: list[float]) -> None:
    root.mkdir(parents=True)
    np.save(root / "features.npy", np.arange(len(advantages) * 2).reshape(-1, 2))
    rows = []
    for index, advantage in enumerate(advantages):
        rows.append(
            {
                "example_id": f"{split}-{index}",
                "task_id": f"task-{index}",
                "split": split,
                "oracle_use_tool": advantage > 0.0,
                "policy_relative_use_tool": advantage > 0.0,
                "policy_relative_advantage": advantage,
                "static_oracle_use_tool": False,
                "consensus_mean_utility_gain": advantage,
                "direct_terminal_valid": True,
                "post_terminal_valid": True,
            }
        )
    (root / "records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (root / "summary.json").write_text(
        json.dumps({"test_assets_read": False}), encoding="utf-8"
    )


def test_policy_robust_target_uses_minimum_snapshot_advantage(tmp_path):
    onpolicy = tmp_path / "onpolicy"
    crossfit = tmp_path / "crossfit"
    _bundle(onpolicy / "train", "train", [1.0, 1.0])
    _bundle(crossfit / "train", "train", [0.5, -0.5])
    _bundle(onpolicy / "val", "val", [1.0])

    result = assemble_policy_robust_features(onpolicy, crossfit, tmp_path / "output")

    assert result["train"]["positive_count"] == 1
    rows = [
        json.loads(line)
        for line in (tmp_path / "output" / "train" / "records.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [row["policy_robust_advantage"] for row in rows] == [0.5, -0.5]
