from __future__ import annotations

import json

import numpy as np

from scripts.sweep_muno21_oracle_delta_components import main


def test_prunes_tiny_add_component_offline(tmp_path, monkeypatch) -> None:
    artifact = tmp_path / "mask.npz"
    prior = np.zeros((8, 8), dtype=np.float32)
    target = prior.copy()
    target[1:4, 1:4] = 1
    committed = target.copy()
    committed[7, 7] = 1
    np.savez_compressed(
        artifact,
        committed_mask=committed,
        prior_mask=prior,
        target_mask=target,
        valid_mask=np.ones_like(prior),
    )
    source = tmp_path / "writeback.jsonl"
    source.write_text(
        json.dumps(
            {
                "task_id": "task",
                "budget": 3.0,
                "target": "COMMIT:ADD",
                "operation": "ADD",
                "split": "val",
                "test_assets_read": False,
                "mask_artifact": str(artifact),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "output"
    monkeypatch.setattr(
        "sys.argv",
        [
            "sweep_muno21_oracle_delta_components.py",
            str(source),
            str(output),
            "--thresholds",
            "0,2",
            "--repetitions",
            "20",
        ],
    )

    main()

    result = json.loads((output / "sweep.json").read_text())
    candidate = next(
        row
        for row in result["candidates"]
        if row["pruning_policy"] == "preserve_largest"
        and row["scope"] == "add_only"
        and row["minimum_pixels"] == 2
    )
    assert candidate["paired_gain_vs_zero"]["observed"] > 0
    assert candidate["missed_edit_rate_delta"] == 0
    assert result["test_assets_read"] is False
