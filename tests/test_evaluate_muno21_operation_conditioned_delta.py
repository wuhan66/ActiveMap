from __future__ import annotations

import json

import numpy as np

from scripts.evaluate_muno21_operation_conditioned_delta import main


def test_operation_conditioned_replay_improves_without_missing_edit(
    tmp_path, monkeypatch
) -> None:
    rows = []
    for task_id, operation in (("add", "ADD"), ("delete", "DELETE")):
        prior = np.zeros((12, 12), dtype=np.float32)
        target = np.zeros_like(prior)
        if operation == "ADD":
            target[2:8, 2:8] = 1
            committed = target.copy()
            committed[11, 11] = 1
        else:
            prior[1:11, 1:11] = 1
            target = prior.copy()
            target[1:3, 1:11] = 0
            committed = target.copy()
            committed[9, 9] = 0
        artifact = tmp_path / f"{task_id}.npz"
        np.savez_compressed(
            artifact,
            committed_mask=committed,
            prior_mask=prior,
            target_mask=target,
            valid_mask=np.ones_like(prior),
        )
        rows.append(
            {
                "task_id": task_id,
                "budget": 3.0,
                "target": f"COMMIT:{operation}",
                "operation": operation,
                "split": "val",
                "test_assets_read": False,
                "mask_artifact": str(artifact),
            }
        )

    source = tmp_path / "writeback.jsonl"
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    output = tmp_path / "output"
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_muno21_operation_conditioned_delta.py",
            str(source),
            str(output),
            "--add-threshold",
            "2",
            "--delete-threshold",
            "2",
            "--repetitions",
            "50",
        ],
    )

    main()

    result = json.loads((output / "summary.json").read_text())
    assert result["overall"]["paired_raster_iou_delta"]["observed"] > 0
    assert result["overall"]["missed_edit_rate_delta"] == 0
    assert result["promotion_gate_passed"] is True
    assert result["test_assets_read"] is False
    assert (output / "table.csv").is_file()
    assert (output / "table.md").is_file()
