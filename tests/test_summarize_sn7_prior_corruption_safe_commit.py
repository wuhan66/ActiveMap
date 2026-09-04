import json
from pathlib import Path

import pytest

from scripts.summarize_sn7_prior_corruption_safe_commit import summarize


def _write_run(root: Path, severity: int, seed: int) -> None:
    run = root / f"severity{severity}_seed{seed}"
    run.mkdir(parents=True)
    rows = []
    for aoi in ("a", "b"):
        rows.extend(
            [
                {
                    "sample_id": f"{seed}-{severity}-{aoi}-keep",
                    "aoi_id": aoi,
                    "split": "val",
                    "target_edit": "KEEP",
                    "predicted_edit": "ADD",
                    "safe_commit_predicted_edit": "KEEP",
                    "prior_map_iou": 0.8,
                    "committed_map_iou": 0.6 - severity * 0.01,
                    "safe_commit_map_iou": 0.8,
                },
                {
                    "sample_id": f"{seed}-{severity}-{aoi}-add",
                    "aoi_id": aoi,
                    "split": "val",
                    "target_edit": "ADD",
                    "predicted_edit": "ADD",
                    "safe_commit_predicted_edit": "ADD",
                    "prior_map_iou": 0.5,
                    "committed_map_iou": 0.9 - severity * 0.01,
                    "safe_commit_map_iou": 0.9 - severity * 0.01,
                },
            ]
        )
    with (run / "per_sample.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    (run / "summary.json").write_text(
        json.dumps(
            {
                "sample_count": len(rows),
                "split": "val",
                "selected_threshold": 0.5,
                "test_assets_read": False,
            }
        ),
        encoding="utf-8",
    )


def test_safe_commit_summary_uses_frozen_paired_rows(tmp_path: Path) -> None:
    for severity in (0, 4):
        for seed in (1, 2, 3):
            _write_run(tmp_path, severity, seed)

    payload = summarize(
        tmp_path,
        severities=[0, 4],
        model_seeds=[1, 2, 3],
        repetitions=100,
        bootstrap_seed=9,
    )

    shifted = payload["severities"][1]
    false_edit = shifted["safe_minus_direct"]["false_edit_rate"]
    assert false_edit["observed"] == -1.0
    assert false_edit["ci95_low"] == -1.0
    assert false_edit["ci95_high"] == -1.0
    assert shifted["safe_minus_direct"]["committed_map_iou"]["observed"] == (
        pytest.approx(0.12)
    )
    assert payload["protocol"]["threshold_recalibrated_on_corruption"] is False
