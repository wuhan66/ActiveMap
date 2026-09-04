from __future__ import annotations

import json
from pathlib import Path

from scripts.build_muno21_baseline_table import main


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_builds_aligned_controller_writeback_table(tmp_path, monkeypatch) -> None:
    rollout = tmp_path / "rollout.json"
    writeback = tmp_path / "writeback.json"
    output = tmp_path / "output"
    _write(
        rollout,
        {
            "protocol": {"split": "val", "test_assets_read": False},
            "results": [
                {
                    "method": "selector",
                    "budget": 1.5,
                    "terminal_accuracy": 0.8,
                    "false_edit_rate": 0.1,
                    "missed_edit_rate": 0.2,
                    "mean_acquisitions": 0.3,
                    "mean_cost": 0.4,
                    "mean_quality_cost_utility": 0.05,
                }
            ],
        },
    )
    _write(
        writeback,
        {
            "protocol": {"test_assets_read": False},
            "budgets": [
                {
                    "budget": 1.5,
                    "sample_count": 8,
                    "mean_raster_iou": 0.7,
                    "mean_prior_raster_iou": 0.6,
                    "mean_raster_iou_gain": 0.1,
                    "mean_episode_utility_v2_balanced": 0.02,
                    "mean_episode_utility_v2_safety": 0.01,
                    "mean_episode_utility_v2_cost_aware": 0.0,
                }
            ],
        },
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_muno21_baseline_table.py",
            str(output),
            "--rollout-summary",
            str(rollout),
            "--writeback",
            f"selector={writeback}",
            "--budget",
            "1.5",
        ],
    )

    main()

    table = json.loads((output / "table.json").read_text())
    assert table["method_count"] == 1
    assert table["row_count"] == 1
    assert table["rows"][0]["mean_raster_iou_gain"] == 0.1
    assert table["test_assets_read"] is False
