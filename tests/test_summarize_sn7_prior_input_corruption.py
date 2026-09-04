import json
from pathlib import Path

import pytest

from scripts.summarize_sn7_prior_input_corruption import summarize


def _write_run(
    root: Path,
    *,
    severity: int,
    seed: int,
    committed_offset: float,
) -> None:
    run = root / f"severity{severity}_model_seed{seed}"
    run.mkdir(parents=True)
    rows = []
    for aoi_id in ("aoi-a", "aoi-b"):
        rows.extend(
            [
                {
                    "sample_id": f"{seed}-{aoi_id}-keep",
                    "aoi_id": aoi_id,
                    "target_edit": "KEEP",
                    "committed_map_iou": 0.8 + committed_offset,
                    "map_iou_delta": 0.1 + committed_offset,
                    "operation_correct": True,
                    "false_edit": False,
                    "missed_edit": False,
                },
                {
                    "sample_id": f"{seed}-{aoi_id}-add",
                    "aoi_id": aoi_id,
                    "target_edit": "ADD",
                    "committed_map_iou": 0.6 + committed_offset,
                    "map_iou_delta": 0.2 + committed_offset,
                    "operation_correct": severity == 0,
                    "false_edit": False,
                    "missed_edit": severity > 0,
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
                "test_assets_read": False,
                "prior_input_corruption": {
                    "max_pixels": severity,
                    "writeback_prior_corrupted": False,
                    "target_geometry_corrupted": False,
                },
            }
        ),
        encoding="utf-8",
    )


def test_summarize_reports_paired_corruption_delta(tmp_path: Path) -> None:
    for severity, offset in ((0, 0.0), (4, -0.05)):
        for seed in (1, 2, 3):
            _write_run(
                tmp_path,
                severity=severity,
                seed=seed,
                committed_offset=offset,
            )

    payload = summarize(
        tmp_path,
        severities=[0, 4],
        model_seeds=[1, 2, 3],
        repetitions=100,
        bootstrap_seed=7,
    )

    corrupted = payload["severities"][1]
    delta = corrupted["delta_vs_zero"]["committed_map_iou"]
    assert delta["observed"] == pytest.approx(-0.05)
    assert delta["ci95_low"] == pytest.approx(-0.05)
    assert delta["ci95_high"] == pytest.approx(-0.05)
    assert corrupted["delta_vs_zero"]["missed_edit_rate"]["observed"] == 1.0
