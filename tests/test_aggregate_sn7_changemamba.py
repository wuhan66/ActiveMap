import json
from pathlib import Path

import pytest

from scripts.aggregate_sn7_changemamba import aggregate_runs


def _write_run(
    root: Path,
    *,
    seed: int,
    score: float,
    manifest: str = "frozen.jsonl",
) -> Path:
    run = root / f"seed{seed}"
    run.mkdir()
    summary = {
        "source_commit": "official",
        "manifest": manifest,
        "manifest_sha256": "manifest-sha",
        "train_count": 10,
        "validation_count": 4,
        "best_epoch": 2,
        "seed": seed,
        "input_contract": "old + new",
        "writeback": "xor",
        "image_size": 128,
        "batch_size": 16,
        "learning_rate": 1e-4,
        "weight_decay": 5e-3,
        "positive_class_weight": 5.0,
        "encoder_checkpoint_sha256": "encoder-sha",
        "test_assets_read": False,
    }
    metrics = {
        "prior_map_iou": 0.7,
        "committed_map_iou": score,
        "map_iou_delta": score - 0.7,
        "edited_map_iou": score - 0.1,
        "change_iou": score - 0.2,
        "operation_accuracy": score - 0.1,
        "keep_false_change_fraction": 0.01,
    }
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    rows = [
        {"epoch": 1, "val": metrics},
        {"epoch": 2, "val": metrics},
    ]
    (run / "history.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return run


def test_aggregate_runs_reports_mean_and_sample_std(tmp_path: Path):
    runs = [
        _write_run(tmp_path, seed=1, score=0.8),
        _write_run(tmp_path, seed=2, score=0.9),
        _write_run(tmp_path, seed=3, score=1.0),
    ]

    result = aggregate_runs(runs)

    assert result["run_count"] == 3
    assert result["aggregate"]["committed_map_iou"]["mean"] == pytest.approx(0.9)
    assert result["aggregate"]["committed_map_iou"]["std"] == pytest.approx(0.1)


def test_aggregate_runs_rejects_protocol_mismatch(tmp_path: Path):
    runs = [
        _write_run(tmp_path, seed=1, score=0.8),
        _write_run(tmp_path, seed=2, score=0.9, manifest="other.jsonl"),
    ]

    with pytest.raises(ValueError, match="protocol mismatch"):
        aggregate_runs(runs)
