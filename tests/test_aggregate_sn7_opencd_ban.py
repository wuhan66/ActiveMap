import json
from pathlib import Path

import pytest

from scripts.aggregate_sn7_opencd_ban import METRIC_KEYS, aggregate_runs


def _run(tmp_path: Path, seed: int, delta: float) -> Path:
    run = tmp_path / str(seed)
    run.mkdir()
    summary = {
        "source_commit": "commit",
        "manifest": "manifest",
        "manifest_sha256": "sha",
        "train_count": 10,
        "validation_count": 4,
        "input_contract": "pair",
        "writeback": "xor",
        "selection_metric": "validation committed_map_iou",
        "image_size": 128,
        "batch_size": 8,
        "positive_class_weight": 5.0,
        "lovasz_weight": 0.75,
        "clip_checkpoint_sha256": "clip",
        "side_checkpoint_sha256": "side",
        "test_assets_read": False,
        "seed": seed,
        "best_epoch": 1,
    }
    metrics = {key: delta for key in METRIC_KEYS}
    (run / "summary.json").write_text(json.dumps(summary))
    (run / "history.jsonl").write_text(
        json.dumps({"epoch": 1, "val": metrics}) + "\n"
    )
    return run


def test_aggregate_runs_computes_seed_statistics(tmp_path: Path) -> None:
    result = aggregate_runs(
        [_run(tmp_path, 1, 0.1), _run(tmp_path, 2, 0.3)]
    )
    assert result["aggregate"]["map_iou_delta"]["mean"] == pytest.approx(0.2)
    assert result["run_count"] == 2


def test_aggregate_rejects_test_read(tmp_path: Path) -> None:
    first = _run(tmp_path, 1, 0.1)
    second = _run(tmp_path, 2, 0.2)
    summary = json.loads((second / "summary.json").read_text())
    summary["test_assets_read"] = True
    (second / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="test-free"):
        aggregate_runs([first, second])
