import json
from pathlib import Path

import pytest

from scripts.aggregate_muno21_direct_vlm_sft import aggregate


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run(root: Path, seed: int, *, target: str = "KEEP") -> None:
    _write_json(
        root / "COMPLETE.json",
        {"status": "complete", "test_assets_read": False},
    )
    _write_json(
        root / "evaluation/summary.json",
        {
            "protocol": {"split": "val", "image_mode": "composite"},
            "adapter": f"/adapter/{seed}",
            "sample_count": 1,
            "operation_metrics": {"accuracy": 0.8, "macro_f1": 0.7},
            "false_edit_rate": 0.1,
            "missed_edit_rate": 0.2,
            "schema_valid_rate": 1.0,
            "test_assets_read": False,
        },
    )
    traces = root / "evaluation/traces.jsonl"
    traces.write_text(
        json.dumps({"example_id": "e1", "target_operation": target}) + "\n",
        encoding="utf-8",
    )
    metrics = {
        "mean_raster_iou": 0.8,
        "mean_raster_iou_gain": 0.01,
        "mean_vector_replay_iou": 1.0,
        "vector_delta_topology_valid_rate": 1.0,
        "mean_episode_utility_v2_balanced": 0.0,
        "mean_episode_utility_v2_safety": -0.1,
        "mean_episode_utility_v2_cost_aware": -0.2,
    }
    _write_json(
        root / "writeback/safe_delta/summary.json",
        {"budgets": [{"budget": 3.0, **metrics}]},
    )


def test_aggregate_requires_identical_validation_support(tmp_path: Path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    _run(first, 1)
    _run(second, 2)
    result = aggregate({1: first, 2: second}, budget=3.0)
    assert result["seed_count"] == 2
    assert result["paired_example_count"] == 1
    assert result["aggregate_metrics"]["accuracy"]["mean"] == pytest.approx(0.8)
    assert result["test_assets_read"] is False

    _run(second, 2, target="ADD")
    with pytest.raises(ValueError, match="identical paired validation support"):
        aggregate({1: first, 2: second}, budget=3.0)
