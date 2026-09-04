import json

import numpy as np
import pytest

from scripts.compare_agent_writebacks import _grouped_delta_draws, _metrics, compare


def _row(task: str, budget: float, raster_iou: float, aoi: str | None = None) -> dict:
    return {
        "task_id": task,
        "aoi_id": aoi or task,
        "budget": budget,
        "target": "COMMIT:ADD",
        "raster_iou": raster_iou,
        "raster_iou_gain": raster_iou - 0.5,
        "added_change_iou": raster_iou,
        "removed_change_iou": 1.0,
        "added_polygon_iou": raster_iou,
        "removed_polygon_iou": 1.0,
        "vector_replay_iou": 1.0,
        "vector_delta_topology_valid": True,
        "component_count_absolute_error": 1.0 - raster_iou,
        "episode_utility_v2_balanced": raster_iou - 0.55,
        "episode_utility_v2_safety": raster_iou - 0.57,
        "episode_utility_v2_cost_aware": raster_iou - 0.53,
        "false_edit": False,
        "missed_edit": False,
        "wrong_edit": False,
        "spent_cost": 0.25,
    }


def _write(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_paired_writeback_bootstrap_detects_map_quality_gain(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    keys = [(f"task-{index}", budget) for index in range(4) for budget in (1.5, 3.0)]
    _write(baseline, [_row(task, budget, 0.6) for task, budget in keys])
    _write(candidate, [_row(task, budget, 0.7) for task, budget in keys])

    result = compare(baseline, candidate, bootstrap=100, seed=7)

    assert result["paired_delta"]["raster_iou_auc"]["ci95_low"] > 0.0
    assert result["paired_delta"]["component_count_absolute_error_auc"]["ci95_high"] < 0.0
    assert result["paired_delta"]["episode_utility_v2_balanced_auc"]["ci95_low"] > 0.0
    assert result["protocol"]["episode_utility_v2_available"] is True
    assert result["test_assets_read"] is False


def test_paired_writeback_can_bootstrap_spatial_aoi_groups(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    keys = [
        (f"task-{index}", "aoi-a" if index < 2 else "aoi-b", budget)
        for index in range(4)
        for budget in (1.5, 3.0)
    ]
    _write(baseline, [_row(task, budget, 0.6, aoi) for task, aoi, budget in keys])
    _write(candidate, [_row(task, budget, 0.7, aoi) for task, aoi, budget in keys])

    result = compare(
        baseline, candidate, bootstrap=100, seed=7, group_key="aoi_id"
    )

    assert result["group_key"] == "aoi_id"
    assert result["group_count"] == 2
    assert result["paired_delta"]["raster_iou_auc"]["ci95_low"] > 0.0


def test_vectorized_grouped_draw_matches_direct_metric_recomputation():
    keys = [
        (f"task-{index}", budget)
        for index in range(4)
        for budget in (1.5, 3.0)
    ]
    baseline = {
        key: _row(*key, 0.5 + 0.01 * index, f"aoi-{index // 2}")
        for index, key in enumerate(keys)
    }
    candidate = {
        key: _row(*key, 0.55 + 0.02 * index, f"aoi-{index // 2}")
        for index, key in enumerate(keys)
    }
    groups = {
        "aoi-0": keys[:4],
        "aoi-1": keys[4:],
    }
    sampled_counts = np.asarray([[2.0, 0.0], [1.0, 1.0], [0.0, 2.0]])
    metric_names = list(_metrics(list(baseline.values())))

    actual = _grouped_delta_draws(
        baseline, candidate, groups, list(groups), sampled_counts, metric_names
    )

    for draw_index, counts in enumerate(sampled_counts.astype(int)):
        sampled_keys = [
            key
            for group_index, group_id in enumerate(groups)
            for _ in range(counts[group_index])
            for key in groups[group_id]
        ]
        left = _metrics([baseline[key] for key in sampled_keys])
        right = _metrics([candidate[key] for key in sampled_keys])
        for name in metric_names:
            assert actual[name][draw_index] == pytest.approx(right[name] - left[name])
