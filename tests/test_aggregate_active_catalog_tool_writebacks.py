import json

from scripts.aggregate_active_catalog_tool_writebacks import aggregate


def _row(task, aoi, iou):
    return {
        "task_id": task,
        "aoi_id": aoi,
        "budget": 2.0,
        "target": "COMMIT:ADD",
        "raster_iou": iou,
        "raster_iou_gain": iou - 0.5,
        "added_change_iou": iou,
        "removed_change_iou": 1.0,
        "added_polygon_iou": iou,
        "removed_polygon_iou": 1.0,
        "vector_replay_iou": 1.0,
        "vector_delta_topology_valid": True,
        "component_count_absolute_error": 1.0 - iou,
    }


def _write(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_writeback_aggregate_uses_shared_aoi_draws(tmp_path):
    baselines = {}
    candidates = {}
    for seed in (1, 2):
        baselines[seed] = tmp_path / f"b-{seed}.jsonl"
        candidates[seed] = tmp_path / f"c-{seed}.jsonl"
        _write(baselines[seed], [_row("e1", "a", 0.6), _row("e2", "b", 0.6)])
        _write(candidates[seed], [_row("e1", "a", 0.8), _row("e2", "b", 0.8)])
    result = aggregate(baselines, candidates, repetitions=50, seed=9)
    assert result["seed_count"] == 2
    assert result["shared_aoi_resampling_across_seeds"] is True
    assert result["paired_delta"]["raster_iou_gain_auc"]["ci95_low"] > 0.0
