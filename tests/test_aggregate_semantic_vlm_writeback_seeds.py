import json

from scripts.aggregate_semantic_vlm_writeback_seeds import aggregate, grouped_bootstrap


def _rows(delta=0.1):
    rows = []
    for task, example in (("a", "1"), ("a", "2"), ("b", "3")):
        rows.append(
            {
                "task_id": task,
                "source_example_id": example,
                "raster_iou": 0.7,
                "prior_raster_iou": 0.6,
                "raster_iou_gain": delta,
                "added_change_iou": 0.5,
                "removed_change_iou": 0.5,
                "added_polygon_iou": 0.5,
                "removed_polygon_iou": 0.5,
                "vector_replay_iou": 0.99,
                "component_count_absolute_error": 1.0,
                "prior_component_count_absolute_error": 2.0,
                "vector_delta_topology_valid": True,
            }
        )
    return rows


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_grouped_bootstrap_counts_tasks_not_rows():
    result = grouped_bootstrap(_rows(), "raster_iou_gain", repetitions=100, seed=1)
    assert result["task_group_count"] == 2


def test_writeback_aggregate_requires_all_seeds(tmp_path):
    paths = {}
    for seed, delta in ((16, 0.1), (19, 0.1), (22, -0.1)):
        path = tmp_path / f"{seed}.jsonl"
        _write(path, _rows(delta))
        paths[seed] = path
    result = aggregate(paths, repetitions=100, bootstrap_seed=1)
    assert result["all_writeback_gates_passed"] is False
    assert result["per_seed"][2]["failed_gates"] == ["nonnegative_raster_gain"]
