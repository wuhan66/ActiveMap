import json

from scripts.summarize_writeback_safety_sweep import summarize


def _row(task: str, iou: float, changed: bool) -> dict:
    return {
        "task_id": task,
        "budget": 1.5,
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
        "episode_utility_v2_balanced": iou - 0.6,
        "episode_utility_v2_safety": iou - 0.6,
        "episode_utility_v2_cost_aware": iou - 0.6,
        "false_edit": False,
        "missed_edit": not changed,
        "wrong_edit": False,
        "spent_cost": 0.2,
        "writeback_changed": changed,
        "fused_confidence": 0.8,
        "test_assets_read": False,
    }


def _write(path, rows):
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )


def test_summary_selects_best_non_degenerate_margin(tmp_path):
    raw = tmp_path / "raw.jsonl"
    good = tmp_path / "good.jsonl"
    degenerate = tmp_path / "degenerate.jsonl"
    _write(raw, [_row("a", 0.6, True), _row("b", 0.6, True)])
    _write(good, [_row("a", 0.8, True), _row("b", 0.7, True)])
    _write(degenerate, [_row("a", 0.9, False), _row("b", 0.9, False)])

    result = summarize(
        {"raw": raw, "good": good, "degenerate": degenerate},
        min_retained_change_rate=0.5,
    )

    assert result["selected_label"] == "good"
    rejected = next(row for row in result["candidates"] if row["label"] == "degenerate")
    assert rejected["feasible"] is False
