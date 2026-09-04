import json

from scripts.sweep_writeback_confidence_gate import sweep


def _row(confidence: float, gain: float) -> dict:
    prior = 0.8
    return {
        "task_id": f"task-{confidence}",
        "budget": 1.5,
        "target": "COMMIT:ADD",
        "effective_operation": "ADD",
        "fusion_weights": [confidence],
        "raster_iou": prior + gain,
        "prior_raster_iou": prior,
        "spent_cost": 0.2,
        "vector_delta_topology_valid": True,
        "test_assets_read": False,
    }


def test_sweep_vetoes_harmful_low_confidence_writeback(tmp_path):
    source = tmp_path / "writeback.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(row)
            for row in (_row(0.3, -0.4), _row(0.9, 0.1))
        )
        + "\n",
        encoding="utf-8",
    )

    result = sweep(source, [0.0, 0.5])

    assert result["selected_confidence_floor"] == 0.5
    selected = next(
        row for row in result["candidates"] if row["confidence_floor"] == 0.5
    )
    assert selected["vetoed_rows"] == 1
    assert selected["retained_change_rate"] == 0.5
    assert selected["raster_iou_gain_auc"] > 0.0
