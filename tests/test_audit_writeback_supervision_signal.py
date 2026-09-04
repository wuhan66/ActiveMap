import json

from scripts.audit_writeback_supervision_signal import audit


def test_audit_detects_missing_raster_change_signal(tmp_path):
    source = tmp_path / "writeback.jsonl"
    rows = [
        {
            "task_id": "a",
            "target": "COMMIT:ADD",
            "prior_raster_iou": 1.0,
            "test_assets_read": False,
        },
        {
            "task_id": "b",
            "target": "COMMIT:DELETE",
            "prior_raster_iou": 1.0,
            "test_assets_read": False,
        },
    ]
    source.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    result = audit(source)

    assert result["has_executable_change_supervision"] is False
    assert result["prior_target_raster_iou"]["exact_match_rows"] == 2
