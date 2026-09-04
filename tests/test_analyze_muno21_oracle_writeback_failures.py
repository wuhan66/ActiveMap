from __future__ import annotations

import json

from scripts.analyze_muno21_oracle_writeback_failures import main


def test_stratifies_oracle_writeback(tmp_path, monkeypatch) -> None:
    source = tmp_path / "writeback.jsonl"
    rows = []
    for index, gain in enumerate((0.2, -0.1)):
        rows.append(
            {
                "task_id": f"task-{index}",
                "aoi_id": "aoi",
                "budget": 1.5,
                "target": "COMMIT:ADD",
                "split": "val",
                "test_assets_read": False,
                "raster_iou": 0.6 + gain,
                "prior_raster_iou": 0.6,
                "raster_iou_gain": gain,
                "added_change_iou": 0.3,
                "removed_change_iou": 0.0,
                "component_count_absolute_error": 3,
                "prior_component_count_absolute_error": 1,
                "false_edit": False,
                "missed_edit": False,
                "wrong_edit": False,
                "vector_delta_topology_valid": True,
                "mask_artifact": f"mask-{index}.npz",
            }
        )
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    output = tmp_path / "output"
    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_muno21_oracle_writeback_failures.py",
            str(source),
            str(output),
            "--repetitions",
            "20",
        ],
    )

    main()

    result = json.loads((output / "slices.json").read_text())
    assert result["record_count"] == 2
    assert result["summaries"][0]["operation"] == "ADD"
    assert result["summaries"][0]["mean_raster_iou_gain"] == 0.05
    assert result["summaries"][0]["mean_component_error_delta"] == 2
    assert result["test_assets_read"] is False


def test_maps_plain_reject_to_keep() -> None:
    from scripts.analyze_muno21_oracle_writeback_failures import _target_operation

    assert _target_operation({"target": "REJECT"}) == "KEEP"
