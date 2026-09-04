import json

import pytest

from scripts.build_sn7_executable_controller_table import build_table


def write_rows(path, gain):
    rows = []
    for task, aoi in (("one", "a"), ("two", "b")):
        for budget in (1.5, 3.0, 4.5):
            rows.append(
                {
                    "task_id": task,
                    "aoi_id": aoi,
                    "budget": budget,
                    "target": "COMMIT:ADD",
                    "split": "val",
                    "test_assets_read": False,
                    "raster_iou": 0.7 + gain,
                    "raster_iou_gain": gain,
                    "added_change_iou": 0.5,
                    "removed_change_iou": 1.0,
                    "added_polygon_iou": 0.5,
                    "removed_polygon_iou": 1.0,
                    "vector_replay_iou": 1.0,
                    "vector_delta_topology_valid": True,
                    "component_count_absolute_error": 0,
                    "episode_utility_v2_balanced": gain,
                    "episode_utility_v2_safety": gain,
                    "episode_utility_v2_cost_aware": gain,
                    "false_edit": False,
                    "missed_edit": False,
                    "wrong_edit": False,
                    "spent_cost": 1.0,
                }
            )
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_executable_table_keeps_absolute_and_paired_metrics(tmp_path):
    reference = tmp_path / "reference.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    write_rows(reference, 0.0)
    write_rows(candidate, 0.1)
    result = build_table(
        {"reference": reference, "candidate": candidate},
        reference="reference",
        candidate="candidate",
        repetitions=20,
        seed=3,
    )
    assert result["record_count"] == 6
    assert result["methods"]["candidate"]["raster_iou_gain_auc"] == pytest.approx(0.1)
    interval = result["candidate_paired_vs_all"]["reference"]["raster_iou_gain_auc"]
    assert interval["delta"] == pytest.approx(0.1)
