import json
from pathlib import Path

import pytest

from scripts.stratify_agent_writeback_pairs import stratify


def _row(task: str, target: str, aoi: str, offset: float) -> dict:
    return {
        "task_id": task,
        "budget": 3.0,
        "target": target,
        "prediction": target,
        "aoi_id": aoi,
        "split": "val",
        "test_assets_read": False,
        "raster_iou": 0.5 + offset,
        "raster_iou_gain": 0.1 + offset,
        "added_change_iou": 0.4 + offset,
        "removed_change_iou": 0.4 + offset,
        "added_polygon_iou": 0.4 + offset,
        "removed_polygon_iou": 0.4 + offset,
        "vector_replay_iou": 1.0,
        "vector_delta_topology_valid": 1.0,
        "component_count_absolute_error": 1.0,
        "episode_utility_v2_balanced": 0.2 + offset,
        "episode_utility_v2_safety": 0.2 + offset,
        "episode_utility_v2_cost_aware": 0.2 + offset,
        "false_edit": 0.0,
        "missed_edit": 0.0,
        "wrong_edit": 0.0,
        "spent_cost": 0.1,
    }


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_operation_stratification_preserves_paired_delta(tmp_path: Path) -> None:
    pairs = []
    for seed in (1, 2):
        baseline = tmp_path / f"baseline-{seed}.jsonl"
        candidate = tmp_path / f"candidate-{seed}.jsonl"
        rows = [
            _row("keep-task", "REJECT", "aoi-1", 0.0),
            _row("add-task", "COMMIT:ADD", "aoi-2", 0.0),
        ]
        _write(baseline, rows)
        _write(
            candidate,
            [
                _row("keep-task", "REJECT", "aoi-1", 0.1),
                _row("add-task", "COMMIT:ADD", "aoi-2", 0.1),
            ],
        )
        pairs.append((seed, baseline, candidate))

    payload = stratify(pairs, repetitions=200, bootstrap_seed=4)

    assert payload["seed_count"] == 2
    assert payload["operations"]["KEEP"]["record_count"] == 2
    assert payload["operations"]["ADD"]["candidate_minus_baseline"][
        "raster_iou"
    ]["observed_delta"] == pytest.approx(0.1)
    assert payload["operations"]["ADD"]["candidate_minus_baseline"][
        "action_flip"
    ]["observed_delta"] == 0.0
