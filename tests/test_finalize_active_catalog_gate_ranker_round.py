from __future__ import annotations

import json

from scripts.evaluate_active_catalog_selector import active_catalog_metrics
from scripts.finalize_active_catalog_gate_ranker_round import RunSpec, finalize


def _rows(calls: set[int], false_calls: set[int] | None = None):
    false_calls = false_calls or set()
    rows = []
    for index in range(8):
        target_call = index % 2 == 0
        predicted_call = index in calls or index in false_calls
        policy_utility = 0.2 if target_call and predicted_call else 0.0
        if not target_call and predicted_call:
            policy_utility = -0.2
        oracle_utility = 0.2 if target_call else 0.0
        rows.append(
            {
                "example_id": f"example-{index}",
                "aoi_id": f"aoi-{(index // 2) % 2}",
                "source_episode": f"episode-{index}",
                "candidate_count": 4,
                "target_selection": "ACQUIRE" if target_call else "STOP",
                "target_evidence_id": f"best-{index}" if target_call else None,
                "predicted_selection": "ACQUIRE" if predicted_call else "STOP",
                "predicted_evidence_id": f"best-{index}" if predicted_call else None,
                "valid_action": True,
                "stop_utility": 0.0,
                "oracle_utility": oracle_utility,
                "policy_utility": policy_utility,
                "policy_cost": 0.2 if predicted_call else 0.0,
                "regret": oracle_utility - policy_utility,
            }
        )
    return rows


def _write_run(tmp_path, label, target, seed, rows):
    folder = tmp_path / label
    folder.mkdir()
    trace = folder / "traces.jsonl"
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    summary = {
        "schema_version": "active-catalog-visual-gate-ranker-evaluation-v1",
        "valid_action_rate": 1.0,
        "metrics": active_catalog_metrics(rows),
        "test_assets_read": False,
    }
    (folder / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return RunSpec(label, target, seed, trace)


def test_finalizer_separates_tuning_target_from_three_seed_replication(tmp_path):
    t10_rows = _rows({0, 2, 4, 6})
    specs = [
        _write_run(tmp_path, "t05_seed1", 0.05, 20260720, _rows({0, 2})),
        _write_run(tmp_path, "t10_seed1", 0.10, 20260720, t10_rows),
        _write_run(tmp_path, "t10_seed2", 0.10, 20260721, t10_rows),
        _write_run(tmp_path, "t10_seed3", 0.10, 20260722, t10_rows),
        _write_run(tmp_path, "t15_seed1", 0.15, 20260720, _rows({0, 2, 4})),
        _write_run(
            tmp_path,
            "t20_seed1",
            0.20,
            20260720,
            _rows({0, 2, 4, 6}, {1, 3}),
        ),
    ]

    report = finalize(
        specs,
        tuning_seed=20260720,
        repetitions=100,
        bootstrap_seed=7,
    )

    assert report["selected_tuning_run"] == "t10_seed1"
    assert report["selected_acquire_target"] == 0.10
    assert report["target_aggregates"]["0.1"]["seed_count"] == 3
    assert report["promotion"]["passed"] is True
    assert report["runs"]["t20_seed1"]["safety_feasible"] is False


def test_finalizer_does_not_promote_unreplicated_selected_target(tmp_path):
    specs = [
        _write_run(tmp_path, "t05_seed1", 0.05, 20260720, _rows({0, 2})),
        _write_run(tmp_path, "t15_seed1", 0.15, 20260720, _rows({0, 2, 4, 6})),
    ]

    report = finalize(
        specs,
        tuning_seed=20260720,
        repetitions=20,
        bootstrap_seed=9,
    )

    assert report["selected_acquire_target"] == 0.15
    assert report["promotion"]["passed"] is False
    assert report["promotion"]["checks"]["three_or_more_independent_seeds"] is False
