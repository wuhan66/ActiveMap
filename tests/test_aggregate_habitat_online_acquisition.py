from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def load_aggregator():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "aggregate_habitat_online_acquisition.py"
    )
    spec = importlib.util.spec_from_file_location("aggregate_habitat_online_acquisition", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_summary(
    root: Path,
    name: str,
    policy: str,
    *,
    score: int = 8,
    seed: int = 1,
    calls: int = 2,
    quality: float = 0.8,
) -> None:
    directory = root / name
    directory.mkdir()
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "activemap-habitat-online-acquisition-pilot-v1",
                "development_only": True,
                "policy": policy,
                "seed": seed,
                "min_unknown_score": score,
                "min_novelty_score": score,
                "post_initial_acquisitions": calls,
                "final_known_cell_fraction": 0.5,
                "final_reference_map_quality": quality if policy != "acquire_all" else None,
                "reference_known_coverage": quality,
                "occupied_iou": quality,
                "free_iou": quality,
                "balanced_iou": quality,
                "false_free_rate": 1.0 - quality,
                "false_obstacle_rate": 1.0 - quality,
                "reference_covered_cells_per_sensor_call": 10.0,
                "success": True,
                "spl": 1.0,
                "collision_count": 0,
            }
        ),
        encoding="utf-8",
    )


def test_aggregate_groups_gated_policies_by_threshold(tmp_path: Path) -> None:
    module = load_aggregator()
    _write_summary(tmp_path, "all", "acquire_all")
    _write_summary(tmp_path, "gate_a", "unknown_gate", score=15)
    _write_summary(tmp_path, "gate_b", "unknown_gate", score=15)

    rows = module.aggregate(tmp_path)

    assert [row["policy"] for row in rows] == ["acquire_all", "unknown_gate@15"]
    assert rows[1]["episode_count"] == 2
    assert rows[1]["final_reference_map_quality"] == 0.8


def test_aggregate_ignores_prior_aggregate_outputs(tmp_path: Path) -> None:
    module = load_aggregator()
    _write_summary(tmp_path, "all", "acquire_all")
    aggregate_dir = tmp_path / "aggregate"
    aggregate_dir.mkdir()
    (aggregate_dir / "summary.json").write_text(
        json.dumps({"schema_version": "activemap-habitat-online-acquisition-aggregate-v1"}),
        encoding="utf-8",
    )

    rows = module.aggregate(tmp_path)

    assert len(rows) == 1
    assert rows[0]["policy"] == "acquire_all"


def test_aggregate_labels_novelty_gate_by_novelty_threshold(tmp_path: Path) -> None:
    module = load_aggregator()
    directory = tmp_path / "novelty"
    directory.mkdir()
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "activemap-habitat-online-acquisition-pilot-v1",
                "development_only": True,
                "policy": "novelty_gate",
                "min_unknown_score": 8,
                "min_novelty_score": 6,
                "post_initial_acquisitions": 2,
                "final_known_cell_fraction": 0.5,
                "final_reference_map_quality": 0.8,
                "success": True,
                "spl": 1.0,
                "collision_count": 0,
            }
        ),
        encoding="utf-8",
    )

    rows = module.aggregate(tmp_path)

    assert rows[0]["policy"] == "novelty_gate@6"


def test_paired_bootstrap_matches_gate_and_reference_by_seed(tmp_path: Path) -> None:
    module = load_aggregator()
    for seed, reference_calls, gate_calls, quality in ((1, 5, 3, 0.9), (2, 7, 4, 0.8)):
        _write_summary(tmp_path, f"all-{seed}", "acquire_all", seed=seed, calls=reference_calls)
        _write_summary(
            tmp_path,
            f"gate-{seed}",
            "novelty_gate",
            seed=seed,
            calls=gate_calls,
            quality=quality,
        )

    reports = module.paired_bootstrap(tmp_path, repetitions=100, seed=3)

    assert len(reports) == 1
    assert reports[0]["policy"] == "novelty_gate@8"
    assert reports[0]["calls_saved_vs_acquire_all"]["mean"] == 2.5
    assert reports[0]["reference_map_quality"]["mean"] == pytest.approx(0.85)
    assert reports[0]["occupied_iou"]["mean"] == pytest.approx(0.85)
    assert reports[0]["false_free_rate"]["mean"] == pytest.approx(0.15)


def test_policy_contrast_reports_candidate_minus_baseline(tmp_path: Path) -> None:
    module = load_aggregator()
    for seed, novelty_calls, novelty_quality, unknown_calls, unknown_quality in (
        (1, 3, 0.9, 5, 0.8),
        (2, 4, 0.8, 5, 0.9),
    ):
        _write_summary(
            tmp_path,
            f"novelty-{seed}",
            "novelty_gate",
            seed=seed,
            score=8,
            calls=novelty_calls,
            quality=novelty_quality,
        )
        _write_summary(
            tmp_path,
            f"unknown-{seed}",
            "unknown_gate",
            seed=seed,
            score=15,
            calls=unknown_calls,
            quality=unknown_quality,
        )

    report = module.paired_policy_contrast(tmp_path, repetitions=100, seed=5)

    assert report is not None
    assert report["difference_definition"] == "candidate_minus_baseline"
    assert report["metrics"]["post_initial_acquisitions"]["mean_difference"] == -1.5
    assert report["metrics"]["final_reference_map_quality"]["mean_difference"] == pytest.approx(0.0)
