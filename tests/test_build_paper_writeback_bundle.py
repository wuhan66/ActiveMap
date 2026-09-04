from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_paper_result_bundles import audit_result_bundles
from scripts.build_paper_writeback_bundle import build_writeback_bundle


def _write(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_writeback_bundle_uses_metric_specific_paired_task_support(tmp_path: Path) -> None:
    writeback_paths = {}
    official_paths = {}
    for seed_index, seed in enumerate((20260821, 20260822, 20260823)):
        writeback_paths[str(seed)] = _write(
            tmp_path / f"writeback-{seed}.jsonl",
            [
                {
                    "task_id": task,
                    "budget": 3.0,
                    "target": target,
                    "raster_iou": 0.7 + 0.01 * seed_index,
                    "added_polygon_iou": 0.6,
                    "removed_polygon_iou": 0.4,
                    "vector_delta_topology_valid": True,
                    "vector_replay_iou": 0.99,
                }
                for task, target in (
                    ("change-a", "COMMIT:ADD"),
                    ("change-b", "COMMIT:DELETE"),
                    ("keep-a", "REJECT"),
                    ("keep-b", "REJECT"),
                )
            ],
        )
        official_paths[str(seed)] = _write(
            tmp_path / f"official-{seed}.jsonl",
            [
                {
                    "task_id": "change-a",
                    "budget": 3.0,
                    "apls_improvement": 0.4,
                    "pixel_f1_improvement": 0.5,
                },
                {
                    "task_id": "change-b",
                    "budget": 3.0,
                    "apls_improvement": 0.6,
                    "pixel_f1_improvement": 0.7,
                },
                {"task_id": "keep-a", "budget": 3.0, "no_change_error_rate": 0.0},
                {"task_id": "keep-b", "budget": 3.0, "no_change_error_rate": 1.0},
            ],
        )
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"status": "complete", "returncode": 0}), encoding="utf-8")
    bundle = build_writeback_bundle(
        Path("configs/experiments/paper_registry.yaml"),
        writeback_paths,
        official_paths,
        variant="agent_tool_to_belief",
        budget=3.0,
        frozen_test_ledger=ledger,
    )
    assert bundle["unit_count"] == 4
    assert bundle["metric_unit_counts"]["apls_improvement"] == 2
    assert bundle["metric_unit_counts"]["no_change_error_rate"] == 2
    assert bundle["metrics"]["apls_improvement"]["mean"] == pytest.approx(0.5)
    assert bundle["metrics"]["no_change_error_rate"]["mean"] == pytest.approx(0.5)
    bundles = tmp_path / "bundles"
    bundles.mkdir()
    (bundles / "writeback.json").write_text(json.dumps(bundle), encoding="utf-8")
    report = audit_result_bundles(Path("configs/experiments/paper_registry.yaml"), bundles)
    assert "executable_vector_writeback/agent_tool_to_belief@3" not in report["cell_errors"]


def test_writeback_bundle_bootstraps_official_aggregate_error_across_seeds(
    tmp_path: Path,
) -> None:
    writeback_paths = {}
    official_paths = {}
    expected_errors = [0.1, 0.2, 0.3]
    for seed, error_rate in zip(
        (20260821, 20260822, 20260823), expected_errors, strict=True
    ):
        writeback_paths[str(seed)] = _write(
            tmp_path / f"writeback-aggregate-{seed}.jsonl",
            [
                {
                    "task_id": task,
                    "budget": 3.0,
                    "target": "COMMIT:ADD",
                    "raster_iou": 0.7,
                    "added_polygon_iou": 0.6,
                    "removed_polygon_iou": 0.4,
                    "vector_delta_topology_valid": True,
                    "vector_replay_iou": 0.99,
                }
                for task in ("change-a", "change-b")
            ],
        )
        official_paths[str(seed)] = _write(
            tmp_path / f"official-aggregate-{seed}.jsonl",
            [
                {
                    "task_id": task,
                    "budget": 3.0,
                    "apls_improvement": 0.5,
                    "pixel_f1_improvement": 0.6,
                }
                for task in ("change-a", "change-b")
            ]
            + [
                {
                    "task_id": "__aggregate__",
                    "budget": 3.0,
                    "no_change_error_rate": error_rate,
                }
            ],
        )
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"status": "complete", "returncode": 0}))
    bundle = build_writeback_bundle(
        Path("configs/experiments/paper_registry.yaml"),
        writeback_paths,
        official_paths,
        variant="agent_tool_to_belief",
        budget=3.0,
        frozen_test_ledger=ledger,
    )
    error = bundle["metrics"]["no_change_error_rate"]
    assert error["mean"] == pytest.approx(0.2)
    assert error["unit_count"] == 3
    assert error["bootstrap_unit"] == "model_seed_official_aggregate"
