import json
from pathlib import Path

import pytest

from scripts.aggregate_sn7_frozen_test_operation_slices import aggregate, render


def _row(
    task_id: str,
    aoi: str,
    operation: str,
    *,
    quality: float,
    cost: float,
) -> dict:
    target = "REJECT" if operation == "KEEP" else f"COMMIT:{operation}"
    return {
        "task_id": task_id,
        "aoi_id": aoi,
        "budget": 1.5,
        "target": target,
        "split": "test",
        "test_assets_read": True,
        "map_quality_before": 0.5,
        "map_quality_after": quality,
        "false_edit": operation == "KEEP" and quality < 0.5,
        "missed_edit": operation != "KEEP" and quality <= 0.5,
        "wrong_edit": False,
        "spent_cost": cost,
        "episode_utility_v2_balanced": quality - cost,
        "episode_utility_v2_safety": quality - cost,
        "episode_utility_v2_cost_aware": quality - cost,
        "vector_delta_topology_valid": True,
    }


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def _pairs(tmp_path: Path) -> list[tuple[int, Path, Path]]:
    records = []
    operations = ("KEEP", "ADD", "DELETE", "RESHAPE")
    for seed in (11, 12, 13):
        reference = []
        candidate = []
        for aoi_index, aoi in enumerate(("aoi-a", "aoi-b")):
            for operation_index, operation in enumerate(operations):
                task_id = f"{aoi}-{operation.lower()}"
                baseline_quality = 0.55 + 0.01 * operation_index
                reference.append(
                    _row(task_id, aoi, operation, quality=baseline_quality, cost=0.0)
                )
                candidate.append(
                    _row(
                        task_id,
                        aoi,
                        operation,
                        quality=baseline_quality + 0.02 + 0.001 * aoi_index,
                        cost=0.1,
                    )
                )
        records.append(
            (
                seed,
                _write(tmp_path / f"reference_{seed}.jsonl", reference),
                _write(tmp_path / f"candidate_{seed}.jsonl", candidate),
            )
        )
    return records


def test_operation_slices_are_paired_and_bootstrapped(tmp_path: Path) -> None:
    payload = aggregate(
        _pairs(tmp_path),
        reference_label="no-tool",
        candidate_label="benefit",
        repetitions=100,
        seed=7,
    )

    assert payload["split"] == "test"
    assert payload["analysis_role"] == "posthoc_descriptive_only"
    assert payload["promotion_decision_used"] is False
    for operation in ("KEEP", "ADD", "DELETE", "RESHAPE"):
        row = payload["operations"][operation]
        assert row["support_per_seed"] == 2
        assert row["aoi_count"] == 2
        assert row["candidate_minus_reference_mean"]["map_quality_after"] == pytest.approx(
            0.0205
        )
        assert row["candidate_minus_reference_aoi_bootstrap"] is not None
    assert "post-hoc descriptive" in render(payload)


def test_operation_slices_reject_non_test_rows(tmp_path: Path) -> None:
    pairs = _pairs(tmp_path)
    _, _, candidate = pairs[0]
    rows = [json.loads(line) for line in candidate.read_text(encoding="utf-8").splitlines()]
    rows[0]["split"] = "val"
    _write(candidate, rows)

    with pytest.raises(ValueError, match="non-test row"):
        aggregate(
            pairs,
            reference_label="no-tool",
            candidate_label="benefit",
            repetitions=10,
            seed=7,
        )


def test_operation_slices_reject_misaligned_pairs(tmp_path: Path) -> None:
    pairs = _pairs(tmp_path)
    _, _, candidate = pairs[0]
    rows = [json.loads(line) for line in candidate.read_text(encoding="utf-8").splitlines()]
    rows[0]["task_id"] = "different"
    _write(candidate, rows)

    with pytest.raises(ValueError, match="not aligned"):
        aggregate(
            pairs,
            reference_label="no-tool",
            candidate_label="benefit",
            repetitions=10,
            seed=7,
        )
