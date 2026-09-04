import json
from pathlib import Path

from scripts.aggregate_muno21_official_three_seeds import aggregate


def _write(path: Path, offset: float, error: float) -> None:
    rows = []
    for budget in (1.5, 3.0):
        for task_id, base in (("a", 0.1), ("b", 0.2)):
            rows.append(
                {
                    "task_id": task_id,
                    "budget": budget,
                    "apls_improvement": base + offset,
                    "pixel_f1_improvement": base + 2 * offset,
                    "official_unit": "change_scenario",
                }
            )
        rows.append(
            {
                "task_id": "__aggregate__",
                "budget": budget,
                "no_change_error_rate": error,
                "official_unit": "official_nochange_aggregate",
            }
        )
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_aggregate_preserves_seed_and_task_structure(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    seed1 = tmp_path / "seed1.jsonl"
    seed2 = tmp_path / "seed2.jsonl"
    seed3 = tmp_path / "seed3.jsonl"
    _write(baseline, 0.0, 0.2)
    _write(seed1, 0.01, 0.1)
    _write(seed2, 0.02, 0.2)
    _write(seed3, 0.03, 0.3)

    result = aggregate(
        baseline,
        [("seed1", seed1), ("seed2", seed2), ("seed3", seed3)],
        repetitions=200,
        seed=7,
    )

    apls = result["aggregate"]["paired_metrics"]["apls_improvement"]
    pixel = result["aggregate"]["paired_metrics"]["pixel_f1_improvement"]
    assert abs(apls["mean_delta"] - 0.02) < 1e-12
    assert abs(pixel["mean_delta"] - 0.04) < 1e-12
    assert abs(result["aggregate"]["mean_no_change_error_rate_delta"]) < 1e-12
    assert result["protocol"]["test_assets_read"] is False
