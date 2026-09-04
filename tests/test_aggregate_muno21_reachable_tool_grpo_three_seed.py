import json

from scripts.aggregate_muno21_reachable_tool_grpo_three_seed import aggregate


def _report(tmp_path, model_seed, bootstrap_seed, delta):
    path = tmp_path / f"seed{model_seed}.json"
    path.write_text(
        json.dumps(
            {
                "protocol": {
                    "paired": True,
                    "test_assets_read": False,
                    "seed": bootstrap_seed,
                    "bootstrap_seed": bootstrap_seed,
                    "model_seed": model_seed,
                    "bootstrap": 100,
                    "budget_coverage": [3.0, 4.5],
                },
                "budgets": [3.0, 4.5],
                "task_count": 11,
                "sample_count": 22,
                "paired_delta": {
                    "episode_utility_v2_balanced_auc": {
                        "delta": delta,
                        "ci95_low": delta - 0.01,
                        "ci95_high": delta + 0.01,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_aggregate_keeps_model_and_bootstrap_seeds_distinct(tmp_path):
    reports = [
        _report(tmp_path, 20260951 + index, 20261061 + index, 0.01 * index)
        for index in range(3)
    ]

    result = aggregate(reports)

    assert result["protocol"]["model_seeds"] == [20260951, 20260952, 20260953]
    assert result["protocol"]["bootstrap_seeds"] == [20261061, 20261062, 20261063]
