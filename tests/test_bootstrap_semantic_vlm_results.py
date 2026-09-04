import json

from scripts.bootstrap_semantic_vlm_results import paired_bootstrap


def _write(path, seed):
    rows = [
        {
            "example_id": f"e{index}",
            "task_id": f"task-{index}",
            "split": "val",
            "target_operation": target,
            "policy_operation": target,
            "baseline_operation": "KEEP",
            "baseline_utility": -0.75 if target != "KEEP" else 1.0,
            "policy_utility": 1.0,
            "predicted_use_tool": index == seed % 2,
        }
        for index, target in enumerate(["KEEP", "ADD", "DELETE", "RESHAPE"])
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_task_grouped_bootstrap_uses_all_fixed_seeds(tmp_path):
    paths = {1: tmp_path / "seed1.jsonl", 2: tmp_path / "seed2.jsonl"}
    for seed, path in paths.items():
        _write(path, seed)

    result = paired_bootstrap(paths, repetitions=100, bootstrap_seed=7)

    assert result["model_training_seeds"] == [1, 2]
    assert result["task_count"] == 4
    assert result["shared_resample_indices_across_model_seeds"] is True
    assert (
        result["fixed_seed_mean_intervals"]["operation_macro_f1_delta"]
        ["observed_fixed_seed_mean"]
        > 0.0
    )
    assert result["test_assets_read"] is False
