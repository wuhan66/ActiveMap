from scripts.bootstrap_single_policy_evaluation import single_seed_bootstrap


def test_single_seed_bootstrap_reports_grouped_delta_intervals():
    rows = []
    for task_index in range(4):
        rows.append(
            {
                "example_id": f"example-{task_index}",
                "task_id": f"task-{task_index}",
                "split": "val",
                "target_operation": "ADD",
                "direct_operation": "KEEP",
                "policy_operation": "ADD" if task_index < 3 else "KEEP",
                "direct_utility": 0.0,
                "policy_utility": 1.0 if task_index < 3 else 0.0,
                "predicted_use_tool": task_index < 3,
            }
        )
    result = single_seed_bootstrap(rows, repetitions=100, seed=9)

    assert result["task_count"] == 4
    assert result["intervals"]["mean_utility_delta"]["observed"] == 0.75
    assert 0.0 <= result["intervals"]["mean_utility_delta"]["bootstrap_probability_gt_zero"] <= 1.0
