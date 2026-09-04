from scripts.analyze_policy_relative_branch_cache import task_bootstrap


def test_task_bootstrap_is_grouped_and_deterministic():
    rows = []
    for task_index in range(4):
        for example_index in range(2):
            direct = float(example_index)
            post = direct + (0.5 if task_index % 2 == 0 else -0.5)
            rows.append(
                {
                    "task_id": f"task-{task_index}",
                    "static_use_tool": task_index == 0,
                    "policy_relative_use_tool": post > direct,
                    "direct_utility": direct,
                    "post_tool_utility": post,
                }
            )
    first = task_bootstrap(rows, repetitions=50, seed=3)
    second = task_bootstrap(rows, repetitions=50, seed=3)

    assert first == second
    assert first["unit"] == "task_id"
    assert first["task_count"] == 4
    interval = first["intervals"]["policy_relative_oracle_gain_over_direct"]
    assert interval["low_95"] <= interval["mean"] <= interval["high_95"]
