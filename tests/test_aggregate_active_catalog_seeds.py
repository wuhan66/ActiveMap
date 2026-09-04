import json

from scripts.aggregate_active_catalog_seeds import aggregate


def _rows(seed):
    rows = []
    for index, target in enumerate(("ACQUIRE", "STOP", "ACQUIRE", "STOP")):
        predicted = target if seed == 1 or index != 2 else "STOP"
        policy_utility = 0.2 if predicted == "ACQUIRE" and target == "ACQUIRE" else 0.0
        row = {
            "example_id": f"e-{index}",
            "task_id": f"task-{index}",
            "aoi_id": f"aoi-{index % 2}",
            "source_episode": f"episode-{index}",
            "split": "val",
            "budget": 1.5,
            "gt_edit": "RESHAPE" if target == "ACQUIRE" else "KEEP",
            "target_selection": target,
            "target_evidence_id": "candidate" if target == "ACQUIRE" else None,
            "predicted_selection": predicted,
            "predicted_evidence_id": "candidate" if predicted == "ACQUIRE" else None,
            "candidate_count": 4,
            "stop_utility": 0.0,
            "oracle_utility": 0.2 if target == "ACQUIRE" else 0.0,
            "policy_utility": policy_utility,
            "policy_cost": 1.0 if predicted == "ACQUIRE" else 0.0,
            "regret": (0.2 if target == "ACQUIRE" else 0.0) - policy_utility,
        }
        for name in ("cheapest", "clear_per_cost", "random"):
            row[f"{name}_selection"] = "ACQUIRE"
            row[f"{name}_evidence_id"] = "candidate"
            row[f"{name}_utility"] = 0.2 if target == "ACQUIRE" else -0.1
            row[f"{name}_cost"] = 1.0
        row.update(
            {
                "always_stop_selection": "STOP",
                "always_stop_evidence_id": None,
                "always_stop_utility": 0.0,
                "always_stop_cost": 0.0,
            }
        )
        rows.append(row)
    return rows


def test_aggregate_uses_shared_aoi_protocol(tmp_path):
    paths = {}
    for seed in (1, 2):
        path = tmp_path / f"seed-{seed}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in _rows(seed)))
        paths[seed] = path
    result = aggregate(paths, repetitions=20, bootstrap_seed=7)
    assert result["seed_count"] == 2
    assert result["aoi_count"] == 2
    assert result["shared_resample_indices_across_model_seeds"] is True
    assert "always_stop" in result["paired_baseline_delta_intervals"]
