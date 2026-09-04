from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.aggregate_agent_three_seeds import aggregate


def _row(task: int, budget: float, value: float) -> dict[str, object]:
    return {
        "task_id": f"task-{task}",
        "budget": budget,
        "target": "COMMIT:ADD",
        "terminal_correct": True,
        "false_edit": False,
        "missed_edit": False,
        "spent_cost": 0.2,
        "acquisitions": 1,
        "joint_utility": value,
        "final_evidence_quality": value,
        "evidence_quality_gain": value,
        "quality_cost_utility": value,
        "episode_utility_v2_balanced": value,
        "episode_utility_v2_safety": value - 0.1,
        "episode_utility_v2_cost_aware": value - 0.2,
        "tool_calls": 1,
        "tool_cost": 0.1,
        "mean_tool_belief_l1_delta": 0.1,
        "tool_action_flips": 1,
        "terminal_edit_changed_after_tools": True,
    }


def test_hierarchical_agent_aggregate(tmp_path: Path) -> None:
    seeds = [21, 22, 23]
    (tmp_path / "seed_pairing.json").write_text(
        json.dumps(
            {
                "test_assets_read": False,
                "pairs": [
                    {"agent_seed": seed, "selector_seed": seed - 10}
                    for seed in seeds
                ],
            }
        ),
        encoding="utf-8",
    )
    for seed in seeds:
        folder = tmp_path / f"seed{seed}"
        folder.mkdir()
        for method, value in (("full", 0.8), ("selector", 0.5)):
            rows = [_row(task, budget, value) for task in range(4) for budget in (1.5, 3.0)]
            (folder / f"{method}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
    result = aggregate(
        tmp_path,
        seeds=seeds,
        candidate="full",
        baselines=["selector"],
        bootstrap=100,
        rng_seed=7,
    )
    delta = result["comparisons"]["selector"]["paired_delta"]["quality_cost_utility_auc"]
    assert delta["delta"] == pytest.approx(0.3)
    assert delta["ci95_low"] == pytest.approx(0.3)
    assert result["protocol"]["test_assets_read"] is False
    assert result["protocol"]["primary_metric"] == "episode_utility_v2_balanced_auc"
