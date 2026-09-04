from __future__ import annotations

import json
from pathlib import Path

import pytest

from activemap.agent.rl import (
    executable_schema_reward,
    sparse_acquisition_reward,
    target_terminal_action,
    task_utility_reward,
    terminal_safety_reward,
)
from activemap.agent.recurrent_grpo import (
    ExecutableRewardConfig,
    audit_trajectory_groups,
    constrained_proxy_rewards,
    executable_trajectory_reward,
    group_relative_advantages,
    informative_group_ids,
    sequence_clipped_surrogate,
    update_false_edit_lagrange,
)
from scripts.assess_agent_rl_readiness import assess_rl_readiness
from scripts.build_agent_rl_states import build_rl_states
from scripts.build_tool_agent_rl_states import build_tool_rl_states
from scripts.train_agent_grpo import audit_rl_splits


def _observation() -> dict[str, object]:
    return {
        "task_id": "task-a",
        "split": "train",
        "step": 0,
        "initial_budget": 2.0,
        "remaining_budget": 2.0,
        "spent_cost": 0.0,
        "selected_evidence_ids": [],
        "belief": {
            "edit_probabilities": [0.7, 0.2, 0.05, 0.05],
            "confidence": 0.7,
            "geometry_delta": [0.0] * 8,
            "uncertainty": 0.3,
            "recommended_edit": "KEEP",
        },
        "candidates": [
            {
                "evidence_id": "evidence-a",
                "cost": 0.5,
                "selector_score": 0.2,
                "features": [0.1],
            }
        ],
        "terminal_score": 0.0,
        "available_tools": [],
        "tool_history": [],
        "terminal_actions": ["COMMIT", "REJECT"],
    }


def test_constrained_rewards_distinguish_safe_and_invalid_actions() -> None:
    observation_json = json.dumps(_observation())
    completions = [
        '{"action":"REJECT"}',
        '{"action":"COMMIT","edit":"ADD"}',
        '{"action":"ACQUIRE","evidence_id":"missing"}',
        "not json",
    ]
    utilities = {
        "REJECT": 0.0,
        "COMMIT:ADD": -1.0,
        "COMMIT:DELETE": -1.0,
        "COMMIT:RESHAPE": -1.0,
        "ACQUIRE:evidence-a": -0.1,
    }
    columns = [utilities] * len(completions)
    observations = [observation_json] * len(completions)
    targets = ["REJECT"] * len(completions)
    stops = [0.0] * len(completions)

    assert task_utility_reward(completions, columns, observations) == [0.0, -1.0, -1.0, -1.0]
    assert executable_schema_reward(completions, observations) == [0.0, 0.0, -1.0, -1.0]
    assert terminal_safety_reward(completions, targets) == [0.0, -1.0, 0.0, 0.0]
    assert sparse_acquisition_reward(completions, columns, stops) == [0.0, 0.0, -1.0, 0.0]


def test_target_terminal_action_ignores_acquisition_utility() -> None:
    target, utility = target_terminal_action(
        {"ACQUIRE:evidence-a": 0.8, "REJECT": -0.75, "COMMIT:ADD": 0.0}
    )
    assert target == "COMMIT:ADD"
    assert utility == 0.0


def test_build_rl_states_is_test_free(tmp_path: Path) -> None:
    observation = _observation()
    trajectory = {
        "trajectory_id": "trajectory-a",
        "task_id": "task-a",
        "split": "train",
        "budget": 2.0,
        "transitions": [
            {
                "observation": observation,
                "action": {"action": "REJECT"},
                "reward": 1.0,
                "done": True,
                "next_observation": None,
                "oracle_action": {"action": "REJECT"},
                "oracle_utilities": {
                    "ACQUIRE:evidence-a": -0.1,
                    "REJECT": 0.0,
                    "COMMIT:ADD": -1.0,
                    "COMMIT:DELETE": -1.0,
                    "COMMIT:RESHAPE": -1.0,
                },
            }
        ],
        "total_reward": 1.0,
        "metadata": {},
    }
    source = tmp_path / "trajectories.jsonl"
    source.write_text(json.dumps(trajectory) + "\n", encoding="utf-8")
    output = tmp_path / "rl.jsonl"

    summary = build_rl_states(source, output, expected_split="train")
    row = json.loads(output.read_text(encoding="utf-8"))
    assert summary["test_assets_read"] is False
    assert summary["online_closed_loop"] is False
    assert row["target_action_key"] == "REJECT"
    assert row["oracle_best_action_key"] == "REJECT"
    assert row["prompt"][1]["content"] == row["observation_json"]


def test_rl_readiness_is_safety_gate_not_gain_gate() -> None:
    static = {
        "protocol": {"test_assets_read": False},
        "selection_passed": True,
        "selected_checkpoint": "checkpoint-400",
    }
    rollout = {
        "llm_validity_by_method": {
            "qwen3_4b_sft_tool_to_belief": {
                "schema_valid_rate": 1.0,
                "executable_valid_rate": 1.0,
                "fallback_rate": 0.0,
            }
        },
        "results": [
            {
                "method": "qwen3_4b_sft_tool_to_belief",
                "sample_count": 10,
                "false_edit_rate": 0.02,
                "mean_tool_calls": 0.1,
                "tool_success_rate": 1.0,
                "mean_tool_belief_l1_delta": 0.05,
                "tool_action_flip_rate": 0.01,
                "terminal_edit_flip_rate": 0.01,
            }
        ],
    }
    reachability = {
        "schema_version": "activemap-agentic-reachability-audit-v1",
        "passed": True,
        "reachable_tool_state_rate": 1.0,
        "fully_bridged_tool_task_rate": 1.0,
    }
    decision = assess_rl_readiness(static, rollout, reachability)
    assert decision["ready_for_single_seed_rl"] is True
    assert decision["protocol"]["does_not_establish"] == "Agent gain or paper promotion"


def test_rl_readiness_fails_closed_without_reachable_tool_trajectories() -> None:
    static = {
        "selection_passed": True,
        "selected_checkpoint": "checkpoint-400",
        "protocol": {"test_assets_read": False},
    }
    rollout = {
        "llm_validity_by_method": {
            "qwen3_4b_sft_tool_to_belief": {
                "schema_valid_rate": 1.0,
                "executable_valid_rate": 1.0,
                "fallback_rate": 0.0,
            }
        },
        "results": [
            {
                "method": "qwen3_4b_sft_tool_to_belief",
                "sample_count": 10,
                "false_edit_rate": 0.02,
                "mean_tool_calls": 0.1,
                "tool_success_rate": 1.0,
                "mean_tool_belief_l1_delta": 0.05,
                "tool_action_flip_rate": 0.01,
                "terminal_edit_flip_rate": 0.01,
            }
        ],
    }

    decision = assess_rl_readiness(static, rollout)

    assert decision["ready_for_single_seed_rl"] is False
    assert "trajectory_reachability" in decision["failed_gates"]


def test_build_grounded_tool_rl_state_preserves_tool_utility(tmp_path: Path) -> None:
    observation = _observation()
    observation["selected_evidence_ids"] = ["evidence-a"]
    observation["candidates"] = []
    observation["available_tools"] = ["IMAGE_QUALITY"]
    observation_json = json.dumps(observation, separators=(",", ":"))
    system = "Choose a grounded tool action."
    tool = {
        "action": "USE_TOOL",
        "tool_call": {
            "call_id": "call-a",
            "tool": "IMAGE_QUALITY",
            "inputs": {"evidence_id": "evidence-a"},
            "parameters": {},
        },
    }
    terminal = {"action": "REJECT"}
    sft = tmp_path / "sft.jsonl"
    sft.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": observation_json},
                    {"role": "assistant", "content": json.dumps(tool)},
                ],
                "trajectory_id": "tool-trajectory",
                "step": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    preferences = tmp_path / "preferences.jsonl"
    preferences.write_text(
        json.dumps(
            {
                "prompt": f"{system}\n{observation_json}\nAction:",
                "chosen": json.dumps(tool),
                "rejected": json.dumps(terminal),
                "chosen_utility": 1.2,
                "rejected_utility": 1.0,
                "trajectory_id": "tool-trajectory",
                "split": "train",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "tool-rl.jsonl"
    summary = build_tool_rl_states(sft, preferences, output, expected_split="train")
    row = json.loads(output.read_text(encoding="utf-8"))
    key = "USE_TOOL:IMAGE_QUALITY:evidence-a"
    assert summary["target_action_counts"] == {"USE_TOOL": 1}
    assert row["oracle_best_action_key"] == key
    assert row["oracle_utilities"][key] == 1.2
    completion = json.dumps(tool, separators=(",", ":"))
    assert task_utility_reward(
        [completion], [row["oracle_utilities"]], [row["observation_json"]]
    ) == [1.2]
    assert sparse_acquisition_reward(
        [completion], [row["oracle_utilities"]], [row["stop_utility"]]
    ) == [0.0]


def test_grpo_audit_rejects_cross_split_task_leakage(tmp_path: Path) -> None:
    observation = _observation()
    utilities = {
        "REJECT": 0.0,
        "COMMIT:ADD": -1.0,
        "COMMIT:DELETE": -1.0,
        "COMMIT:RESHAPE": -1.0,
    }

    def row(split: str, trajectory: str) -> dict[str, object]:
        current = {**observation, "split": split}
        serialized = json.dumps(current, separators=(",", ":"))
        return {
            "prompt": [{"role": "user", "content": serialized}],
            "observation_json": serialized,
            "oracle_utilities": utilities,
            "target_action_key": "REJECT",
            "stop_utility": 0.0,
            "oracle_best_action_key": "REJECT",
            "oracle_best_advantage": 0.0,
            "trajectory_id": trajectory,
            "step": 0,
            "split": split,
        }

    with pytest.raises(ValueError, match="task leakage"):
        audit_rl_splits([row("train", "train-a")], [row("val", "val-a")])


def test_grpo_audit_rejects_inconsistent_reward_metadata() -> None:
    observation = _observation()
    serialized = json.dumps(observation, separators=(",", ":"))
    row = {
        "prompt": [{"role": "user", "content": serialized}],
        "observation_json": serialized,
        "oracle_utilities": {"REJECT": 0.0, "COMMIT:ADD": -1.0},
        "target_action_key": "REJECT",
        "stop_utility": 0.0,
        "oracle_best_action_key": "COMMIT:ADD",
        "oracle_best_advantage": 0.0,
        "trajectory_id": "train-a",
        "step": 0,
        "split": "train",
    }
    with pytest.raises(ValueError, match="oracle_best_action_key"):
        audit_rl_splits([row], None)


def _executable_rollout(
    group: str, rollout: str, quality: float, *, false_edit: bool = False
) -> dict[str, object]:
    return {
        "group_id": group,
        "rollout_id": rollout,
        "split": "train",
        "map_quality_before": 0.4,
        "map_quality_after": quality,
        "topology_quality_before": 0.5,
        "topology_quality_after": 0.6,
        "false_edit": false_edit,
        "missed_edit": False,
        "wrong_edit": False,
        "spent_cost": 0.5,
        "budget": 2.0,
        "invalid_action_count": 0,
        "failed_tool_count": 0,
        "contains_nonstop_action": quality > 0.5,
    }


def test_executable_trajectory_reward_uses_map_delta_and_safety() -> None:
    safe, components = executable_trajectory_reward(
        _executable_rollout("a", "0", 0.8),
        ExecutableRewardConfig(),
    )
    unsafe, _ = executable_trajectory_reward(
        _executable_rollout("a", "1", 0.8, false_edit=True),
        false_edit_lagrange=0.5,
    )
    assert components["map_quality_delta"] == pytest.approx(0.4)
    assert safe - unsafe == pytest.approx(1.5)


def test_group_relative_advantages_are_centered() -> None:
    advantages = group_relative_advantages([0.0, 1.0, 2.0, 3.0])
    assert float(advantages.mean()) == pytest.approx(0.0, abs=1e-8)
    assert advantages[-1] > 0
    assert advantages[0] < 0
    assert group_relative_advantages([1.0, 1.0]).tolist() == [0.0, 0.0]


def test_recurrent_grpo_readiness_requires_variable_groups() -> None:
    rows = [
        _executable_rollout(group, str(index), 0.5 + 0.1 * index)
        for group in ("a", "b")
        for index in range(4)
    ]
    report = audit_trajectory_groups(rows)
    assert report["ready_for_recurrent_grpo"] is True
    assert report["variable_reward_group_rate"] == 1.0

    collapsed = [
        _executable_rollout(group, str(index), 0.5)
        for group in ("a", "b")
        for index in range(4)
    ]
    report = audit_trajectory_groups(collapsed)
    assert report["ready_for_recurrent_grpo"] is False
    assert "reward_variance" in report["failed_gates"]


def test_recurrent_grpo_readiness_uses_executed_terminal_actions() -> None:
    rows = [
        {
            **_executable_rollout("a", str(index), 0.5 + 0.1 * index),
            "target": "REJECT",
            "prediction": "COMMIT:ADD",
            "tool_calls": 1,
        }
        for index in range(4)
    ]
    report = audit_trajectory_groups(
        rows,
        minimum_keep_trajectories=1,
        minimum_commit_trajectories=1,
        minimum_tool_trajectories=1,
    )
    assert report["executed_keep_trajectories"] == 0
    assert report["target_terminal_action_counts"] == {"REJECT": 4}
    assert report["executed_terminal_action_counts"] == {"COMMIT:ADD": 4}
    assert "executed_keep_support" in report["failed_gates"]


def test_false_edit_dual_update_is_projected() -> None:
    assert update_false_edit_lagrange(0.5, 0.10, 0.05) > 0.5
    assert update_false_edit_lagrange(0.0, 0.0, 0.05) == 0.0


def test_proxy_constraint_penalizes_false_edits() -> None:
    rows = [
        {"episode_utility_v2_proxy_balanced": 0.5, "false_edit": False},
        {"episode_utility_v2_proxy_balanced": 0.5, "false_edit": True},
    ]
    rewards = constrained_proxy_rewards(rows, false_edit_lagrange=0.3)
    assert rewards.tolist() == pytest.approx([0.5, 0.2])


def test_informative_groups_drop_zero_variance_samples() -> None:
    groups = {
        "flat": [
            {"episode_utility_v2_proxy_balanced": 0.5} for _ in range(4)
        ],
        "variable": [
            {"episode_utility_v2_proxy_balanced": value}
            for value in (0.2, 0.2, 0.5, 0.8)
        ],
    }
    assert informative_group_ids(groups) == {"variable"}


def test_sequence_clipping_handles_both_advantage_signs() -> None:
    assert sequence_clipped_surrogate(1.0, 0.0, 1.0) == pytest.approx(1.28)
    assert sequence_clipped_surrogate(-1.0, 0.0, -1.0) == pytest.approx(-0.8)
