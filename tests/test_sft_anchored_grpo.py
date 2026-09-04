import json
import sys
from argparse import Namespace

import pytest

from scripts.train_recurrent_proxy_grpo import (
    executable_reward_config,
    main,
    prepare_executable_row,
    reward_for_row,
    sft_action_counts,
    sft_row_split,
    terminal_action_counts,
    validate_joint_training_args,
)
from activemap.agent.recurrent_grpo import ExecutableRewardConfig


def args(**overrides):
    values = {
        "sft_replay_weight": 0.1,
        "sft_replay_probability": 0.25,
        "sft_tool_replay_fraction": 0.5,
        "sft_replay": "train.jsonl",
        "sequence_kl_coef": 0.01,
        "objective": "sequence-clip",
    }
    values.update(overrides)
    return Namespace(**values)


def row(action: str, split: str = "train") -> dict:
    return {
        "messages": [
            {"role": "system", "content": "choose"},
            {"role": "user", "content": json.dumps({"split": split})},
            {"role": "assistant", "content": json.dumps({"action": action})},
        ]
    }


def test_joint_training_argument_contract() -> None:
    validate_joint_training_args(args())
    with pytest.raises(ValueError, match="requires --sft-replay"):
        validate_joint_training_args(args(sft_replay=None))
    with pytest.raises(ValueError, match="sequence-clip"):
        validate_joint_training_args(args(objective="reinforce"))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_joint_training_args(args(sft_replay_probability=1.1))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_joint_training_args(args(sft_tool_replay_fraction=-0.1))
    with pytest.raises(ValueError, match="requires --objective candidate-clip"):
        validate_joint_training_args(
            args(verify_candidate_parity=True, objective="sequence-clip")
        )
    validate_joint_training_args(
        args(verify_candidate_parity=True, objective="candidate-clip")
    )


def test_sft_replay_audits_actions_and_split() -> None:
    rows = [row("COMMIT"), row("USE_TOOL"), row("USE_TOOL")]
    assert sft_action_counts(rows) == {"COMMIT": 1, "USE_TOOL": 2}
    assert sft_row_split(rows[0]) == "train"
    assert sft_row_split(row("REJECT", split="test")) == "test"


def test_terminal_action_counts_do_not_treat_oracle_targets_as_policy_actions() -> None:
    trajectories = [
        {"target": "REJECT", "prediction": "COMMIT:ADD"},
        {"target": "REJECT", "prediction": "COMMIT:DELETE"},
    ]
    assert terminal_action_counts(trajectories, "target") == {"REJECT": 2}
    assert terminal_action_counts(trajectories, "prediction") == {
        "COMMIT:ADD": 1,
        "COMMIT:DELETE": 1,
    }


def test_executable_row_joins_writeback_and_failure_counts() -> None:
    trajectory = {
        "task_id": "task-a",
        "budget": 2.0,
        "evaluation_seed": 7,
        "split": "train",
        "tool_calls": 2,
        "tool_successes": 1,
        "acquisitions": 1,
        "target": "REJECT",
    }
    writeback = {
        "task_id": "task-a",
        "budget": 2.0,
        "split": "train",
        "map_quality_before": 0.4,
        "map_quality_after": 0.8,
        "topology_quality_before": 1.0,
        "topology_quality_after": 1.0,
        "false_edit": False,
        "missed_edit": False,
        "wrong_edit": False,
        "spent_cost": 0.5,
    }
    merged = prepare_executable_row(
        trajectory,
        writeback,
        [{"executable": True}, {"executable": False}],
        rollout_id=3,
    )
    assert merged["group_id"] == "task-a|budget=2"
    assert merged["rollout_id"] == "rollout-3"
    assert merged["invalid_action_count"] == 1
    assert merged["failed_tool_count"] == 1
    reward, components = reward_for_row(
        merged,
        reward_mode="executable",
        false_edit_lagrange=0.0,
        config=ExecutableRewardConfig(),
    )
    assert components["map_quality_delta"] == pytest.approx(0.4)
    assert reward == pytest.approx(0.4 - 0.025 - 1.0 - 0.1)


def test_executable_row_requires_explicit_writeback_quality() -> None:
    trajectory = {
        "task_id": "task-a",
        "budget": 2.0,
        "evaluation_seed": 7,
        "split": "train",
    }
    with pytest.raises(ValueError, match="lacks reward fields"):
        prepare_executable_row(trajectory, {"task_id": "task-a", "budget": 2.0}, [], rollout_id=0)


def test_executable_reward_config_uses_cli_defaults() -> None:
    configured = executable_reward_config(args())
    assert configured == ExecutableRewardConfig()


def test_executable_audit_only_keeps_four_rollouts_per_group(tmp_path, monkeypatch) -> None:
    rollout_args = []
    writeback_args = []
    for index in range(4):
        trajectory = tmp_path / f"trajectory-{index}.jsonl"
        calls = tmp_path / f"calls-{index}.jsonl"
        writeback = tmp_path / f"writeback-{index}.jsonl"
        trajectory.write_text(
            json.dumps(
                {
                    "task_id": "task-a",
                    "budget": 2.0,
                    "evaluation_seed": 100 + index,
                    "split": "train",
                    "target": "REJECT",
                    "false_edit": False,
                    "missed_edit": False,
                    "wrong_edit": False,
                    "tool_calls": int(index == 3),
                    "tool_successes": int(index == 3),
                    "acquisitions": int(index == 3),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        calls.write_text(
            json.dumps(
                {
                    "task_id": "task-a",
                    "budget": 2.0,
                    "evaluation_seed": 100 + index,
                    "executable": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        writeback.write_text(
            json.dumps(
                {
                    "task_id": "task-a",
                    "budget": 2.0,
                    "split": "train",
                    "map_quality_before": 0.4,
                    "map_quality_after": 0.5 + 0.1 * index,
                    "topology_quality_before": 1.0,
                    "topology_quality_after": 1.0,
                    "false_edit": False,
                    "missed_edit": False,
                    "wrong_edit": False,
                    "spent_cost": 0.5,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        rollout_args.extend(["--rollout", str(trajectory), str(calls)])
        writeback_args.extend(["--writeback", str(writeback)])
    output = tmp_path / "audit"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_recurrent_proxy_grpo.py",
            "adapter",
            str(output),
            "--reward-mode",
            "executable",
            "--audit-only",
            *rollout_args,
            *writeback_args,
        ],
    )
    main()
    audit = json.loads((output / "input_audit.json").read_text(encoding="utf-8"))
    assert audit["reward_mode"] == "executable"
    assert audit["groups"] == 1
    assert audit["variable_reward_groups"] == 1
    assert (output / "trajectory_groups.jsonl").exists()
    assert (output / "reward_components.jsonl").exists()
