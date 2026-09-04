#!/usr/bin/env python3
"""Single-pass group-relative policy update over recurrent rollout traces.

The historical default is a proxy-reward diagnostic.  ``--reward-mode
executable`` switches to rewards computed after vector-map writeback and is
the only mode that can support the paper's recurrent GRPO claim.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from activemap.agent.recurrent_grpo import (
    ExecutableRewardConfig,
    constrained_proxy_rewards,
    executable_trajectory_reward,
    group_relative_advantages,
    update_false_edit_lagrange,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("adapter", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rollout", nargs=2, action="append", required=True,
                        metavar=("TRAJECTORIES", "LLM_CALLS"))
    parser.add_argument(
        "--writeback",
        action="append",
        type=Path,
        help=(
            "Executable writeback JSONL for each --rollout pair; required "
            "when --reward-mode executable"
        ),
    )
    parser.add_argument(
        "--reward-mode", choices=("proxy", "executable"), default="proxy"
    )
    parser.add_argument("--map-quality-weight", type=float, default=1.0)
    parser.add_argument("--topology-quality-weight", type=float, default=0.25)
    parser.add_argument("--false-edit-weight", type=float, default=1.0)
    parser.add_argument("--missed-edit-weight", type=float, default=0.5)
    parser.add_argument("--wrong-edit-weight", type=float, default=0.25)
    parser.add_argument("--normalized-cost-weight", type=float, default=0.10)
    parser.add_argument("--invalid-action-weight", type=float, default=1.0)
    parser.add_argument("--failed-tool-weight", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--entropy-coef", type=float, default=1e-3)
    parser.add_argument("--sft-replay", type=Path)
    parser.add_argument("--sft-replay-weight", type=float, default=0.0)
    parser.add_argument("--sft-replay-probability", type=float, default=0.25)
    parser.add_argument("--sft-tool-replay-fraction", type=float, default=0.0)
    parser.add_argument("--sequence-kl-coef", type=float, default=0.0)
    parser.add_argument(
        "--dynamic-sampling", choices=("all", "variable-only"), default="all"
    )
    parser.add_argument("--minimum-reward-std", type=float, default=1e-6)
    parser.add_argument(
        "--objective",
        choices=("reinforce", "sequence-clip", "candidate-clip"),
        default="reinforce",
    )
    parser.add_argument("--clip-epsilon-low", type=float, default=0.2)
    parser.add_argument("--clip-epsilon-high", type=float, default=0.28)
    parser.add_argument(
        "--candidate-score-batch-size",
        type=int,
        default=1,
        help="candidate completions evaluated together for --objective candidate-clip",
    )
    parser.add_argument("--false-edit-lagrange", type=float, default=0.0)
    parser.add_argument("--false-edit-target", type=float)
    parser.add_argument("--false-edit-dual-lr", type=float, default=0.0)
    parser.add_argument("--minimum-false-edit-trajectories", type=int, default=0)
    parser.add_argument("--minimum-keep-trajectories", type=int, default=0)
    parser.add_argument("--minimum-commit-trajectories", type=int, default=0)
    parser.add_argument("--minimum-tool-trajectories", type=int, default=0)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--seed", type=int, default=20260861)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="write audited grouped rewards without loading the language model",
    )
    parser.add_argument(
        "--verify-candidate-parity",
        action="store_true",
        help=(
            "load the candidate policy, recompute its complete categorical "
            "action distribution, verify behavior/update probability parity, "
            "and exit without an optimizer update"
        ),
    )
    return parser.parse_args()


def validate_joint_training_args(args: argparse.Namespace) -> None:
    reward_mode = getattr(args, "reward_mode", "proxy")
    if reward_mode not in {"proxy", "executable"}:
        raise ValueError("--reward-mode must be proxy or executable")
    for name in (
        "map_quality_weight",
        "topology_quality_weight",
        "false_edit_weight",
        "missed_edit_weight",
        "wrong_edit_weight",
        "normalized_cost_weight",
        "invalid_action_weight",
        "failed_tool_weight",
    ):
        if getattr(args, name, 0.0) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if args.sft_replay_weight < 0:
        raise ValueError("--sft-replay-weight must be non-negative")
    if not 0.0 <= args.sft_replay_probability <= 1.0:
        raise ValueError("--sft-replay-probability must be in [0, 1]")
    if not 0.0 <= args.sft_tool_replay_fraction <= 1.0:
        raise ValueError("--sft-tool-replay-fraction must be in [0, 1]")
    if args.sft_replay_weight > 0 and args.sft_replay is None:
        raise ValueError("positive --sft-replay-weight requires --sft-replay")
    if args.sequence_kl_coef < 0:
        raise ValueError("--sequence-kl-coef must be non-negative")
    if args.sequence_kl_coef > 0 and args.objective not in {
        "sequence-clip", "candidate-clip"
    }:
        raise ValueError(
            "--sequence-kl-coef requires --objective sequence-clip or candidate-clip"
        )
    if getattr(args, "candidate_score_batch_size", 1) <= 0:
        raise ValueError("--candidate-score-batch-size must be positive")
    if (
        getattr(args, "verify_candidate_parity", False)
        and args.objective != "candidate-clip"
    ):
        raise ValueError(
            "--verify-candidate-parity requires --objective candidate-clip"
        )


def sft_action(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError("SFT replay rows require three messages")
    content = json.loads(str(messages[2].get("content", "")))
    return str(content.get("action", "")).strip() or "UNKNOWN"


def sft_action_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[sft_action(row)] += 1
    return dict(sorted(counts.items()))


def terminal_action_counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    """Count actual terminal decisions separately from oracle targets."""

    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        action = str(row.get(field, "UNKNOWN")).strip() or "UNKNOWN"
        counts[action] += 1
    return dict(sorted(counts.items()))


def candidate_training_payload_summary(calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate the exact categorical behaviour policy needed by candidate GRPO.

    A candidate-sampled action is not drawn from the raw completion likelihood:
    it is drawn from a temperature-scaled softmax over all executable action
    completions.  Training must have the full action support and prompt tokens
    from collection, otherwise its PPO ratio would be off-policy.
    """

    candidate_sizes: list[int] = []
    temperatures: list[float] = []
    for index, row in enumerate(calls):
        payload = row.get("training_payload")
        if not isinstance(payload, dict):
            raise ValueError(f"candidate-clip call {index} lacks training_payload")
        prompt_ids = payload.get("prompt_token_ids")
        decoder = payload.get("structured_decoder")
        if not isinstance(prompt_ids, list) or not prompt_ids:
            raise ValueError(
                f"candidate-clip call {index} lacks exact prompt_token_ids; "
                "recollect rollouts with the current candidate decoder"
            )
        if not isinstance(decoder, dict) or decoder.get("decoder") != "candidate-sample":
            raise ValueError(
                f"candidate-clip call {index} was not collected by candidate-sample"
            )
        completions = decoder.get("candidate_completions")
        token_ids = decoder.get("candidate_completion_token_ids")
        probabilities = decoder.get("candidate_probabilities")
        chosen_index = decoder.get("chosen_index")
        temperature = decoder.get("candidate_temperature")
        if not isinstance(completions, list) or len(completions) < 2:
            raise ValueError(f"candidate-clip call {index} lacks candidate completions")
        if not isinstance(token_ids, list) or len(token_ids) != len(completions):
            raise ValueError(f"candidate-clip call {index} has invalid candidate token ids")
        if any(not isinstance(value, list) or not value for value in token_ids):
            raise ValueError(f"candidate-clip call {index} has an empty candidate completion")
        if not isinstance(probabilities, list) or len(probabilities) != len(completions):
            raise ValueError(f"candidate-clip call {index} has invalid candidate probabilities")
        try:
            probability_values = [float(value) for value in probabilities]
            temperature_value = float(temperature)
            chosen_index_value = int(chosen_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"candidate-clip call {index} has malformed decoder metadata") from exc
        if (
            not all(math.isfinite(value) and value > 0.0 for value in probability_values)
            or not math.isclose(sum(probability_values), 1.0, rel_tol=0.0, abs_tol=1e-5)
        ):
            raise ValueError(
                f"candidate-clip call {index} has non-normalized behaviour probabilities"
            )
        if not math.isfinite(temperature_value) or temperature_value <= 0.0:
            raise ValueError(f"candidate-clip call {index} has invalid candidate temperature")
        if not 0 <= chosen_index_value < len(completions):
            raise ValueError(f"candidate-clip call {index} has invalid chosen_index")
        candidate_sizes.append(len(completions))
        temperatures.append(temperature_value)
    if not candidate_sizes:
        raise ValueError("candidate-clip received no policy calls")
    return {
        "calls": len(candidate_sizes),
        "candidate_size_min": min(candidate_sizes),
        "candidate_size_max": max(candidate_sizes),
        "temperatures": sorted(set(temperatures)),
    }


def sft_row_split(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError("SFT replay rows require three messages")
    payload = json.loads(str(messages[1].get("content", "")))
    return str(payload.get("split", ""))


def executable_reward_config(args: argparse.Namespace) -> ExecutableRewardConfig:
    """Build the immutable reward contract from CLI values for auditability."""

    return ExecutableRewardConfig(
        map_quality=float(getattr(args, "map_quality_weight", 1.0)),
        topology_quality=float(getattr(args, "topology_quality_weight", 0.25)),
        false_edit=float(getattr(args, "false_edit_weight", 1.0)),
        missed_edit=float(getattr(args, "missed_edit_weight", 0.5)),
        wrong_edit=float(getattr(args, "wrong_edit_weight", 0.25)),
        normalized_cost=float(getattr(args, "normalized_cost_weight", 0.10)),
        invalid_action=float(getattr(args, "invalid_action_weight", 1.0)),
        failed_tool=float(getattr(args, "failed_tool_weight", 0.10)),
    )


def _writeback_index(path: Path) -> dict[tuple[str, float], dict[str, Any]]:
    rows = read_jsonl(path)
    indexed: dict[tuple[str, float], dict[str, Any]] = {}
    for row in rows:
        if row.get("split") == "test" or row.get("test_assets_read") is True:
            raise ValueError("test writeback rows are forbidden")
        key = (str(row["task_id"]), float(row["budget"]))
        if key in indexed:
            raise ValueError(f"duplicate writeback task-budget row: {key}")
        indexed[key] = row
    return indexed


def prepare_executable_row(
    trajectory: dict[str, Any],
    writeback: dict[str, Any],
    calls: list[dict[str, Any]],
    *,
    rollout_id: int,
) -> dict[str, Any]:
    """Join terminal rollout metadata with the executed map writeback result."""

    required = {
        "map_quality_before",
        "map_quality_after",
        "topology_quality_before",
        "topology_quality_after",
    }
    missing = sorted(required - writeback.keys())
    if missing:
        raise ValueError(
            "executable writeback lacks reward fields: " + ", ".join(missing)
        )
    if trajectory.get("split") == "test" or writeback.get("split") == "test":
        raise ValueError("test trajectories are forbidden")
    if str(trajectory["task_id"]) != str(writeback["task_id"]):
        raise ValueError("trajectory and writeback task ids do not match")
    if float(trajectory["budget"]) != float(writeback["budget"]):
        raise ValueError("trajectory and writeback budgets do not match")
    invalid_actions = int(
        trajectory.get(
            "invalid_action_count",
            sum(not bool(call.get("executable", True)) for call in calls),
        )
    )
    failed_tools = int(
        trajectory.get(
            "failed_tool_count",
            max(0, int(trajectory.get("tool_calls", 0))
                - int(trajectory.get("tool_successes", 0))),
        )
    )
    if invalid_actions < 0 or failed_tools < 0:
        raise ValueError("executable failure counts cannot be negative")
    row = dict(trajectory)
    row.update(
        {
            "group_id": f"{trajectory['task_id']}|budget={float(trajectory['budget']):g}",
            "rollout_id": f"rollout-{rollout_id}",
            "map_quality_before": float(writeback["map_quality_before"]),
            "map_quality_after": float(writeback["map_quality_after"]),
            "topology_quality_before": float(writeback["topology_quality_before"]),
            "topology_quality_after": float(writeback["topology_quality_after"]),
            "false_edit": bool(writeback.get("false_edit", trajectory.get("false_edit", False))),
            "missed_edit": bool(writeback.get("missed_edit", trajectory.get("missed_edit", False))),
            "wrong_edit": bool(writeback.get("wrong_edit", trajectory.get("wrong_edit", False))),
            "spent_cost": float(writeback.get("spent_cost", trajectory.get("spent_cost", 0.0))),
            "budget": float(writeback["budget"]),
            "invalid_action_count": invalid_actions,
            "failed_tool_count": failed_tools,
            "contains_nonstop_action": bool(
                trajectory.get(
                    "contains_nonstop_action",
                    int(trajectory.get("acquisitions", 0)) > 0
                    or int(trajectory.get("tool_calls", 0)) > 0,
                )
            ),
        }
    )
    return row


def reward_for_row(
    row: dict[str, Any],
    *,
    reward_mode: str,
    false_edit_lagrange: float,
    config: ExecutableRewardConfig,
) -> tuple[float, dict[str, float]]:
    if reward_mode == "proxy":
        values = constrained_proxy_rewards(
            [row], false_edit_lagrange=false_edit_lagrange
        )
        return float(values[0]), {
            "proxy_reward": float(values[0]),
            "false_edit": float(bool(row.get("false_edit", False))),
        }
    if reward_mode == "executable":
        return executable_trajectory_reward(
            row, config=config, false_edit_lagrange=false_edit_lagrange
        )
    raise ValueError(f"unsupported reward mode: {reward_mode}")


def informative_reward_groups(
    groups: dict[tuple[str, float], list[dict[str, Any]]],
    *,
    reward_mode: str,
    false_edit_lagrange: float,
    config: ExecutableRewardConfig,
    minimum_reward_std: float,
) -> set[tuple[str, float]]:
    selected = set()
    for group_id, rows in groups.items():
        if len(rows) < 4:
            continue
        rewards = np.asarray(
            [
                reward_for_row(
                    row,
                    reward_mode=reward_mode,
                    false_edit_lagrange=false_edit_lagrange,
                    config=config,
                )[0]
                for row in rows
            ],
            dtype=np.float64,
        )
        if float(rewards.std()) > minimum_reward_std:
            selected.add(group_id)
    return selected


def main() -> None:
    args = parse_args()
    validate_joint_training_args(args)
    if args.output.exists():
        raise FileExistsError(f"refusing existing output: {args.output}")
    writeback_paths = args.writeback or []
    if writeback_paths and len(writeback_paths) != len(args.rollout):
        raise ValueError("each --rollout pair must have exactly one --writeback path")
    if args.reward_mode == "executable" and not writeback_paths:
        raise ValueError("--reward-mode executable requires --writeback for every rollout")
    args.output.mkdir(parents=True)
    reward_config = executable_reward_config(args)
    terminal_by_key: dict[tuple[int, str, float, int], dict[str, Any]] = {}
    calls: list[dict[str, Any]] = []
    for rollout_id, pair in enumerate(args.rollout):
        trajectories_path, calls_path = map(Path, pair)
        trajectories = read_jsonl(trajectories_path)
        call_rows = read_jsonl(calls_path)
        calls_by_key: dict[tuple[str, float, int], list[dict[str, Any]]] = defaultdict(list)
        for row in call_rows:
            key = (str(row["task_id"]), float(row["budget"]), int(row["evaluation_seed"]))
            calls_by_key[key].append(row)
        writeback_by_key = (
            _writeback_index(writeback_paths[rollout_id])
            if args.reward_mode == "executable"
            else {}
        )
        trajectory_keys: set[tuple[str, float, int]] = set()
        for row in trajectories:
            if row.get("split") == "test":
                raise ValueError("test trajectories are forbidden")
            base_key = (
                str(row["task_id"]),
                float(row["budget"]),
                int(row["evaluation_seed"]),
            )
            if base_key in trajectory_keys:
                raise ValueError(
                    f"duplicate trajectory identity within rollout {rollout_id}: {base_key}"
                )
            trajectory_keys.add(base_key)
            terminal_key = (rollout_id, *base_key)
            if args.reward_mode == "executable":
                writeback_key = (base_key[0], base_key[1])
                if writeback_key not in writeback_by_key:
                    raise ValueError(
                        f"writeback has no terminal row for rollout {rollout_id}: {writeback_key}"
                    )
                enriched = prepare_executable_row(
                    row,
                    writeback_by_key[writeback_key],
                    calls_by_key.get(base_key, []),
                    rollout_id=rollout_id,
                )
            else:
                enriched = dict(row)
                enriched["group_id"] = f"{base_key[0]}|budget={base_key[1]:g}"
                enriched["rollout_id"] = f"rollout-{rollout_id}"
            enriched["_terminal_key"] = terminal_key
            terminal_by_key[terminal_key] = enriched
        if not trajectory_keys:
            raise ValueError(f"rollout {rollout_id} contains no trajectories")
        for row in call_rows:
            key = (str(row["task_id"]), float(row["budget"]), int(row["evaluation_seed"]))
            if key not in trajectory_keys:
                raise ValueError(f"LLM call has no terminal trajectory: {key}")
            calls.append(
                {
                    **row,
                    "rollout_id": rollout_id,
                    "terminal_key": (rollout_id, *key),
                }
            )

    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in terminal_by_key.values():
        groups[(str(row["task_id"]), float(row["budget"]))].append(row)
    if not groups:
        raise ValueError("no recurrent GRPO groups were loaded")
    if args.max_groups is not None:
        if args.max_groups <= 0:
            raise ValueError("--max-groups must be positive")
        kept = set(sorted(groups, key=str)[: args.max_groups])
        groups = {key: value for key, value in groups.items() if key in kept}

    all_trajectories = [row for group in groups.values() for row in group]
    observed_false_edit_rate = float(
        np.mean([bool(row.get("false_edit", False)) for row in all_trajectories])
    )
    false_edit_trajectories = sum(
        bool(row.get("false_edit", False)) for row in all_trajectories
    )
    executed_terminal_actions = terminal_action_counts(all_trajectories, "prediction")
    target_terminal_actions = terminal_action_counts(all_trajectories, "target")
    keep_trajectories = executed_terminal_actions.get("REJECT", 0)
    commit_trajectories = sum(
        count
        for action, count in executed_terminal_actions.items()
        if action.startswith("COMMIT")
    )
    tool_trajectories = sum(
        int(row.get("tool_calls", 0)) > 0 for row in all_trajectories
    )
    if false_edit_trajectories < args.minimum_false_edit_trajectories:
        raise ValueError(
            "insufficient false-edit support: "
            f"{false_edit_trajectories} < {args.minimum_false_edit_trajectories}"
        )
    if keep_trajectories < args.minimum_keep_trajectories:
        raise ValueError(
            "insufficient executed KEEP support: "
            f"{keep_trajectories} < {args.minimum_keep_trajectories}"
        )
    if commit_trajectories < args.minimum_commit_trajectories:
        raise ValueError(
            "insufficient executed COMMIT support: "
            f"{commit_trajectories} < {args.minimum_commit_trajectories}"
        )
    if tool_trajectories < args.minimum_tool_trajectories:
        raise ValueError(
            "insufficient executed tool support: "
            f"{tool_trajectories} < {args.minimum_tool_trajectories}"
        )
    false_edit_lagrange = args.false_edit_lagrange
    if args.false_edit_target is not None:
        if not 0.0 <= args.false_edit_target <= 1.0:
            raise ValueError("--false-edit-target must be in [0, 1]")
        if args.false_edit_dual_lr > 0:
            false_edit_lagrange = update_false_edit_lagrange(
                false_edit_lagrange,
                observed_false_edit_rate,
                args.false_edit_target,
                learning_rate=args.false_edit_dual_lr,
            )
    informative = informative_reward_groups(
        groups,
        reward_mode=args.reward_mode,
        false_edit_lagrange=false_edit_lagrange,
        config=reward_config,
        minimum_reward_std=args.minimum_reward_std,
    )
    selected_groups = set(groups)
    if args.dynamic_sampling == "variable-only":
        selected_groups = informative
    if not selected_groups:
        raise ValueError("dynamic sampling retained no GRPO groups")

    advantages: dict[tuple[int, str, float, int], float] = {}
    reward_records: list[dict[str, Any]] = []
    variable_groups = 0
    for group_id, group in groups.items():
        if group_id not in selected_groups:
            continue
        if len(group) < 4:
            raise ValueError("every GRPO group must contain at least four rollouts")
        reward_pairs = [
            reward_for_row(
                row,
                reward_mode=args.reward_mode,
                false_edit_lagrange=false_edit_lagrange,
                config=reward_config,
            )
            for row in group
        ]
        rewards = np.asarray([pair[0] for pair in reward_pairs], dtype=np.float64)
        if rewards.std() > args.minimum_reward_std:
            variable_groups += 1
        normalized = group_relative_advantages(rewards)
        for row, advantage in zip(group, normalized, strict=True):
            key = tuple(row["_terminal_key"])
            advantages[key] = float(advantage)
        for row, (reward, components) in zip(group, reward_pairs, strict=True):
            reward_records.append(
                {
                    "group_id": f"{group_id[0]}|budget={group_id[1]:g}",
                    "rollout_id": row["rollout_id"],
                    "reward": reward,
                    "components": components,
                }
            )
    if variable_groups == 0:
        raise ValueError("all GRPO groups have zero reward variance")
    calls = [
        row
        for row in calls
        if (str(row["terminal_key"][1]), float(row["terminal_key"][2]))
        in selected_groups
    ]
    if not calls:
        raise ValueError("selected GRPO groups contain no policy calls")
    candidate_behavior_summary = (
        candidate_training_payload_summary(calls)
        if args.objective == "candidate-clip"
        else None
    )
    selected_trajectories = [
        row
        for group_id, group in groups.items()
        if group_id in selected_groups
        for row in group
    ]
    tool_positive = sum(int(row.get("tool_calls", 0)) > 0 for row in selected_trajectories)

    sft_rows = read_jsonl(args.sft_replay) if args.sft_replay is not None else []
    if any(sft_row_split(row) == "test" for row in sft_rows):
        raise ValueError("test SFT replay rows are forbidden")
    replay_action_counts = sft_action_counts(sft_rows) if sft_rows else {}
    if args.sft_replay_weight > 0 and replay_action_counts.get("USE_TOOL", 0) == 0:
        raise ValueError("SFT replay must contain USE_TOOL actions")

    reward_values = [record["reward"] for record in reward_records]
    component_names = sorted(
        {
            name
            for record in reward_records
            for name in record["components"]
        }
    )
    component_means = {
        name: float(
            np.mean([record["components"].get(name, 0.0) for record in reward_records])
        )
        for name in component_names
    }
    audit = {
        "schema_version": "activemap-sft-anchored-recurrent-grpo-v4",
        "claim_boundary": (
            "policy update over executable vector-map writeback trajectories"
            if args.reward_mode == "executable"
            else "policy update with terminal-operation proxy reward; not executable-map GRPO"
        ),
        "reward_mode": args.reward_mode,
        "reward_config": asdict(reward_config) if args.reward_mode == "executable" else None,
        "groups": len(groups),
        "selected_groups": len(selected_groups),
        "dynamic_sampling": args.dynamic_sampling,
        "variable_reward_groups": variable_groups,
        "variable_reward_group_rate": variable_groups / len(groups),
        "trajectories": len(selected_trajectories),
        "llm_calls": len(calls),
        "mean_reward": float(np.mean(reward_values)),
        "reward_std": float(np.std(reward_values)),
        "reward_component_means": component_means,
        "objective": args.objective,
        "candidate_behavior_policy": candidate_behavior_summary,
        "candidate_score_batch_size": (
            args.candidate_score_batch_size
            if args.objective == "candidate-clip"
            else None
        ),
        "clip_epsilon_low": args.clip_epsilon_low,
        "clip_epsilon_high": args.clip_epsilon_high,
        "observed_false_edit_rate": observed_false_edit_rate,
        "false_edit_trajectories": false_edit_trajectories,
        "keep_trajectories": keep_trajectories,
        "commit_trajectories": commit_trajectories,
        "tool_trajectories": tool_trajectories,
        "executed_terminal_action_counts": executed_terminal_actions,
        "target_terminal_action_counts": target_terminal_actions,
        "minimum_keep_trajectories": args.minimum_keep_trajectories,
        "minimum_commit_trajectories": args.minimum_commit_trajectories,
        "minimum_tool_trajectories": args.minimum_tool_trajectories,
        "false_edit_target": args.false_edit_target,
        "false_edit_lagrange": false_edit_lagrange,
        "tool_positive_trajectories": tool_positive,
        "tool_positive_trajectory_rate": tool_positive / len(selected_trajectories),
        "sft_replay_path": str(args.sft_replay) if args.sft_replay else None,
        "sft_replay_rows": len(sft_rows),
        "sft_replay_action_counts": replay_action_counts,
        "sft_replay_weight": args.sft_replay_weight,
        "sft_replay_probability": args.sft_replay_probability,
        "sft_tool_replay_fraction": args.sft_tool_replay_fraction,
        "sequence_kl_coef": args.sequence_kl_coef,
        "test_assets_read": False,
    }
    (args.output / "input_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output / "reward_components.jsonl").open("w", encoding="utf-8") as handle:
        for record in reward_records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    with (args.output / "trajectory_groups.jsonl").open("w", encoding="utf-8") as handle:
        for row in selected_trajectories:
            serializable = {
                key: value for key, value in row.items() if not key.startswith("_")
            }
            handle.write(json.dumps(serializable, separators=(",", ":")) + "\n")
    if args.audit_only and not args.verify_candidate_parity:
        return

    import torch
    from peft import AutoPeftModelForCausalLM, PeftConfig
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, set_seed

    from activemap.agent.sft_data import ActionSFTDataset

    set_seed(args.seed)
    random.Random(args.seed).shuffle(calls)
    peft_config = PeftConfig.from_pretrained(args.adapter)
    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.adapter, is_trainable=True, torch_dtype=torch.bfloat16
    ).cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    if args.objective == "candidate-clip":
        # Gradient checkpointing in Transformers is active only in training mode.
        # Candidate collection is deterministic apart from categorical sampling,
        # so keep every dropout module in evaluation mode while allowing the
        # decoder layers to checkpoint activations during the policy update.
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.eval()
    sft_dataset = (
        ActionSFTDataset(sft_rows, tokenizer, max_length=args.max_length)
        if args.sft_replay_weight > 0
        else None
    )
    replay_rng = random.Random(args.seed + 1)
    tool_replay_indices = [
        index for index, row in enumerate(sft_rows) if sft_action(row) == "USE_TOOL"
    ]
    other_replay_indices = [
        index for index, row in enumerate(sft_rows) if sft_action(row) != "USE_TOOL"
    ]
    if args.sft_tool_replay_fraction > 0 and not tool_replay_indices:
        raise ValueError("positive tool replay fraction requires USE_TOOL rows")
    if args.sft_tool_replay_fraction < 1 and sft_dataset is not None and not other_replay_indices:
        raise ValueError("tool replay fraction below one requires non-tool rows")

    causal_model = model.get_base_model()

    def completion_statistics(row: dict[str, Any]):
        payload = row["training_payload"]
        prompt_ids = tokenizer(payload["prompt"], add_special_tokens=False)["input_ids"]
        completion_ids = list(map(int, payload["completion_token_ids"]))
        prompt_ids = prompt_ids[-max(1, args.max_length - len(completion_ids)):]
        input_ids = torch.tensor([prompt_ids + completion_ids], device="cuda")
        completion_start = len(prompt_ids)
        decoder_outputs = causal_model.model(input_ids=input_ids, use_cache=False)
        completion_hidden = decoder_outputs.last_hidden_state[:, completion_start - 1 : -1]
        completion_logits = causal_model.lm_head(completion_hidden).float()
        log_probs = completion_logits.log_softmax(-1)
        targets = input_ids[:, completion_start:]
        token_logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        token_probs = log_probs.exp()
        entropy = -(token_probs * log_probs).sum(-1).mean()
        return token_logp.mean(), entropy

    def candidate_statistics(row: dict[str, Any]):
        """Return the exact categorical policy statistics for one action call."""

        payload = row["training_payload"]
        decoder = payload["structured_decoder"]
        prompt_ids = list(map(int, payload["prompt_token_ids"]))
        all_completion_ids = [
            list(map(int, token_ids))
            for token_ids in decoder["candidate_completion_token_ids"]
        ]
        prompt_length = len(prompt_ids)
        sequence_scores: list[Any] = []
        for start in range(0, len(all_completion_ids), args.candidate_score_batch_size):
            batch = all_completion_ids[start : start + args.candidate_score_batch_size]
            maximum_length = prompt_length + max(len(token_ids) for token_ids in batch)
            input_ids = torch.full(
                (len(batch), maximum_length),
                int(tokenizer.pad_token_id),
                dtype=torch.long,
                device="cuda",
            )
            attention_mask = torch.zeros_like(input_ids)
            for batch_index, completion_ids in enumerate(batch):
                length = prompt_length + len(completion_ids)
                input_ids[batch_index, :prompt_length] = torch.tensor(
                    prompt_ids, dtype=torch.long, device="cuda"
                )
                input_ids[batch_index, prompt_length:length] = torch.tensor(
                    completion_ids, dtype=torch.long, device="cuda"
                )
                attention_mask[batch_index, :length] = 1
            decoder_outputs = causal_model.model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            )
            for batch_index, completion_ids in enumerate(batch):
                completion_length = len(completion_ids)
                positions = slice(
                    prompt_length - 1, prompt_length - 1 + completion_length
                )
                # Only completion positions contribute to the categorical
                # action score.  Projecting every prompt token to the full
                # vocabulary creates a multi-GB differentiable tensor at a
                # 2k context and can OOM a 24GB card without changing the loss.
                completion_hidden = decoder_outputs.last_hidden_state[
                    batch_index, positions
                ]
                completion_logits = causal_model.lm_head(completion_hidden).float()
                log_probs = completion_logits.log_softmax(-1)
                targets = input_ids[
                    batch_index, prompt_length : prompt_length + completion_length
                ]
                token_logp = log_probs.gather(
                    -1, targets.unsqueeze(-1)
                ).squeeze(-1)
                # This is deliberately the same length-normalized action score
                # used at collection time by StructuredLLMPolicy.
                sequence_scores.append(token_logp.mean())
        scores = torch.stack(sequence_scores)
        temperature = float(decoder["candidate_temperature"])
        current_log_probs = torch.log_softmax(scores / temperature, dim=0)
        current_probabilities = current_log_probs.exp()
        old_probabilities = torch.tensor(
            decoder["candidate_probabilities"], dtype=torch.float32, device="cuda"
        )
        chosen_index = int(decoder["chosen_index"])
        chosen_logprob = current_log_probs[chosen_index]
        entropy = -(current_probabilities * current_log_probs).sum()
        categorical_kl = (
            current_probabilities
            * (current_log_probs - old_probabilities.clamp_min(1e-12).log())
        ).sum()
        return (
            chosen_logprob,
            entropy,
            categorical_kl,
            current_probabilities,
            old_probabilities,
            chosen_index,
        )

    if args.objective == "sequence-clip":
        model.eval()
        old_logp_path = (args.output / "old_sequence_logp.jsonl").open("w", encoding="utf-8")
        with torch.no_grad():
            for index, row in enumerate(calls):
                sequence_logp, _ = completion_statistics(row)
                row["old_sequence_logp"] = float(sequence_logp)
                old_logp_path.write(
                    json.dumps({"call_index": index, "old_sequence_logp": float(sequence_logp)})
                    + "\n"
                )
        old_logp_path.close()
        model.train()
    elif args.objective == "candidate-clip":
        # Rollouts were collected in inference mode.  Keeping dropout disabled
        # here ensures that the categorical PPO ratio is defined against that
        # exact behaviour distribution while gradients remain enabled.
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.eval()
        parity_errors: list[float] = []
        with torch.no_grad():
            for row in calls[: min(8, len(calls))]:
                _, _, _, current_probabilities, old_probabilities, _ = candidate_statistics(row)
                parity_errors.append(
                    float((current_probabilities - old_probabilities).abs().max())
                )
        max_probability_error = max(parity_errors, default=0.0)
        parity = {
            "checked_calls": len(parity_errors),
            "max_probability_abs_error": max_probability_error,
            "threshold": 0.005,
        }
        (args.output / "candidate_behavior_parity.json").write_text(
            json.dumps(parity, indent=2) + "\n", encoding="utf-8"
        )
        if max_probability_error > 0.005:
            raise ValueError(
                "candidate behaviour-policy parity failed; refusing an off-policy update"
            )
        if args.verify_candidate_parity:
            (args.output / "PARITY_VERIFIED").write_text(
                f"checked_calls={len(parity_errors)}\n"
                f"max_probability_abs_error={max_probability_error:.12g}\n",
                encoding="utf-8",
            )
            return
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    updates = max(1, args.epochs * math.ceil(len(calls) / args.gradient_accumulation))
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, updates // 10), updates)
    history = (args.output / "history.jsonl").open("w", encoding="utf-8")
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    for epoch in range(args.epochs):
        for index, row in enumerate(calls):
            if args.objective == "candidate-clip":
                (
                    policy_logp,
                    entropy,
                    approximate_kl,
                    _current_probabilities,
                    old_probabilities,
                    chosen_index,
                ) = candidate_statistics(row)
                old_logp = old_probabilities[chosen_index].clamp_min(1e-12).log()
            else:
                policy_logp, entropy = completion_statistics(row)
                approximate_kl = torch.tensor(0.0, device="cuda")
            advantage = advantages[tuple(row["terminal_key"])]
            if args.objective in {"sequence-clip", "candidate-clip"}:
                if args.objective == "sequence-clip":
                    old_logp = torch.tensor(row["old_sequence_logp"], device="cuda")
                ratio = torch.exp(torch.clamp(policy_logp - old_logp, -20.0, 20.0))
                clipped_ratio = torch.clamp(
                    ratio, 1.0 - args.clip_epsilon_low, 1.0 + args.clip_epsilon_high
                )
                advantage_tensor = torch.tensor(float(advantage), device="cuda")
                policy_loss = -torch.minimum(
                    ratio * advantage_tensor, clipped_ratio * advantage_tensor
                )
            else:
                ratio = torch.tensor(1.0, device="cuda")
                policy_loss = -float(advantage) * policy_logp
            if args.objective == "sequence-clip":
                approximate_kl = 0.5 * (policy_logp - old_logp).square()
            grpo_loss = (
                policy_loss
                - args.entropy_coef * entropy
                + args.sequence_kl_coef * approximate_kl
            )
            (grpo_loss / args.gradient_accumulation).backward()

            replay_applied = False
            sft_loss_value = 0.0
            if (
                sft_dataset is not None
                and replay_rng.random() < args.sft_replay_probability
            ):
                use_tool_replay = (
                    replay_rng.random() < args.sft_tool_replay_fraction
                )
                replay_pool = tool_replay_indices if use_tool_replay else other_replay_indices
                replay_index = replay_rng.choice(replay_pool)
                replay_item = sft_dataset[replay_index]
                sft_outputs = model(
                    input_ids=replay_item["input_ids"].unsqueeze(0).cuda(),
                    attention_mask=replay_item["attention_mask"].unsqueeze(0).cuda(),
                    labels=replay_item["labels"].unsqueeze(0).cuda(),
                    use_cache=False,
                )
                sft_loss = sft_outputs.loss
                (
                    args.sft_replay_weight * sft_loss / args.gradient_accumulation
                ).backward()
                replay_applied = True
                sft_loss_value = float(sft_loss.detach())
            loss_value = float(grpo_loss.detach()) + args.sft_replay_weight * sft_loss_value
            if (index + 1) % args.gradient_accumulation == 0 or index + 1 == len(calls):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                event = {
                    "step": global_step, "epoch": epoch,
                    "loss": loss_value, "grpo_loss": float(grpo_loss.detach()),
                    "policy_loss": float(policy_loss.detach()),
                    "entropy": float(entropy.detach()), "advantage": advantage,
                    "policy_logprob": float(policy_logp.detach()),
                    "sequence_ratio": float(ratio.detach()),
                    "approximate_sequence_kl": float(approximate_kl.detach()),
                    "sft_replay_applied": replay_applied,
                    "sft_loss": sft_loss_value,
                    "learning_rate": scheduler.get_last_lr()[0],
                }
                history.write(json.dumps(event, separators=(",", ":")) + "\n")
                history.flush()
    history.close()
    final = args.output / "final"
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    (args.output / "COMPLETED").write_text(f"updates={global_step}\n")


if __name__ == "__main__":
    main()
