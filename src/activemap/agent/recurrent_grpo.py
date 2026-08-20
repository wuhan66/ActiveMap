"""Audited trajectory rewards for recurrent executable-map GRPO."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class ExecutableRewardConfig:
    """Weights for a terminal reward computed after executable map writeback."""

    map_quality: float = 1.0
    topology_quality: float = 0.25
    false_edit: float = 1.0
    missed_edit: float = 0.5
    wrong_edit: float = 0.25
    normalized_cost: float = 0.10
    invalid_action: float = 1.0
    failed_tool: float = 0.10


REQUIRED_EXECUTABLE_FIELDS = {
    "map_quality_before",
    "map_quality_after",
    "topology_quality_before",
    "topology_quality_after",
    "false_edit",
    "missed_edit",
    "wrong_edit",
    "spent_cost",
    "budget",
    "invalid_action_count",
    "failed_tool_count",
}


def executable_trajectory_reward(
    row: Mapping[str, Any],
    config: ExecutableRewardConfig | None = None,
    *,
    false_edit_lagrange: float = 0.0,
) -> tuple[float, dict[str, float]]:
    """Score one complete trajectory after vector writeback and topology checks."""

    missing = REQUIRED_EXECUTABLE_FIELDS - row.keys()
    if missing:
        raise ValueError(f"trajectory lacks executable reward fields: {sorted(missing)}")
    config = config or ExecutableRewardConfig()
    budget = float(row["budget"])
    if budget <= 0:
        raise ValueError("trajectory budget must be positive")
    components = {
        "map_quality_delta": float(row["map_quality_after"])
        - float(row["map_quality_before"]),
        "topology_quality_delta": float(row["topology_quality_after"])
        - float(row["topology_quality_before"]),
        "false_edit": float(bool(row["false_edit"])),
        "missed_edit": float(bool(row["missed_edit"])),
        "wrong_edit": float(bool(row["wrong_edit"])),
        "normalized_cost": float(row["spent_cost"]) / budget,
        "invalid_action": float(row["invalid_action_count"]),
        "failed_tool": float(row["failed_tool_count"]),
    }
    values = np.asarray(list(components.values()), dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("trajectory reward components must be finite")
    reward = (
        config.map_quality * components["map_quality_delta"]
        + config.topology_quality * components["topology_quality_delta"]
        - (config.false_edit + false_edit_lagrange) * components["false_edit"]
        - config.missed_edit * components["missed_edit"]
        - config.wrong_edit * components["wrong_edit"]
        - config.normalized_cost * components["normalized_cost"]
        - config.invalid_action * components["invalid_action"]
        - config.failed_tool * components["failed_tool"]
    )
    return float(reward), components


def group_relative_advantages(
    rewards: Iterable[float], *, epsilon: float = 1e-6
) -> np.ndarray:
    """Return GRPO advantages normalized within one initial-state group."""

    values = np.asarray(list(rewards), dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("GRPO requires at least two rewards per group")
    if not np.isfinite(values).all():
        raise ValueError("GRPO rewards must be finite")
    scale = float(values.std())
    if scale <= epsilon:
        return np.zeros_like(values)
    return (values - values.mean()) / (scale + epsilon)


def constrained_proxy_rewards(
    rows: Iterable[Mapping[str, Any]],
    *,
    reward_key: str = "episode_utility_v2_proxy_balanced",
    false_edit_lagrange: float = 0.0,
) -> np.ndarray:
    """Apply an additional false-edit constraint to legacy proxy rewards."""

    if false_edit_lagrange < 0:
        raise ValueError("false-edit Lagrange multiplier cannot be negative")
    rewards = []
    for row in rows:
        if reward_key not in row:
            raise ValueError(f"trajectory lacks proxy reward field: {reward_key}")
        reward = float(row[reward_key])
        false_edit = float(bool(row.get("false_edit", False)))
        constrained = reward - false_edit_lagrange * false_edit
        if not np.isfinite(constrained):
            raise ValueError("constrained proxy rewards must be finite")
        rewards.append(constrained)
    return np.asarray(rewards, dtype=np.float64)


def informative_group_ids(
    groups: Mapping[Any, Iterable[Mapping[str, Any]]],
    *,
    reward_key: str = "episode_utility_v2_proxy_balanced",
    false_edit_lagrange: float = 0.0,
    minimum_group_size: int = 4,
    minimum_reward_std: float = 1e-6,
) -> set[Any]:
    """Select groups that can produce a non-zero group-relative gradient."""

    if minimum_group_size < 2 or minimum_reward_std < 0:
        raise ValueError("invalid informative-group thresholds")
    selected = set()
    for group_id, rows_iter in groups.items():
        rows = list(rows_iter)
        if len(rows) < minimum_group_size:
            continue
        rewards = constrained_proxy_rewards(
            rows,
            reward_key=reward_key,
            false_edit_lagrange=false_edit_lagrange,
        )
        if float(rewards.std()) > minimum_reward_std:
            selected.add(group_id)
    return selected


def sequence_clipped_surrogate(
    current_logp: float,
    old_logp: float,
    advantage: float,
    *,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.28,
) -> float:
    """Return the GSPO-style sequence-level clipped surrogate objective."""

    if epsilon_low <= 0 or epsilon_high <= 0:
        raise ValueError("clip epsilons must be positive")
    values = np.asarray([current_logp, old_logp, advantage], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("surrogate inputs must be finite")
    ratio = float(np.exp(np.clip(current_logp - old_logp, -20.0, 20.0)))
    clipped_ratio = float(np.clip(ratio, 1.0 - epsilon_low, 1.0 + epsilon_high))
    return float(min(ratio * advantage, clipped_ratio * advantage))


def audit_trajectory_groups(
    rows: Iterable[Mapping[str, Any]],
    *,
    minimum_group_size: int = 4,
    minimum_variable_group_rate: float = 0.20,
    minimum_nonstop_rate: float = 0.05,
    minimum_keep_trajectories: int = 0,
    minimum_commit_trajectories: int = 0,
    minimum_tool_trajectories: int = 0,
    reward_epsilon: float = 1e-6,
) -> dict[str, Any]:
    """Gate recurrent GRPO on exploration and executable reward variance."""

    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("split") == "test":
            raise ValueError("recurrent GRPO audit refuses test trajectories")
        group_id = str(row.get("group_id", ""))
        rollout_id = str(row.get("rollout_id", ""))
        if not group_id or not rollout_id:
            raise ValueError("trajectory requires group_id and rollout_id")
        groups.setdefault(group_id, []).append(row)
    if not groups:
        raise ValueError("trajectory audit received no groups")
    if min(
        minimum_keep_trajectories,
        minimum_commit_trajectories,
        minimum_tool_trajectories,
    ) < 0:
        raise ValueError("minimum trajectory coverage thresholds cannot be negative")

    variable = 0
    rewards: list[float] = []
    non_stop = 0
    trajectories = 0
    terminal_action_counts: Counter[str] = Counter()
    target_action_counts: Counter[str] = Counter()
    tool_trajectories = 0
    undersized = []
    duplicate_rollouts = []
    for group_id, group in groups.items():
        rollout_ids = [str(row["rollout_id"]) for row in group]
        if len(set(rollout_ids)) != len(rollout_ids):
            duplicate_rollouts.append(group_id)
        if len(group) < minimum_group_size:
            undersized.append(group_id)
        group_rewards = []
        for row in group:
            reward, _ = executable_trajectory_reward(row)
            group_rewards.append(reward)
            rewards.append(reward)
            trajectories += 1
            non_stop += bool(row.get("contains_nonstop_action", False))
            terminal_action = str(
                row.get("prediction", row.get("terminal_action", "UNKNOWN"))
            ).strip() or "UNKNOWN"
            terminal_action_counts[terminal_action] += 1
            target_action = str(row.get("target", "UNKNOWN")).strip() or "UNKNOWN"
            target_action_counts[target_action] += 1
            tool_trajectories += int(row.get("tool_calls", 0)) > 0
        variable += float(np.std(group_rewards)) > reward_epsilon
    if duplicate_rollouts:
        raise ValueError(f"duplicate rollout ids in {len(duplicate_rollouts)} groups")
    variable_rate = variable / len(groups)
    non_stop_rate = non_stop / trajectories
    keep_trajectories = terminal_action_counts["REJECT"]
    commit_trajectories = sum(
        count
        for action, count in terminal_action_counts.items()
        if action.startswith("COMMIT")
    )
    gates = {
        "minimum_group_size": not undersized,
        "reward_variance": variable_rate >= minimum_variable_group_rate,
        "nonstop_exploration": non_stop_rate >= minimum_nonstop_rate,
        "executed_keep_support": keep_trajectories >= minimum_keep_trajectories,
        "executed_commit_support": commit_trajectories >= minimum_commit_trajectories,
        "tool_trajectory_support": tool_trajectories >= minimum_tool_trajectories,
    }
    return {
        "schema_version": "activemap-recurrent-grpo-readiness-v1",
        "groups": len(groups),
        "trajectories": trajectories,
        "minimum_observed_group_size": min(len(group) for group in groups.values()),
        "variable_reward_groups": variable,
        "variable_reward_group_rate": variable_rate,
        "nonstop_trajectory_rate": non_stop_rate,
        "executed_terminal_action_counts": dict(sorted(terminal_action_counts.items())),
        "target_terminal_action_counts": dict(sorted(target_action_counts.items())),
        "executed_keep_trajectories": keep_trajectories,
        "executed_commit_trajectories": commit_trajectories,
        "tool_trajectories": tool_trajectories,
        "minimum_keep_trajectories": minimum_keep_trajectories,
        "minimum_commit_trajectories": minimum_commit_trajectories,
        "minimum_tool_trajectories": minimum_tool_trajectories,
        "mean_reward": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "undersized_groups": undersized,
        "reward_config": asdict(ExecutableRewardConfig()),
        "gates": gates,
        "ready_for_recurrent_grpo": all(gates.values()),
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "test_assets_read": False,
    }


def update_false_edit_lagrange(
    current: float,
    observed_false_edit_rate: float,
    target_false_edit_rate: float,
    *,
    learning_rate: float = 0.05,
    maximum: float = 20.0,
) -> float:
    """Projected dual update for the false-edit constraint."""

    if learning_rate <= 0 or maximum <= 0:
        raise ValueError("dual learning rate and maximum must be positive")
    updated = current + learning_rate * (
        observed_false_edit_rate - target_false_edit_rate
    )
    return float(np.clip(updated, 0.0, maximum))
