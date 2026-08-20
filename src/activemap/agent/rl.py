"""Reward components for constrained structured-action policy optimization."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from activemap.agent.records import AgentAction, AgentActionType, AgentObservation
from activemap.models import EditOperation


def completion_text(completion: Any) -> str:
    """Extract assistant text from plain or conversational TRL completions."""

    if isinstance(completion, str):
        return completion
    if isinstance(completion, Mapping):
        return str(completion.get("content", ""))
    if isinstance(completion, Sequence):
        for message in reversed(completion):
            if isinstance(message, Mapping) and message.get("role") == "assistant":
                return str(message.get("content", ""))
        if completion and isinstance(completion[-1], Mapping):
            return str(completion[-1].get("content", ""))
    return str(completion)


def extract_json_object(text: str) -> dict[str, Any]:
    """Decode the first complete JSON object without accepting trailing prose."""

    start = text.find("{")
    if start < 0:
        raise ValueError("completion contains no JSON object")
    value, end = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(value, dict):
        raise ValueError("completion JSON must be an object")
    if text[start + end :].strip():
        raise ValueError("completion contains text after the JSON action")
    return value


def parse_completion_action(completion: Any) -> AgentAction:
    return AgentAction.model_validate(extract_json_object(completion_text(completion)))


def reward_action_key(action: AgentAction) -> str:
    """Return a utility key that distinguishes grounded tool/evidence calls."""

    if action.action != AgentActionType.USE_TOOL or action.tool_call is None:
        return action.key
    evidence_id = action.tool_call.inputs.get("evidence_id")
    return f"USE_TOOL:{action.tool_call.tool.value}:{evidence_id}"


def action_is_executable(action: AgentAction, observation: AgentObservation) -> bool:
    if action.action == AgentActionType.ACQUIRE:
        return action.evidence_id in {candidate.evidence_id for candidate in observation.candidates}
    if action.action == AgentActionType.USE_TOOL:
        if action.tool_call is None:
            return False
        evidence_id = action.tool_call.inputs.get("evidence_id")
        return (
            action.tool_call.tool in observation.available_tools
            and evidence_id in observation.selected_evidence_ids
        )
    return action.action in observation.terminal_actions


def _aligned_columns(completions: Sequence[Any], values: Sequence[Any], name: str) -> None:
    if len(values) != len(completions):
        raise ValueError(
            f"reward column {name!r} has {len(values)} values for {len(completions)} completions"
        )


def task_utility_reward(
    completions: Sequence[Any],
    oracle_utilities: Sequence[Mapping[str, float]],
    observation_json: Sequence[str],
    **_: Any,
) -> list[float]:
    """Return frozen counterfactual utility, with an explicit invalid floor."""

    _aligned_columns(completions, oracle_utilities, "oracle_utilities")
    _aligned_columns(completions, observation_json, "observation_json")
    rewards: list[float] = []
    for completion, utilities, serialized_observation in zip(
        completions, oracle_utilities, observation_json, strict=True
    ):
        try:
            action = parse_completion_action(completion)
            observation = AgentObservation.model_validate_json(serialized_observation)
            if not action_is_executable(action, observation):
                raise ValueError("action is not executable")
            rewards.append(float(utilities.get(reward_action_key(action), -1.0)))
        except (TypeError, ValueError):
            rewards.append(-1.0)
    return rewards


def executable_schema_reward(
    completions: Sequence[Any], observation_json: Sequence[str], **_: Any
) -> list[float]:
    """Apply a hard constraint penalty to malformed or non-executable actions."""

    _aligned_columns(completions, observation_json, "observation_json")
    rewards: list[float] = []
    for completion, serialized_observation in zip(completions, observation_json, strict=True):
        try:
            action = parse_completion_action(completion)
            observation = AgentObservation.model_validate_json(serialized_observation)
            rewards.append(0.0 if action_is_executable(action, observation) else -1.0)
        except (TypeError, ValueError):
            rewards.append(-1.0)
    return rewards


def terminal_safety_reward(
    completions: Sequence[Any], target_action_key: Sequence[str], **_: Any
) -> list[float]:
    """Penalize false edits most strongly, then missed and wrong edits."""

    _aligned_columns(completions, target_action_key, "target_action_key")
    rewards: list[float] = []
    for completion, target in zip(completions, target_action_key, strict=True):
        try:
            action = parse_completion_action(completion)
        except (TypeError, ValueError):
            rewards.append(0.0)
            continue
        if action.action not in {AgentActionType.COMMIT, AgentActionType.REJECT}:
            rewards.append(0.0)
        elif target == AgentActionType.REJECT.value and action.action == AgentActionType.COMMIT:
            rewards.append(-1.0)
        elif target.startswith("COMMIT:") and action.action == AgentActionType.REJECT:
            rewards.append(-0.75)
        elif target.startswith("COMMIT:") and action.key != target:
            rewards.append(-0.5)
        else:
            rewards.append(0.0)
    return rewards


def sparse_acquisition_reward(
    completions: Sequence[Any],
    oracle_utilities: Sequence[Mapping[str, float]],
    stop_utility: Sequence[float],
    **_: Any,
) -> list[float]:
    """Penalize evidence/tool calls that cannot beat the best terminal action."""

    _aligned_columns(completions, oracle_utilities, "oracle_utilities")
    _aligned_columns(completions, stop_utility, "stop_utility")
    rewards: list[float] = []
    for completion, utilities, stop in zip(
        completions, oracle_utilities, stop_utility, strict=True
    ):
        try:
            action = parse_completion_action(completion)
        except (TypeError, ValueError):
            rewards.append(0.0)
            continue
        if action.action not in {AgentActionType.ACQUIRE, AgentActionType.USE_TOOL}:
            rewards.append(0.0)
            continue
        rewards.append(
            -1.0 if float(utilities.get(reward_action_key(action), -1.0)) <= float(stop) else 0.0
        )
    return rewards


def target_terminal_action(utilities: Mapping[str, float]) -> tuple[str, float]:
    """Recover the ground-truth terminal action from frozen terminal utilities."""

    terminal = {
        key: float(value)
        for key, value in utilities.items()
        if key == AgentActionType.REJECT.value or key.startswith("COMMIT:")
    }
    if not terminal:
        raise ValueError("oracle utilities contain no terminal action")
    key = max(terminal, key=terminal.__getitem__)
    if key.startswith("COMMIT:"):
        edit = key.split(":", 1)[1]
        if edit not in {item.value for item in EditOperation if item != EditOperation.KEEP}:
            raise ValueError(f"invalid terminal edit in oracle utilities: {edit}")
    return key, terminal[key]
