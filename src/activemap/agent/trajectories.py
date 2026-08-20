"""Build agent trajectories, SFT messages, and preference pairs from oracle states."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from activemap.agent.identifiers import public_evidence_id, public_task_id
from activemap.agent.records import (
    AgentAction,
    AgentActionType,
    AgentCandidate,
    AgentObservation,
    AgentTrajectory,
    AgentTransition,
)
from activemap.agent.tools import CounterfactualBeliefUpdater, belief_from_features
from activemap.models import EditOperation
from activemap.selector_records import SelectorSample

ScoreFunction = Callable[[SelectorSample], np.ndarray]

SYSTEM_PROMPT = (
    "You are the ActiveMap maintenance controller. Choose exactly one structured action. "
    "Use exactly one of these JSON forms with no extra keys: "
    '{"action":"ACQUIRE","evidence_id":"<candidate id>"}; '
    '{"action":"COMMIT","edit":"ADD|DELETE|RESHAPE"}; '
    '{"action":"REJECT"}; or '
    '{"action":"USE_TOOL","tool_call":<listed tool call>}. '
    "ACQUIRE must use an id from candidates, and USE_TOOL is allowed only when the tool is "
    "listed in available_tools. Every action has a budget cost. Return JSON only."
)


def load_selector_states(path: Path, *, split: str) -> list[SelectorSample]:
    records: list[SelectorSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = SelectorSample.model_validate_json(line)
            except Exception as exc:
                raise ValueError(f"invalid selector sample at line {line_number}") from exc
            if record.split == split:
                records.append(record)
    if not records:
        raise ValueError(f"no selector samples found for split={split!r} in {path}")
    return records


def _ground_truth(sample: SelectorSample) -> EditOperation:
    value = sample.metadata.get("gt_edit")
    return EditOperation(str(value)) if value is not None else sample.edit_type


def _terminal_action(sample: SelectorSample) -> AgentAction:
    target = _ground_truth(sample)
    if target == EditOperation.KEEP:
        return AgentAction(action=AgentActionType.REJECT)
    return AgentAction(action=AgentActionType.COMMIT, edit=target)


def _terminal_reward(sample: SelectorSample, action: AgentAction) -> float:
    target = _ground_truth(sample)
    if action.action == AgentActionType.REJECT:
        return 1.0 if target == EditOperation.KEEP else -0.75
    if action.edit == target:
        return 1.0
    return -1.0 if target == EditOperation.KEEP else -0.5


def _all_terminal_utilities(sample: SelectorSample) -> dict[str, float]:
    correct = _terminal_action(sample)
    reject = AgentAction(action=AgentActionType.REJECT)
    utilities = {
        AgentActionType.REJECT.value: (
            sample.stop_utility
            if reject.key == correct.key
            else sample.stop_utility - 0.75
        )
    }
    for edit in EditOperation:
        if edit == EditOperation.KEEP:
            continue
        action = AgentAction(action=AgentActionType.COMMIT, edit=edit)
        if action.key == correct.key:
            utilities[action.key] = sample.stop_utility
        elif _ground_truth(sample) == EditOperation.KEEP:
            utilities[action.key] = sample.stop_utility - 1.0
        else:
            utilities[action.key] = sample.stop_utility - 0.5
    return utilities


def _oracle_action(sample: SelectorSample) -> AgentAction:
    best_index = int(np.argmax(sample.oracle_utilities))
    if sample.oracle_utilities[best_index] > sample.stop_utility:
        return AgentAction(
            action=AgentActionType.ACQUIRE,
            evidence_id=sample.evidence_ids[best_index],
        )
    return _terminal_action(sample)


def _public_action(action: AgentAction) -> AgentAction:
    if action.action != AgentActionType.ACQUIRE:
        return action
    return AgentAction(
        action=AgentActionType.ACQUIRE,
        evidence_id=public_evidence_id(str(action.evidence_id)),
    )


def _belief(sample: SelectorSample, selected: list[str]) -> Any:
    if selected and isinstance(sample.metadata.get("evidence_predictions"), dict):
        return CounterfactualBeliefUpdater(sample).fuse(selected)
    return belief_from_features(sample)


def _observation(
    sample: SelectorSample,
    *,
    score_fn: ScoreFunction,
    top_k: int,
    required_evidence_id: str | None = None,
) -> tuple[AgentObservation, bool]:
    scores = np.asarray(score_fn(sample), dtype=np.float64)
    if scores.shape not in {(len(sample.evidence_ids),), (len(sample.evidence_ids) + 1,)}:
        raise ValueError(f"selector score shape mismatch for {sample.sample_id}")
    terminal_score = float(scores[-1]) if len(scores) == len(sample.evidence_ids) + 1 else None
    ranked = sorted(
        range(len(sample.evidence_ids)), key=lambda index: float(scores[index]), reverse=True
    )
    natural = ranked[:top_k]
    oracle_visible = required_evidence_id is None or required_evidence_id in {
        sample.evidence_ids[index] for index in natural
    }
    selected_indices = list(natural)
    if required_evidence_id is not None and not oracle_visible:
        required_index = sample.evidence_ids.index(required_evidence_id)
        if selected_indices:
            selected_indices[-1] = required_index
        else:
            selected_indices.append(required_index)
    selected = list(sample.metadata.get("selected_evidence_ids", []))
    budget = float(sample.metadata.get("budget", 1.0))
    remaining = float(np.clip(sample.state_features[0], 0.0, 1.0) * budget)
    candidates = [
        AgentCandidate(
            evidence_id=public_evidence_id(sample.evidence_ids[index]),
            cost=sample.evidence_costs[index],
            selector_score=float(scores[index]),
            features=sample.evidence_features[index],
        )
        for index in selected_indices
    ]
    return (
        AgentObservation(
            task_id=public_task_id(
                str(sample.metadata.get("source_episode", sample.sample_id))
            ),
            split=sample.split,
            step=int(sample.metadata.get("oracle_step", 0)),
            initial_budget=budget,
            remaining_budget=remaining,
            spent_cost=max(budget - remaining, 0.0),
            selected_evidence_ids=[public_evidence_id(item) for item in selected],
            belief=_belief(sample, selected),
            candidates=candidates,
            terminal_score=terminal_score,
        ),
        oracle_visible,
    )


def _synthetic_terminal_observation(
    sample: SelectorSample, observation: AgentObservation, acquired: str
) -> AgentObservation:
    selected = [*list(sample.metadata.get("selected_evidence_ids", [])), acquired]
    acquired_index = sample.evidence_ids.index(acquired)
    remaining = max(observation.remaining_budget - sample.evidence_costs[acquired_index], 0.0)
    return AgentObservation(
        task_id=observation.task_id,
        split=observation.split,
        step=observation.step + 1,
        initial_budget=observation.initial_budget,
        remaining_budget=remaining,
        spent_cost=observation.initial_budget - remaining,
        selected_evidence_ids=[public_evidence_id(item) for item in selected],
        belief=_belief(sample, selected),
        candidates=[],
        terminal_score=None,
    )


def _action_from_key(key: str) -> AgentAction:
    if key == AgentActionType.REJECT.value:
        return AgentAction(action=AgentActionType.REJECT)
    prefix, value = key.split(":", maxsplit=1)
    if prefix == AgentActionType.ACQUIRE.value:
        return AgentAction(action=AgentActionType.ACQUIRE, evidence_id=value)
    return AgentAction(action=AgentActionType.COMMIT, edit=EditOperation(value))


def build_agent_trajectories(
    samples: list[SelectorSample],
    *,
    score_fn: ScoreFunction,
    top_k: int = 8,
) -> tuple[list[AgentTrajectory], dict[str, float | int]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    grouped: dict[tuple[str, float], list[SelectorSample]] = defaultdict(list)
    for sample in samples:
        key = (
            str(sample.metadata.get("source_episode", sample.sample_id)),
            float(sample.metadata.get("budget", 1.0)),
        )
        grouped[key].append(sample)

    trajectories: list[AgentTrajectory] = []
    acquisition_states = 0
    natural_top_k_hits = 0
    for (task_id, budget), states in sorted(grouped.items()):
        ordered = sorted(states, key=lambda item: int(item.metadata.get("oracle_step", 0)))
        transitions: list[AgentTransition] = []
        for index, sample in enumerate(ordered):
            raw_action = _oracle_action(sample)
            required = (
                raw_action.evidence_id
                if raw_action.action == AgentActionType.ACQUIRE
                else None
            )
            action = _public_action(raw_action)
            observation, visible = _observation(
                sample,
                score_fn=score_fn,
                top_k=top_k,
                required_evidence_id=required,
            )
            utilities = {
                f"ACQUIRE:{public_evidence_id(evidence_id)}": float(utility)
                for evidence_id, utility in zip(
                    sample.evidence_ids, sample.oracle_utilities, strict=True
                )
            }
            utilities.update(_all_terminal_utilities(sample))
            if action.action == AgentActionType.ACQUIRE:
                acquisition_states += 1
                natural_top_k_hits += int(visible)
                reward = float(utilities[action.key])
                if index + 1 < len(ordered):
                    next_sample = ordered[index + 1]
                    next_action = _oracle_action(next_sample)
                    next_required = (
                        next_action.evidence_id
                        if next_action.action == AgentActionType.ACQUIRE
                        else None
                    )
                    next_observation, _ = _observation(
                        next_sample,
                        score_fn=score_fn,
                        top_k=top_k,
                        required_evidence_id=next_required,
                    )
                else:
                    next_observation = _synthetic_terminal_observation(
                        sample, observation, str(raw_action.evidence_id)
                    )
                transitions.append(
                    AgentTransition(
                        observation=observation,
                        action=action,
                        reward=reward,
                        done=False,
                        next_observation=next_observation,
                        oracle_action=action,
                        oracle_utilities=utilities,
                    )
                )
                if index + 1 == len(ordered):
                    terminal = _terminal_action(sample)
                    terminal_utilities = _all_terminal_utilities(sample)
                    transitions.append(
                        AgentTransition(
                            observation=next_observation,
                            action=terminal,
                            reward=_terminal_reward(sample, terminal),
                            done=True,
                            oracle_action=terminal,
                            oracle_utilities=terminal_utilities,
                        )
                    )
            else:
                transitions.append(
                    AgentTransition(
                        observation=observation,
                        action=action,
                        reward=_terminal_reward(sample, action),
                        done=True,
                        oracle_action=action,
                        oracle_utilities=utilities,
                    )
                )
                break
        if not transitions:
            continue
        public_id = public_task_id(task_id)
        trajectory_id = f"{public_id}__b{str(budget).replace('.', 'p')}"
        trajectories.append(
            AgentTrajectory(
                trajectory_id=trajectory_id,
                task_id=public_id,
                split=ordered[0].split,
                budget=budget,
                transitions=transitions,
                total_reward=float(sum(item.reward for item in transitions)),
                metadata={"top_k": top_k, "state_count": len(ordered)},
            )
        )
    return trajectories, {
        "trajectory_count": len(trajectories),
        "transition_count": sum(len(item.transitions) for item in trajectories),
        "acquisition_state_count": acquisition_states,
        "natural_top_k_recall": natural_top_k_hits / max(acquisition_states, 1),
    }


def _prompt(observation: AgentObservation) -> str:
    return observation.model_dump_json(exclude_none=True)


def write_agent_datasets(
    trajectories: list[AgentTrajectory], output_dir: Path
) -> dict[str, int | str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / "trajectories.jsonl"
    sft_path = output_dir / "sft.jsonl"
    preference_path = output_dir / "preferences.jsonl"
    sft_count = 0
    preference_count = 0
    with (
        trajectory_path.open("w", encoding="utf-8") as trajectory_handle,
        sft_path.open("w", encoding="utf-8") as sft_handle,
        preference_path.open("w", encoding="utf-8") as preference_handle,
    ):
        for trajectory in trajectories:
            trajectory_handle.write(trajectory.model_dump_json(exclude_none=True) + "\n")
            for transition in trajectory.transitions:
                prompt = _prompt(transition.observation)
                chosen = transition.action.model_dump_json(exclude_none=True)
                sft_handle.write(
                    json.dumps(
                        {
                            "messages": [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": prompt},
                                {"role": "assistant", "content": chosen},
                            ],
                            "trajectory_id": trajectory.trajectory_id,
                            "step": transition.observation.step,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                sft_count += 1
                alternatives = sorted(
                    (
                        (key, value)
                        for key, value in transition.oracle_utilities.items()
                        if key != transition.action.key
                    ),
                    key=lambda item: item[1],
                    reverse=True,
                )
                if alternatives:
                    rejected = _action_from_key(alternatives[0][0]).model_dump_json(
                        exclude_none=True
                    )
                    preference_handle.write(
                        json.dumps(
                            {
                                "system": SYSTEM_PROMPT,
                                "prompt": f"{SYSTEM_PROMPT}\n{prompt}\nAction:",
                                "chosen": chosen,
                                "rejected": rejected,
                                "chosen_utility": transition.oracle_utilities.get(
                                    transition.action.key, transition.reward
                                ),
                                "rejected_utility": alternatives[0][1],
                                "trajectory_id": trajectory.trajectory_id,
                            },
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    preference_count += 1
    return {
        "trajectory_path": str(trajectory_path.resolve()),
        "sft_path": str(sft_path.resolve()),
        "preference_path": str(preference_path.resolve()),
        "trajectory_count": len(trajectories),
        "sft_count": sft_count,
        "preference_count": preference_count,
    }


def build_agent_datasets(
    samples_path: Path,
    output_dir: Path,
    *,
    split: str = "train",
    score_fn: ScoreFunction | None = None,
    top_k: int = 8,
) -> dict[str, Any]:
    if split not in {"train", "val", "test"}:
        raise ValueError("agent dataset split must be train, val, or test")
    test_assets_read = split == "test"
    if test_assets_read:
        from activemap.frozen_test import assert_frozen_test_access

        assert_frozen_test_access()
        if output_dir.exists():
            raise FileExistsError(
                f"refusing to overwrite frozen test agent labels: {output_dir}"
            )
    samples = load_selector_states(samples_path, split=split)
    scorer = score_fn or (
        lambda sample: -np.asarray(sample.evidence_costs, dtype=np.float32)
    )
    trajectories, metrics = build_agent_trajectories(samples, score_fn=scorer, top_k=top_k)
    summary = {
        "samples": str(samples_path.resolve()),
        "split": split,
        "top_k": top_k,
        "scorer": "external" if score_fn is not None else "cheapest_baseline",
        "test_assets_read": test_assets_read,
        **metrics,
        **write_agent_datasets(trajectories, output_dir),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
