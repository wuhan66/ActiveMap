"""Closed-loop environment connecting selector, updater belief, and agent actions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

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
from activemap.agent.tools import (
    AgentPolicy,
    BeliefUpdater,
    ScoreFunction,
    ToolBeliefUpdater,
    belief_from_features,
)
from activemap.evaluation.episode_utility import UTILITY_PROFILES
from activemap.features import ONLINE_OBSERVABLE_STATE_CONTRACT
from activemap.geo_tools.records import GeoToolName, GeoToolResult
from activemap.geo_tools.registry import GeoToolRegistry
from activemap.models import EditOperation
from activemap.selector_records import SelectorSample


@dataclass(frozen=True)
class RewardConfig:
    correct_terminal: float = 1.0
    false_edit: float = 1.0
    missed_edit: float = 0.75
    wrong_edit: float = 0.5


class MapMaintenanceEnv:
    """One editable-object maintenance episode with auditable transitions."""

    def __init__(
        self,
        sample: SelectorSample,
        *,
        budget: float,
        score_fn: ScoreFunction,
        belief_updater: BeliefUpdater,
        top_k: int = 8,
        reward_config: RewardConfig | None = None,
        tool_registry: GeoToolRegistry | None = None,
        tool_belief_updater: ToolBeliefUpdater | None = None,
        asset_paths: Mapping[str, str] | None = None,
        tool_parameters: Mapping[str, Mapping[str, object]] | None = None,
        tool_inputs_by_evidence: Mapping[str, Mapping[str, object]] | None = None,
        public_identifiers: bool = False,
    ) -> None:
        if budget <= 0:
            raise ValueError("budget must be positive")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.sample = sample
        self.initial_budget = float(budget)
        self.score_fn = score_fn
        self.belief_updater = belief_updater
        self.top_k = top_k
        self.reward_config = reward_config or RewardConfig()
        self.tool_registry = tool_registry
        self.tool_belief_updater = tool_belief_updater
        self.asset_paths = dict(asset_paths) if asset_paths is not None else None
        self.tool_parameters = (
            {key: dict(value) for key, value in tool_parameters.items()}
            if tool_parameters is not None
            else None
        )
        self.tool_inputs_by_evidence = (
            {key: dict(value) for key, value in tool_inputs_by_evidence.items()}
            if tool_inputs_by_evidence is not None
            else None
        )
        self.public_identifiers = public_identifiers
        self.utility_mode = str(sample.metadata.get("utility_mode", "proxy"))
        self.cost_weight = float(sample.metadata.get("cost_weight", 0.0))
        online_state_contract = sample.metadata.get("online_state_contract")
        if online_state_contract is None:
            self.online_state7_source: str | None = None
        elif (
            isinstance(online_state_contract, Mapping)
            and dict(online_state_contract) == ONLINE_OBSERVABLE_STATE_CONTRACT
        ):
            self.online_state7_source = "fused_belief_confidence"
        else:
            raise ValueError("unsupported online_state_contract")
        # Legacy offline rollouts use state[7] as an oracle-derived utility
        # history. Online controllers instead reserve it for an observable belief
        # summary, so it must never seed the environment's internal utility state.
        self.initial_gain = (
            0.0 if self.online_state7_source is not None else float(sample.state_features[7])
        )
        self.preacquired_spent_cost = float(
            sample.metadata.get("preacquired_spent_cost", 0.0)
        )
        self.preacquired_evidence_penalty = float(
            sample.metadata.get("preacquired_evidence_penalty", 0.0)
        )
        if not 0.0 <= self.preacquired_spent_cost <= self.initial_budget:
            raise ValueError("preacquired_spent_cost must be within the episode budget")
        if self.preacquired_evidence_penalty < 0.0:
            raise ValueError("preacquired_evidence_penalty must be non-negative")
        if self.utility_mode == "executable":
            profile = UTILITY_PROFILES[str(sample.metadata.get("utility_profile", "balanced"))]
            outcomes = sample.metadata.get("executable_outcomes")
            if not isinstance(outcomes, dict):
                raise ValueError("executable utility requires executable_outcomes metadata")
            self.evidence_penalties = (
                np.asarray(sample.evidence_costs, dtype=np.float64)
                * profile.cost
                / self.initial_budget
            )
            self.quality_gains = np.asarray(
                [
                    float(outcomes[item]["terminal_score_before_cost"])
                    for item in sample.evidence_ids
                ],
                dtype=np.float64,
            )
        else:
            self.evidence_penalties = (
                np.asarray(sample.evidence_costs, dtype=np.float64) * self.cost_weight
            )
            self.evidence_penalties += (
                np.asarray(sample.false_edit_risks, dtype=np.float64)
                * sample.false_edit_penalty_weight
            )
            initial_marginals = np.asarray(sample.oracle_utilities, dtype=np.float64)
            self.quality_gains = self.initial_gain + np.maximum(
                initial_marginals + self.evidence_penalties, 0.0
            )
        prediction_ids = (
            set(sample.metadata["evidence_predictions"])
            if isinstance(sample.metadata.get("evidence_predictions"), dict)
            else set()
        )
        self.total_evidence_count = len(
            set(sample.evidence_ids)
            | set(sample.metadata.get("selected_evidence_ids", []))
            | prediction_ids
        )
        self.reset()

    def _exposed_evidence_id(self, raw_id: str) -> str:
        return public_evidence_id(raw_id) if self.public_identifiers else raw_id

    def _raw_evidence_id(self, exposed_id: str) -> str:
        if not self.public_identifiers:
            return exposed_id
        matches = [
            raw_id
            for raw_id in self.sample.evidence_ids
            if public_evidence_id(raw_id) == exposed_id
        ]
        selected_matches = [
            raw_id for raw_id in self.selected if public_evidence_id(raw_id) == exposed_id
        ]
        matches.extend(item for item in selected_matches if item not in matches)
        if len(matches) != 1:
            raise ValueError(f"unknown public evidence_id: {exposed_id}")
        return matches[0]

    def reset(self) -> AgentObservation:
        initial = list(self.sample.metadata.get("selected_evidence_ids", []))
        self.selected = initial
        self.available = [
            index
            for index, evidence_id in enumerate(self.sample.evidence_ids)
            if evidence_id not in initial
        ]
        self.remaining_budget = self.initial_budget - self.preacquired_spent_cost
        self.current_gain = self.initial_gain
        self.spent_penalty = self.preacquired_evidence_penalty
        self.step_index = 0
        self.done = False
        self.tool_history: list[GeoToolResult] = []
        self.belief = (
            self.belief_updater.fuse(self.selected)
            if self.selected
            else belief_from_features(self.sample)
        )
        return self.observation()

    def _marginal_utility(self, index: int) -> float:
        gain = float(self.quality_gains[index]) - self.current_gain
        if self.utility_mode != "executable":
            gain = max(gain, 0.0)
        return gain - float(self.evidence_penalties[index])

    def _current_sample(self, indices: list[int]) -> SelectorSample:
        state = list(self.sample.state_features)
        state[0] = self.remaining_budget / self.initial_budget
        state[1] = 1.0 - state[0]
        state[2:6] = self.belief.edit_probabilities
        state[6] = len(self.selected) / max(self.total_evidence_count, 1)
        state[7] = (
            self.belief.confidence
            if self.online_state7_source == "fused_belief_confidence"
            else self.current_gain
        )
        hypothesis = list(self.sample.hypothesis_features)
        hypothesis[:4] = self.belief.edit_probabilities
        hypothesis[12] = self.belief.uncertainty
        hypothesis[13] = self.belief.confidence
        return self.sample.model_copy(
            update={
                "hypothesis_features": hypothesis,
                "state_features": state,
                "evidence_ids": [self.sample.evidence_ids[index] for index in indices],
                "evidence_features": [self.sample.evidence_features[index] for index in indices],
                "evidence_costs": [self.sample.evidence_costs[index] for index in indices],
                "false_edit_risks": [self.sample.false_edit_risks[index] for index in indices],
                "oracle_utilities": [self._marginal_utility(index) for index in indices],
            }
        )

    def current_sample(self) -> SelectorSample:
        """Return the deployable state with currently affordable evidence only."""
        affordable = [
            index
            for index in self.available
            if self.sample.evidence_costs[index] <= self.remaining_budget
        ]
        return self._current_sample(affordable)

    def observation(self) -> AgentObservation:
        if self.done:
            raise RuntimeError("episode is done; call reset before requesting another observation")
        affordable = [
            index
            for index in self.available
            if self.sample.evidence_costs[index] <= self.remaining_budget
        ]
        if affordable:
            current = self._current_sample(affordable)
            scores = np.asarray(self.score_fn(current), dtype=np.float64)
            if scores.shape not in {(len(affordable),), (len(affordable) + 1,)}:
                raise ValueError("selector score shape does not match affordable actions")
            terminal_score = float(scores[-1]) if len(scores) == len(affordable) + 1 else None
            ranked = sorted(
                range(len(affordable)), key=lambda index: float(scores[index]), reverse=True
            )[: self.top_k]
            candidates = [
                AgentCandidate(
                    evidence_id=self._exposed_evidence_id(current.evidence_ids[index]),
                    cost=current.evidence_costs[index],
                    selector_score=float(scores[index]),
                    features=current.evidence_features[index],
                )
                for index in ranked
            ]
        else:
            terminal_score = None
            candidates = []
        return AgentObservation(
            task_id=(
                public_task_id(
                    str(self.sample.metadata.get("source_episode", self.sample.sample_id))
                )
                if self.public_identifiers
                else self.sample.sample_id
            ),
            split=self.sample.split,
            step=self.step_index,
            initial_budget=self.initial_budget,
            remaining_budget=max(self.remaining_budget, 0.0),
            spent_cost=self.initial_budget - self.remaining_budget,
            selected_evidence_ids=[self._exposed_evidence_id(item) for item in self.selected],
            belief=self.belief,
            candidates=candidates,
            terminal_score=terminal_score,
            available_tools=(
                [
                    name
                    for name in self.tool_registry.names()
                    if self.tool_registry.cost(name) <= self.remaining_budget
                ]
                if self.tool_registry
                else []
            ),
            tool_history=list(self.tool_history),
        )

    def _ground_truth(self) -> EditOperation:
        value = self.sample.metadata.get("gt_edit")
        return EditOperation(str(value)) if value is not None else self.sample.edit_type

    def _terminal_reward(self, action: AgentAction) -> float:
        target = self._ground_truth()
        if action.action == AgentActionType.REJECT:
            return (
                self.reward_config.correct_terminal
                if target == EditOperation.KEEP
                else -self.reward_config.missed_edit
            )
        if action.edit == target:
            return self.reward_config.correct_terminal
        if target == EditOperation.KEEP:
            return -self.reward_config.false_edit
        return -self.reward_config.wrong_edit

    def oracle_action(self) -> AgentAction:
        affordable = [
            index
            for index in self.available
            if self.sample.evidence_costs[index] <= self.remaining_budget
        ]
        if affordable:
            best = max(affordable, key=self._marginal_utility)
            if self._marginal_utility(best) > self.sample.stop_utility:
                return AgentAction(
                    action=AgentActionType.ACQUIRE,
                    evidence_id=self._exposed_evidence_id(self.sample.evidence_ids[best]),
                )
        target = self._ground_truth()
        if target == EditOperation.KEEP:
            return AgentAction(action=AgentActionType.REJECT)
        return AgentAction(action=AgentActionType.COMMIT, edit=target)

    def step(self, action: AgentAction) -> AgentTransition:
        if self.done:
            raise RuntimeError("cannot step a completed episode")
        before = self.observation()
        oracle = self.oracle_action()
        utility_map = {
            f"ACQUIRE:{self._exposed_evidence_id(self.sample.evidence_ids[index])}": (
                self._marginal_utility(index)
            )
            for index in self.available
            if self.sample.evidence_costs[index] <= self.remaining_budget
        }
        target = self._ground_truth()
        utility_map[
            "REJECT" if target == EditOperation.KEEP else f"COMMIT:{target.value}"
        ] = self.sample.stop_utility

        if action.action == AgentActionType.ACQUIRE:
            raw_evidence_id = self._raw_evidence_id(str(action.evidence_id))
            try:
                index = self.sample.evidence_ids.index(raw_evidence_id)
            except ValueError as exc:
                raise ValueError(f"unknown evidence_id: {action.evidence_id}") from exc
            if index not in self.available:
                raise ValueError("evidence is unavailable or already selected")
            cost = float(self.sample.evidence_costs[index])
            if cost > self.remaining_budget:
                raise ValueError("evidence cost exceeds remaining budget")
            self.available.remove(index)
            self.selected.append(self.sample.evidence_ids[index])
            self.remaining_budget -= cost
            reward = self._marginal_utility(index)
            self.current_gain = max(self.current_gain, float(self.quality_gains[index]))
            self.spent_penalty += float(self.evidence_penalties[index])
            self.step_index += 1
            self.belief = self.belief_updater.fuse(self.selected)
            next_observation = self.observation()
            done = False
        elif action.action == AgentActionType.USE_TOOL:
            if self.tool_registry is None or action.tool_call is None:
                raise ValueError("USE_TOOL requires a configured tool registry")
            expected_cost = self.tool_registry.cost(action.tool_call.tool)
            if expected_cost > self.remaining_budget:
                raise ValueError("tool cost exceeds remaining budget")
            evidence_id = action.tool_call.inputs.get("evidence_id")
            resolved_call = action.tool_call
            raw_tool_evidence_id = None
            if evidence_id is not None:
                raw_tool_evidence_id = self._raw_evidence_id(str(evidence_id))
                if self.public_identifiers:
                    resolved_call = action.tool_call.model_copy(
                        update={
                            "inputs": {
                                **action.tool_call.inputs,
                                "evidence_id": raw_tool_evidence_id,
                            }
                        }
                    )
            if self.asset_paths is not None:
                if evidence_id is None:
                    raise ValueError("agent tool calls must reference an evidence_id")
                if raw_tool_evidence_id not in self.selected:
                    raise ValueError("tools may only inspect acquired evidence")
                inputs = (
                    dict(self.tool_inputs_by_evidence.get(str(raw_tool_evidence_id), {}))
                    if self.tool_inputs_by_evidence is not None
                    else {}
                )
                inputs.update(resolved_call.inputs)
                evidence_path = self.asset_paths.get(str(raw_tool_evidence_id))
                if evidence_path is None:
                    raise ValueError(f"missing asset path for evidence: {raw_tool_evidence_id}")
                if resolved_call.tool == GeoToolName.IMAGE_QUALITY:
                    inputs["image_path"] = evidence_path
                elif resolved_call.tool == GeoToolName.TEMPORAL_CHANGE:
                    anchor_id = self.sample.metadata.get("initial_evidence_id")
                    if anchor_id is None and self.selected:
                        anchor_id = self.selected[0]
                    anchor_path = self.asset_paths.get(str(anchor_id))
                    if anchor_path is None:
                        raise ValueError("TEMPORAL_CHANGE requires the initial evidence asset")
                    inputs.update(
                        {
                            "before_path": evidence_path,
                            "after_path": anchor_path,
                            # Prevent the generic registry resolver from adding a second path.
                            "image_path": evidence_path,
                        }
                    )
                else:
                    inputs.setdefault("image_path", evidence_path)
                parameters = (
                    dict(self.tool_parameters.get(str(raw_tool_evidence_id), {}))
                    if self.tool_parameters is not None
                    else {}
                )
                parameters.update(resolved_call.parameters)
                resolved_call = resolved_call.model_copy(
                    update={"inputs": inputs, "parameters": parameters}
                )
            result = self.tool_registry.execute(
                resolved_call,
                asset_paths=self.asset_paths,
            )
            if evidence_id is not None:
                result = result.model_copy(
                    update={
                        "outputs": {
                            **result.outputs,
                            "evidence_id": self._exposed_evidence_id(
                                str(raw_tool_evidence_id)
                            ),
                        }
                    }
                )
            self.remaining_budget -= result.cost
            self.spent_penalty += float(result.cost) * self.cost_weight
            self.step_index += 1
            self.tool_history.append(result)
            if self.tool_belief_updater is not None:
                self.belief = self.tool_belief_updater.update(self.belief, result)
            reward = -float(result.cost) - (0.1 if not result.success else 0.0)
            next_observation = self.observation()
            done = False
        else:
            reward = self._terminal_reward(action)
            self.done = True
            done = True
            next_observation = None
        return AgentTransition(
            observation=before,
            action=action,
            reward=reward,
            done=done,
            next_observation=next_observation,
            oracle_action=oracle,
            oracle_utilities=utility_map,
        )


def rollout_agent_policy(
    env: MapMaintenanceEnv,
    policy: AgentPolicy,
    *,
    max_acquisitions: int | None = None,
    max_tool_calls: int | None = 4,
    max_steps: int = 8,
) -> AgentTrajectory:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if max_acquisitions is not None and max_acquisitions < 0:
        raise ValueError("max_acquisitions must be non-negative")
    if max_tool_calls is not None and max_tool_calls < 0:
        raise ValueError("max_tool_calls must be non-negative")
    observation = env.reset()
    transitions: list[AgentTransition] = []
    acquisition_count = 0
    tool_call_count = 0
    step_count = 0
    while True:
        action = policy.act(observation) if step_count < max_steps else None
        action_limit_reached = action is None or (
            action.action == AgentActionType.ACQUIRE
            and max_acquisitions is not None
            and acquisition_count >= max_acquisitions
        ) or (
            action.action == AgentActionType.USE_TOOL
            and max_tool_calls is not None
            and tool_call_count >= max_tool_calls
        )
        if action_limit_reached:
            predicted = observation.belief.predicted_edit
            action = (
                AgentAction(action=AgentActionType.REJECT)
                if predicted == EditOperation.KEEP
                else AgentAction(action=AgentActionType.COMMIT, edit=predicted)
            )
        transition = env.step(action)
        transitions.append(transition)
        step_count += 1
        if transition.done:
            break
        acquisition_count += int(action.action == AgentActionType.ACQUIRE)
        tool_call_count += int(action.action == AgentActionType.USE_TOOL)
        next_observation = transition.next_observation
        if next_observation is None:  # pragma: no cover - enforced by AgentTransition
            raise RuntimeError("non-terminal transition lost its next observation")
        observation = next_observation
    task_id = env.sample.metadata.get("source_episode", env.sample.sample_id)
    return AgentTrajectory(
        trajectory_id=f"{task_id}__agent__b{str(env.initial_budget).replace('.', 'p')}",
        task_id=str(task_id),
        split=env.sample.split,
        budget=env.initial_budget,
        transitions=transitions,
        total_reward=float(sum(item.reward for item in transitions)),
        metadata={
            "max_acquisitions": max_acquisitions,
            "max_tool_calls": max_tool_calls,
            "max_steps": max_steps,
            "acquisition_count": acquisition_count,
            "tool_call_count": tool_call_count,
        },
    )
