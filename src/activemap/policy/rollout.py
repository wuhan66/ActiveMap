"""Sequential budgeted evidence acquisition with auditable action traces."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np

from activemap.evaluation.episode_utility import UTILITY_PROFILES
from activemap.selector_records import SelectorSample

ScoreFunction = Callable[[SelectorSample], np.ndarray]


@dataclass(frozen=True)
class RolloutStep:
    step: int
    evidence_id: str
    score: float
    cost: float
    utility: float
    utility_after: float
    remaining_budget: float


@dataclass(frozen=True)
class RolloutResult:
    sample_id: str
    budget: float
    stop_reason: str
    selected_evidence_ids: tuple[str, ...]
    spent_cost: float
    utility: float
    oracle_utility: float
    regret: float
    steps: tuple[RolloutStep, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _state_after_actions(
    sample: SelectorSample,
    *,
    initial_budget: float,
    remaining_budget: float,
    selected_count: int,
    total_evidence_count: int,
    current_gain: float,
    edit_probabilities: list[float] | None = None,
) -> list[float]:
    state = list(sample.state_features)
    denominator = max(initial_budget, 1e-6)
    state[0] = float(np.clip(remaining_budget / denominator, 0.0, 1.0))
    state[1] = float(np.clip((initial_budget - remaining_budget) / denominator, 0.0, 1.0))
    if edit_probabilities is not None:
        state[2:6] = edit_probabilities
    state[6] = float(
        np.clip(selected_count / max(total_evidence_count, 1), 0.0, 1.0)
    )
    state[7] = current_gain
    return state


def rollout_score_policy(
    sample: SelectorSample,
    *,
    budget: float,
    score_fn: ScoreFunction,
    allow_stop: bool = True,
    max_steps: int | None = None,
) -> RolloutResult:
    """Repeatedly rescore remaining candidates while updating budget/history state."""

    if budget < 0:
        raise ValueError("budget must be non-negative")
    available = list(range(len(sample.evidence_ids)))
    remaining = float(budget)
    initial_gain = float(sample.state_features[7])
    utility_mode = str(sample.metadata.get("utility_mode", "proxy"))
    if utility_mode == "executable":
        profile = UTILITY_PROFILES[str(sample.metadata.get("utility_profile", "balanced"))]
        outcomes = sample.metadata.get("executable_outcomes")
        if not isinstance(outcomes, dict):
            raise ValueError("executable utility requires executable_outcomes metadata")
        penalties = (
            np.asarray(sample.evidence_costs, dtype=np.float64) * profile.cost / budget
        )
        quality_gains = np.asarray(
            [float(outcomes[item]["terminal_score_before_cost"]) for item in sample.evidence_ids],
            dtype=np.float64,
        )
    else:
        cost_weight = float(sample.metadata.get("cost_weight", 0.0))
        penalties = np.asarray(sample.evidence_costs, dtype=np.float64) * cost_weight
        penalties += (
            np.asarray(sample.false_edit_risks, dtype=np.float64)
            * sample.false_edit_penalty_weight
        )
        initial_marginals = np.asarray(sample.oracle_utilities, dtype=np.float64)
        quality_gains = initial_gain + np.maximum(initial_marginals + penalties, 0.0)
    current_gain = initial_gain
    spent_penalty = 0.0
    total_utility = float(sample.stop_utility)
    selected: list[str] = []
    selected_for_belief = list(sample.metadata.get("selected_evidence_ids", []))
    belief_updater = None
    if isinstance(sample.metadata.get("evidence_predictions"), dict):
        from activemap.agent.tools import CounterfactualBeliefUpdater

        belief_updater = CounterfactualBeliefUpdater(sample)
    trace: list[RolloutStep] = []
    limit = max_steps if max_steps is not None else len(available)
    stop_reason = "candidate_exhausted"

    while available and len(trace) < limit:
        affordable = [index for index in available if sample.evidence_costs[index] <= remaining]
        if not affordable:
            stop_reason = "budget_exhausted"
            break
        belief = (
            belief_updater.fuse(selected_for_belief)
            if belief_updater is not None and selected_for_belief
            else None
        )
        state = _state_after_actions(
            sample,
            initial_budget=budget,
            remaining_budget=remaining,
            selected_count=len(selected_for_belief),
            total_evidence_count=len(
                set(sample.evidence_ids)
                | set(sample.metadata.get("selected_evidence_ids", []))
            ),
            current_gain=current_gain,
            edit_probabilities=belief.edit_probabilities if belief is not None else None,
        )
        hypothesis = list(sample.hypothesis_features)
        if belief is not None:
            hypothesis[:4] = belief.edit_probabilities
            hypothesis[12] = belief.uncertainty
            hypothesis[13] = belief.confidence
        current = sample.model_copy(
            update={
                "state_features": state,
                "hypothesis_features": hypothesis,
                "evidence_ids": [sample.evidence_ids[index] for index in available],
                "evidence_features": [sample.evidence_features[index] for index in available],
                "evidence_costs": [sample.evidence_costs[index] for index in available],
                "false_edit_risks": [sample.false_edit_risks[index] for index in available],
                "oracle_utilities": [
                    (
                        float(quality_gains[index]) - current_gain
                        if utility_mode == "executable"
                        else max(float(quality_gains[index]) - current_gain, 0.0)
                    ) - float(penalties[index])
                    for index in available
                ],
            }
        )
        scores = np.asarray(score_fn(current), dtype=np.float32)
        if scores.shape not in {(len(available),), (len(available) + 1,)}:
            raise ValueError(f"score shape mismatch for sample {sample.sample_id}")
        stop_score = float(scores[-1]) if len(scores) == len(available) + 1 else -np.inf
        local_affordable = [available.index(index) for index in affordable]
        best_local = max(local_affordable, key=lambda index: float(scores[index]))
        if allow_stop and stop_score >= float(scores[best_local]):
            stop_reason = "policy_stop"
            break

        global_index = available.pop(best_local)
        cost = float(sample.evidence_costs[global_index])
        gain = float(quality_gains[global_index]) - current_gain
        if utility_mode != "executable":
            gain = max(gain, 0.0)
        utility = gain - float(penalties[global_index])
        remaining -= cost
        current_gain = max(current_gain, float(quality_gains[global_index]))
        spent_penalty += float(penalties[global_index])
        total_utility = current_gain - initial_gain - spent_penalty
        selected.append(sample.evidence_ids[global_index])
        selected_for_belief.append(sample.evidence_ids[global_index])
        trace.append(
            RolloutStep(
                step=len(trace) + 1,
                evidence_id=sample.evidence_ids[global_index],
                score=float(scores[best_local]),
                cost=cost,
                utility=utility,
                utility_after=total_utility,
                remaining_budget=max(remaining, 0.0),
            )
        )
    else:
        if len(trace) >= limit and available:
            stop_reason = "max_steps"

    oracle_utility = float(sample.stop_utility)
    indices = range(len(sample.evidence_ids))
    for size in range(1, len(sample.evidence_ids) + 1):
        for subset in combinations(indices, size):
            subset_cost = sum(float(sample.evidence_costs[index]) for index in subset)
            if subset_cost > budget + 1e-8:
                continue
            subset_utility = (
                max(initial_gain, *(float(quality_gains[index]) for index in subset))
                - initial_gain
                - sum(float(penalties[index]) for index in subset)
            )
            oracle_utility = max(oracle_utility, subset_utility)
    return RolloutResult(
        sample_id=sample.sample_id,
        budget=float(budget),
        stop_reason=stop_reason,
        selected_evidence_ids=tuple(selected),
        spent_cost=float(budget - remaining),
        utility=total_utility,
        oracle_utility=float(oracle_utility),
        regret=float(oracle_utility - total_utility),
        steps=tuple(trace),
    )


def evaluate_rollouts(
    samples: Iterable[SelectorSample],
    *,
    method: str,
    budget: float,
    score_fn: ScoreFunction,
    allow_stop: bool = True,
    max_steps: int | None = None,
) -> tuple[dict[str, float | int | str], list[RolloutResult]]:
    results = [
        rollout_score_policy(
            sample,
            budget=budget,
            score_fn=score_fn,
            allow_stop=allow_stop,
            max_steps=max_steps,
        )
        for sample in samples
    ]
    if not results:
        raise ValueError("rollout evaluation requires at least one sample")
    return (
        {
            "method": method,
            "budget": budget,
            "sample_count": len(results),
            "mean_utility": float(np.mean([result.utility for result in results])),
            "mean_oracle_utility": float(
                np.mean([result.oracle_utility for result in results])
            ),
            "mean_regret": float(np.mean([result.regret for result in results])),
            "mean_cost": float(np.mean([result.spent_cost for result in results])),
            "mean_steps": float(np.mean([len(result.steps) for result in results])),
            "policy_stop_rate": float(
                np.mean([result.stop_reason == "policy_stop" for result in results])
            ),
        },
        results,
    )
