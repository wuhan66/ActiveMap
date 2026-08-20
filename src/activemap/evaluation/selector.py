"""Budgeted selector evaluation shared by heuristics and learned policies."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from activemap.selector_records import SelectorSample

ScoreFunction = Callable[[SelectorSample], np.ndarray]


@dataclass(frozen=True)
class SelectorMetrics:
    method: str
    budget: float
    sample_count: int
    mean_utility: float
    mean_oracle_utility: float
    mean_regret: float
    top1_accuracy: float
    stop_accuracy: float
    mean_cost: float

    def as_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def evaluate_score_policy(
    samples: Iterable[SelectorSample],
    *,
    method: str,
    budget: float,
    score_fn: ScoreFunction,
    allow_stop: bool = True,
) -> SelectorMetrics:
    utilities: list[float] = []
    oracle_utilities: list[float] = []
    regrets: list[float] = []
    top1_hits: list[float] = []
    stop_hits: list[float] = []
    costs_used: list[float] = []

    for sample in samples:
        scores = np.asarray(score_fn(sample), dtype=np.float32)
        costs = np.asarray(sample.evidence_costs, dtype=np.float32)
        oracle = np.asarray(sample.oracle_utilities, dtype=np.float32)
        if scores.shape not in {oracle.shape, (len(oracle) + 1,)}:
            raise ValueError(f"score shape mismatch for sample {sample.sample_id}")
        stop_score = float(scores[-1]) if len(scores) == len(oracle) + 1 else -np.inf
        evidence_scores = scores[: len(oracle)]
        affordable_oracle = [index for index, cost in enumerate(costs) if cost <= budget]
        best_policy_index = (
            max(affordable_oracle, key=lambda index: float(evidence_scores[index]))
            if affordable_oracle
            else None
        )
        policy_stopped = allow_stop and (
            best_policy_index is None or stop_score >= float(evidence_scores[best_policy_index])
        )
        selected = [] if policy_stopped or best_policy_index is None else [best_policy_index]
        oracle_value = sample.stop_utility if allow_stop else -np.inf
        if affordable_oracle:
            oracle_value = max(oracle_value, float(np.max(oracle[affordable_oracle])))

        best_oracle_index = (
            max(affordable_oracle, key=lambda index: float(oracle[index]))
            if affordable_oracle
            else -1
        )
        should_stop = allow_stop and (
            best_oracle_index == -1
            or sample.stop_utility >= float(oracle[best_oracle_index])
        )
        if selected:
            policy_value = float(np.max(oracle[selected]))
            chosen_index = selected[0]
        else:
            policy_value = sample.stop_utility
            chosen_index = len(oracle) if policy_stopped else -1
        used = float(sum(costs[index] for index in selected))

        utilities.append(policy_value)
        oracle_utilities.append(oracle_value)
        regrets.append(oracle_value - policy_value)
        oracle_action_index = len(oracle) if should_stop else best_oracle_index
        top1_hits.append(float(chosen_index == oracle_action_index))
        stop_hits.append(float(policy_stopped == should_stop))
        costs_used.append(used)

    if not utilities:
        raise ValueError("evaluation requires at least one sample")
    return SelectorMetrics(
        method=method,
        budget=budget,
        sample_count=len(utilities),
        mean_utility=float(np.mean(utilities)),
        mean_oracle_utility=float(np.mean(oracle_utilities)),
        mean_regret=float(np.mean(regrets)),
        top1_accuracy=float(np.mean(top1_hits)),
        stop_accuracy=float(np.mean(stop_hits)),
        mean_cost=float(np.mean(costs_used)),
    )


def initial_states_for_budget(
    samples: Sequence[SelectorSample], budget: float
) -> list[SelectorSample]:
    """Select one expanded rollout start per episode for the requested budget."""

    has_expanded_states = any("budget" in sample.metadata for sample in samples)
    if not has_expanded_states:
        return list(samples)
    selected = [
        sample
        for sample in samples
        if int(sample.metadata.get("oracle_step", -1)) == 0
        and abs(float(sample.metadata.get("budget", -1.0)) - budget) <= 1e-6
    ]
    if not selected:
        raise ValueError(f"no initial selector states found for budget {budget:g}")
    source_ids = [
        str(sample.metadata.get("source_episode", sample.sample_id)) for sample in selected
    ]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError(f"duplicate initial selector states found for budget {budget:g}")
    return selected


def metrics_by_edit_type(
    samples: Iterable[SelectorSample],
    *,
    method: str,
    budget: float,
    score_fn: ScoreFunction,
    allow_stop: bool = True,
) -> dict[str, SelectorMetrics]:
    grouped: dict[str, list[SelectorSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.edit_type.value].append(sample)
    return {
        edit_type: evaluate_score_policy(
            group,
            method=method,
            budget=budget,
            score_fn=score_fn,
            allow_stop=allow_stop,
        )
        for edit_type, group in sorted(grouped.items())
    }
