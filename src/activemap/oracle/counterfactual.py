"""Ground-truth-only oracle used to supervise evidence policies offline."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

QualityFunction = Callable[[tuple[int, ...]], float]
RiskFunction = Callable[[tuple[int, ...], int], float]


@dataclass(frozen=True)
class ActionUtility:
    action_index: int
    quality_before: float
    quality_after: float
    gain: float
    cost: float
    false_edit_risk: float
    utility: float


@dataclass(frozen=True)
class OracleStep:
    selected_before: tuple[int, ...]
    action_index: int | None
    utility: float
    cumulative_cost: float
    stopped: bool


def counterfactual_action_utilities(
    *,
    selected: Sequence[int],
    candidate_count: int,
    costs: Sequence[float],
    quality_fn: QualityFunction,
    risk_fn: RiskFunction | None = None,
    cost_weight: float = 0.18,
    false_edit_weight: float = 0.35,
) -> list[ActionUtility]:
    if len(costs) != candidate_count:
        raise ValueError("cost count must match candidate_count")
    selected_tuple = tuple(sorted(set(selected)))
    quality_before = float(quality_fn(selected_tuple))
    results: list[ActionUtility] = []
    for action_index in range(candidate_count):
        if action_index in selected_tuple:
            continue
        selected_after = tuple(sorted((*selected_tuple, action_index)))
        quality_after = float(quality_fn(selected_after))
        gain = quality_after - quality_before
        risk = float(risk_fn(selected_tuple, action_index)) if risk_fn else 0.0
        utility = gain - cost_weight * float(costs[action_index]) - false_edit_weight * risk
        results.append(
            ActionUtility(
                action_index=action_index,
                quality_before=quality_before,
                quality_after=quality_after,
                gain=gain,
                cost=float(costs[action_index]),
                false_edit_risk=risk,
                utility=utility,
            )
        )
    return results


def greedy_oracle_rollout(
    *,
    candidate_count: int,
    costs: Sequence[float],
    budget: float,
    quality_fn: QualityFunction,
    risk_fn: RiskFunction | None = None,
    cost_weight: float = 0.18,
    false_edit_weight: float = 0.35,
    stop_margin: float = 0.0,
) -> list[OracleStep]:
    selected: list[int] = []
    cumulative_cost = 0.0
    trajectory: list[OracleStep] = []
    while True:
        utilities = counterfactual_action_utilities(
            selected=selected,
            candidate_count=candidate_count,
            costs=costs,
            quality_fn=quality_fn,
            risk_fn=risk_fn,
            cost_weight=cost_weight,
            false_edit_weight=false_edit_weight,
        )
        affordable = [result for result in utilities if cumulative_cost + result.cost <= budget]
        if not affordable:
            trajectory.append(OracleStep(tuple(selected), None, stop_margin, cumulative_cost, True))
            break
        best = max(affordable, key=lambda result: (result.utility, -result.action_index))
        if best.utility <= stop_margin:
            trajectory.append(
                OracleStep(tuple(selected), None, best.utility, cumulative_cost, True)
            )
            break
        trajectory.append(
            OracleStep(tuple(selected), best.action_index, best.utility, cumulative_cost, False)
        )
        selected.append(best.action_index)
        cumulative_cost += best.cost
    return trajectory
