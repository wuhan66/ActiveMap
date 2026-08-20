"""Deterministic baseline scores under a shared evidence budget."""

from __future__ import annotations

import hashlib
from enum import Enum

import numpy as np

from activemap.selector_records import SelectorSample


class BaselineName(str, Enum):
    ALWAYS_STOP = "always_stop"
    RANDOM = "random"
    QUALITY = "quality"
    UNCERTAINTY = "uncertainty"
    MAPEX = "mapex"
    CHEAPEST = "cheapest"


def baseline_scores(sample: SelectorSample, baseline: BaselineName, seed: int = 0) -> np.ndarray:
    evidence = np.asarray(sample.evidence_features, dtype=np.float32)
    costs = np.asarray(sample.evidence_costs, dtype=np.float32)
    if baseline == BaselineName.ALWAYS_STOP:
        return np.concatenate(
            [np.full(len(costs), -np.inf, dtype=np.float32), np.zeros(1, dtype=np.float32)]
        )
    if baseline == BaselineName.QUALITY:
        return evidence[:, 0]
    if baseline == BaselineName.UNCERTAINTY:
        return evidence[:, 11]
    if baseline == BaselineName.MAPEX:
        quality = evidence[:, 0]
        coverage = evidence[:, 9]
        uncertainty = evidence[:, 11]
        return quality * coverage * uncertainty
    if baseline == BaselineName.CHEAPEST:
        return -costs
    if baseline == BaselineName.RANDOM:
        digest = hashlib.sha1(f"{sample.sample_id}|{seed}".encode()).digest()
        local_seed = int.from_bytes(digest[:8], "little")
        return np.random.default_rng(local_seed).random(len(costs)).astype(np.float32)
    raise ValueError(f"unsupported baseline: {baseline}")


def rank_affordable_actions(
    scores: np.ndarray,
    costs: np.ndarray,
    budget: float,
) -> list[int]:
    ranking = np.argsort(-scores, kind="stable")
    selected: list[int] = []
    spent = 0.0
    for index in ranking:
        action_cost = float(costs[index])
        if spent + action_cost <= budget:
            selected.append(int(index))
            spent += action_cost
    return selected
