"""Versioned quality-cost-safety utility for complete map-update episodes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

SCHEMA_VERSION = "activemap-episode-utility-v2"


@dataclass(frozen=True)
class EpisodeUtilityProfile:
    """Predeclared trade-off weights for robustness, not model selection."""

    name: str
    cost: float
    false_edit: float
    missed_edit: float
    wrong_edit: float
    invalid_topology: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("utility profile name cannot be empty")
        weights = (
            self.cost,
            self.false_edit,
            self.missed_edit,
            self.wrong_edit,
            self.invalid_topology,
        )
        if min(weights) < 0:
            raise ValueError("utility weights must be non-negative")


UTILITY_PROFILES: dict[str, EpisodeUtilityProfile] = {
    "balanced": EpisodeUtilityProfile(
        name="balanced",
        cost=0.10,
        false_edit=0.50,
        missed_edit=0.25,
        wrong_edit=0.25,
        invalid_topology=0.25,
    ),
    "safety": EpisodeUtilityProfile(
        name="safety",
        cost=0.05,
        false_edit=1.00,
        missed_edit=0.25,
        wrong_edit=0.25,
        invalid_topology=0.50,
    ),
    "cost_aware": EpisodeUtilityProfile(
        name="cost_aware",
        cost=0.25,
        false_edit=0.50,
        missed_edit=0.25,
        wrong_edit=0.25,
        invalid_topology=0.25,
    ),
}


def utility_protocol(
    *,
    quality_source: str,
    paper_primary: bool,
    terminal_error_source: str = "terminal_operation",
) -> dict[str, Any]:
    """Describe the immutable utility contract stored beside every result."""

    return {
        "schema_version": SCHEMA_VERSION,
        "formula": (
            "(final_map_quality - prior_map_quality) "
            "- w_cost*(spent_cost/budget) - w_false_edit*I(false_edit) "
            "- w_missed_edit*I(missed_edit) - w_wrong_edit*I(wrong_edit) "
            "- w_invalid_topology*I(invalid_topology)"
        ),
        "quality_source": quality_source,
        "terminal_error_source": terminal_error_source,
        "paper_primary": paper_primary,
        "cost_counting": "spent_cost includes acquisition and tool cost exactly once",
        "profiles": {name: asdict(profile) for name, profile in UTILITY_PROFILES.items()},
        "selection_rule": (
            "balanced is primary; safety and cost_aware are predeclared robustness profiles"
        ),
    }


def episode_utility(
    *,
    final_map_quality: float,
    prior_map_quality: float,
    spent_cost: float,
    budget: float,
    false_edit: bool,
    missed_edit: bool,
    wrong_edit: bool,
    topology_valid: bool = True,
    profile: EpisodeUtilityProfile,
) -> dict[str, float]:
    """Score one episode while retaining every auditable additive component."""

    if not 0.0 <= final_map_quality <= 1.0:
        raise ValueError("final_map_quality must be between zero and one")
    if not 0.0 <= prior_map_quality <= 1.0:
        raise ValueError("prior_map_quality must be between zero and one")
    if budget <= 0.0:
        raise ValueError("budget must be positive")
    if spent_cost < 0.0:
        raise ValueError("spent_cost cannot be negative")
    if sum((bool(false_edit), bool(missed_edit), bool(wrong_edit))) > 1:
        raise ValueError("terminal error categories must be mutually exclusive")

    quality_gain = float(final_map_quality - prior_map_quality)
    normalized_cost = float(min(spent_cost / budget, 1.0))
    cost_penalty = profile.cost * normalized_cost
    false_edit_penalty = profile.false_edit * float(false_edit)
    missed_edit_penalty = profile.missed_edit * float(missed_edit)
    wrong_edit_penalty = profile.wrong_edit * float(wrong_edit)
    topology_penalty = profile.invalid_topology * float(not topology_valid)
    value = (
        quality_gain
        - cost_penalty
        - false_edit_penalty
        - missed_edit_penalty
        - wrong_edit_penalty
        - topology_penalty
    )
    return {
        "value": float(value),
        "quality_gain": quality_gain,
        "normalized_cost": normalized_cost,
        "cost_penalty": float(cost_penalty),
        "false_edit_penalty": float(false_edit_penalty),
        "missed_edit_penalty": float(missed_edit_penalty),
        "wrong_edit_penalty": float(wrong_edit_penalty),
        "invalid_topology_penalty": float(topology_penalty),
    }


def score_episode_profiles(**kwargs: Any) -> dict[str, dict[str, float]]:
    """Evaluate all predeclared profiles on identical episode outcomes."""

    return {
        name: episode_utility(profile=profile, **kwargs)
        for name, profile in UTILITY_PROFILES.items()
    }
