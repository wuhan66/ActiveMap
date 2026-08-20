"""Validated feature records used by oracle imitation and selector training."""

from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator, model_validator

from activemap.features import EVIDENCE_DIM, HYPOTHESIS_DIM, STATE_DIM
from activemap.models import EditOperation, StrictModel


class SelectorSample(StrictModel):
    sample_id: str
    split: str
    edit_type: EditOperation
    hypothesis_features: list[float]
    state_features: list[float]
    evidence_ids: list[str]
    evidence_features: list[list[float]]
    evidence_costs: list[float]
    false_edit_risks: list[float]
    oracle_utilities: list[float]
    stop_utility: float = 0.0
    false_edit_penalty_weight: float = Field(default=0.35, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("split")
    @classmethod
    def known_split(cls, value: str) -> str:
        if value not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        return value

    @model_validator(mode="after")
    def dimensions_are_consistent(self) -> SelectorSample:
        if len(self.hypothesis_features) != HYPOTHESIS_DIM:
            raise ValueError(f"hypothesis_features must have {HYPOTHESIS_DIM} values")
        if len(self.state_features) != STATE_DIM:
            raise ValueError(f"state_features must have {STATE_DIM} values")
        candidate_count = len(self.evidence_ids)
        vectors = {
            "evidence_features": len(self.evidence_features),
            "evidence_costs": len(self.evidence_costs),
            "false_edit_risks": len(self.false_edit_risks),
            "oracle_utilities": len(self.oracle_utilities),
        }
        terminal_only = bool(self.metadata.get("runtime_terminal_only", False))
        if not candidate_count and not terminal_only:
            raise ValueError("at least one evidence candidate is required")
        for name, length in vectors.items():
            if length != candidate_count:
                raise ValueError(f"{name} length must match evidence_ids")
        if len(set(self.evidence_ids)) != candidate_count:
            raise ValueError("evidence_ids must be unique")
        if any(len(features) != EVIDENCE_DIM for features in self.evidence_features):
            raise ValueError(f"every evidence feature vector must have {EVIDENCE_DIM} values")
        if any(cost <= 0 for cost in self.evidence_costs):
            raise ValueError("evidence costs must be positive")
        if any(not 0 <= risk <= 1 for risk in self.false_edit_risks):
            raise ValueError("false edit risks must be between zero and one")
        return self

    def target_index(self, allow_stop: bool = True) -> int:
        if not self.oracle_utilities:
            raise RuntimeError("runtime terminal-only samples do not define a training target")
        best_index = max(range(len(self.oracle_utilities)), key=self.oracle_utilities.__getitem__)
        if allow_stop and self.stop_utility >= self.oracle_utilities[best_index]:
            return len(self.oracle_utilities)
        return best_index
