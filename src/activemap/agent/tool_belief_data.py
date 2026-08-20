"""Validated supervision records for grounded tool-to-belief learning."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import field_validator, model_validator

from activemap.agent.records import AgentBelief
from activemap.geo_tools.records import GeoToolName, GeoToolResult
from activemap.models import EditOperation, StrictModel


class ToolBeliefExample(StrictModel):
    schema_version: Literal["tool-belief-v1"] = "tool-belief-v1"
    record_id: str
    episode_id: str
    split: str
    evidence_id: str
    prior_belief: AgentBelief
    tool_result: GeoToolResult
    target_belief: AgentBelief
    gt_edit: EditOperation
    metadata: dict[str, Any]

    @field_validator("split")
    @classmethod
    def split_is_train_or_validation(cls, value: str) -> str:
        if value not in {"train", "val"}:
            raise ValueError("grounded tool supervision may contain only train or val")
        return value

    @model_validator(mode="after")
    def failure_target_is_observational_only(self) -> ToolBeliefExample:
        if not self.tool_result.success and self.target_belief != self.prior_belief:
            raise ValueError("failed tool results must preserve the prior belief target")
        return self


class ToolBeliefSequenceStep(StrictModel):
    evidence_id: str
    quality_result: GeoToolResult
    temporal_result: GeoToolResult
    individual_target_belief: AgentBelief
    cumulative_target_belief: AgentBelief

    @model_validator(mode="after")
    def tools_match_sequence_contract(self) -> ToolBeliefSequenceStep:
        if self.quality_result.tool != GeoToolName.IMAGE_QUALITY:
            raise ValueError("quality_result must contain IMAGE_QUALITY")
        if self.temporal_result.tool != GeoToolName.TEMPORAL_CHANGE:
            raise ValueError("temporal_result must contain TEMPORAL_CHANGE")
        if not self.quality_result.success or not self.temporal_result.success:
            raise ValueError("recurrent sequence supervision requires successful tool calls")
        return self


class ToolBeliefSequenceExample(StrictModel):
    schema_version: Literal["tool-belief-sequence-v1"] = "tool-belief-sequence-v1"
    sequence_id: str
    episode_id: str
    split: str
    initial_belief: AgentBelief
    steps: list[ToolBeliefSequenceStep]
    gt_edit: EditOperation
    metadata: dict[str, Any]

    @field_validator("split")
    @classmethod
    def split_is_train_or_validation(cls, value: str) -> str:
        if value not in {"train", "val"}:
            raise ValueError("recurrent tool supervision may contain only train or val")
        return value

    @model_validator(mode="after")
    def sequence_is_complete_and_unique(self) -> ToolBeliefSequenceExample:
        if len(self.steps) != 3:
            raise ValueError("MUNO21 recurrent supervision requires exactly three steps")
        evidence_ids = [step.evidence_id for step in self.steps]
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("sequence evidence IDs must be unique")
        if self.metadata.get("test_assets_read") is not False:
            raise ValueError("sequence metadata must certify test_assets_read=false")
        return self


class PostAcquisitionToolPairExample(StrictModel):
    """One reachable tool pair conditioned on the normal post-acquisition belief."""

    schema_version: Literal["post-acquisition-tool-pair-v1"] = (
        "post-acquisition-tool-pair-v1"
    )
    example_id: str
    task_id: str
    split: str
    evidence_id: str
    post_acquisition_belief: AgentBelief
    quality_result: GeoToolResult
    temporal_result: GeoToolResult
    semantic_result: GeoToolResult | None = None
    target_belief: AgentBelief
    gt_edit: EditOperation
    evidence_cost: float
    tool_cost: float
    metadata: dict[str, Any]

    @field_validator("split")
    @classmethod
    def split_is_train_or_validation(cls, value: str) -> str:
        if value not in {"train", "val"}:
            raise ValueError("post-acquisition tool data may contain only train or val")
        return value

    @model_validator(mode="after")
    def pair_is_grounded_and_test_free(self) -> PostAcquisitionToolPairExample:
        if self.quality_result.tool != GeoToolName.IMAGE_QUALITY:
            raise ValueError("quality_result must contain IMAGE_QUALITY")
        if self.temporal_result.tool != GeoToolName.TEMPORAL_CHANGE:
            raise ValueError("temporal_result must contain TEMPORAL_CHANGE")
        if not self.quality_result.success or not self.temporal_result.success:
            raise ValueError("post-acquisition supervision requires successful tools")
        if self.semantic_result is not None:
            if self.semantic_result.tool != GeoToolName.RASTER_SEGMENT:
                raise ValueError("semantic_result must contain RASTER_SEGMENT")
            if not self.semantic_result.success:
                raise ValueError("semantic_result must be successful")
        if self.evidence_cost <= 0.0 or self.tool_cost <= 0.0:
            raise ValueError("evidence and tool costs must be positive")
        if self.metadata.get("test_assets_read") is not False:
            raise ValueError("metadata must certify test_assets_read=false")
        return self
