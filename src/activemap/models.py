"""Strict records shared by dataset construction, policies, and evaluation."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EditOperation(str, Enum):
    KEEP = "KEEP"
    ADD = "ADD"
    DELETE = "DELETE"
    RESHAPE = "RESHAPE"


class Decision(str, Enum):
    COMMIT = "COMMIT"
    REJECT = "REJECT"
    REQUEST_MORE_EVIDENCE = "REQUEST_MORE_EVIDENCE"


class GeoJSONGeometry(StrictModel):
    type: str
    coordinates: Any

    @field_validator("type")
    @classmethod
    def supported_geometry(cls, value: str) -> str:
        supported = {"Polygon", "MultiPolygon", "LineString", "MultiLineString"}
        if value not in supported:
            raise ValueError(f"unsupported geometry type: {value}")
        return value


class EditRecord(StrictModel):
    op: EditOperation
    object_id: str | None = None
    geometry: GeoJSONGeometry | None = None

    @model_validator(mode="after")
    def operation_fields_are_consistent(self) -> EditRecord:
        if self.op in {EditOperation.KEEP, EditOperation.DELETE, EditOperation.RESHAPE}:
            if not self.object_id:
                raise ValueError(f"{self.op} requires object_id")
        if self.op in {EditOperation.ADD, EditOperation.RESHAPE} and self.geometry is None:
            raise ValueError(f"{self.op} requires geometry")
        if self.op in {EditOperation.KEEP, EditOperation.DELETE} and self.geometry is not None:
            raise ValueError(f"{self.op} must not include geometry")
        return self


class CandidateHypothesis(EditRecord):
    source: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class EvidenceItem(StrictModel):
    evidence_id: str
    timestamp: str
    region: tuple[int, int, int, int]
    scale: int = Field(ge=1)
    image_path: str
    clear_fraction: float = Field(ge=0.0, le=1.0)
    cost: float = Field(gt=0.0)
    udm_path: str | None = None
    # Optional observation aligned to the map prior.  This is distinct from
    # evidence-item imagery and is consumed only by paired-temporal updaters.
    prior_image_path: str | None = None
    prior_udm_path: str | None = None
    prior_timestamp: str | None = None

    @field_validator("region")
    @classmethod
    def region_has_positive_area(
        cls, value: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:
        x_min, y_min, x_max, y_max = value
        if x_max <= x_min or y_max <= y_min:
            raise ValueError("region must have positive width and height")
        return value


class EpisodeRecord(StrictModel):
    episode_id: str
    aoi_id: str | None = None
    anchor_timestamp: str | None = None
    split: str
    source_dataset: str
    map_before: str
    target_map: str
    prior_geometry: GeoJSONGeometry | None = None
    target_geometry: GeoJSONGeometry | None = None
    hypothesis: CandidateHypothesis
    evidence_catalog: list[EvidenceItem]
    gt_edit: EditRecord
    is_synthetic: bool
    derivation_version: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("split")
    @classmethod
    def split_is_known(cls, value: str) -> str:
        if value not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        return value

    @model_validator(mode="after")
    def evidence_ids_are_unique(self) -> EpisodeRecord:
        ids = [item.evidence_id for item in self.evidence_catalog]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence_id values must be unique within an episode")
        return self


class DecisionRecord(StrictModel):
    episode_id: str
    decision: Decision
    edit: EditRecord | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_ids: list[str] = Field(default_factory=list)
    contradiction_ids: list[str] = Field(default_factory=list)
    cost: float = Field(ge=0.0)
    topology_valid: bool | None = None
    provenance: dict[str, str]

    @model_validator(mode="after")
    def decision_has_valid_edit(self) -> DecisionRecord:
        if self.decision == Decision.COMMIT and self.edit is None:
            raise ValueError("COMMIT requires edit")
        if self.decision != Decision.COMMIT and self.edit is not None:
            raise ValueError("only COMMIT may include edit")
        return self
