"""Auditable propose, verify, commit, revise, and rollback for editable maps."""

from __future__ import annotations

from enum import Enum
from typing import Any

import geopandas as gpd
from pydantic import Field, model_validator
from shapely import make_valid
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from activemap.models import EditOperation, EditRecord, StrictModel
from activemap.vector_map import apply_edit, topology_is_valid


class VerificationDecision(str, Enum):
    APPROVE = "APPROVE"
    REVISE = "REVISE"
    REJECT = "REJECT"


class TransactionState(str, Enum):
    PROPOSED = "PROPOSED"
    VERIFIED = "VERIFIED"
    REVISION_REQUIRED = "REVISION_REQUIRED"
    REJECTED = "REJECTED"
    COMMITTED = "COMMITTED"
    ROLLED_BACK = "ROLLED_BACK"


class EditProposal(StrictModel):
    proposal_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    edit: EditRecord
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_ids: list[str] = Field(default_factory=list)
    tool_call_ids: list[str] = Field(default_factory=list)
    provenance: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def proposal_is_executable(self) -> EditProposal:
        if self.edit.op == EditOperation.KEEP:
            raise ValueError("KEEP is a rejection, not an editable-map proposal")
        if self.edit.op == EditOperation.ADD and not self.edit.object_id:
            raise ValueError("transactional ADD requires an explicit object_id")
        return self


class VerificationReport(StrictModel):
    proposal_id: str
    decision: VerificationDecision
    checks: dict[str, bool]
    reasons: list[str]
    topology_valid: bool | None = None


def _family(geometry: BaseGeometry) -> str | None:
    if geometry.geom_type in {"LineString", "MultiLineString"}:
        return "linear"
    if geometry.geom_type in {"Polygon", "MultiPolygon"}:
        return "polygonal"
    return None


def _map_family(frame: gpd.GeoDataFrame) -> str | None:
    families = {
        family
        for geometry in frame.geometry
        if geometry is not None and not geometry.is_empty
        for family in [_family(geometry)]
        if family is not None
    }
    return next(iter(families)) if len(families) == 1 else None


class EditableMapTransaction:
    """One isolated proposal transaction; the caller decides whether to retain its output."""

    def __init__(
        self,
        frame: gpd.GeoDataFrame,
        proposal: EditProposal,
        *,
        id_column: str = "object_id",
        min_confidence: float = 0.5,
        max_overlap_ratio: float = 0.25,
    ) -> None:
        if id_column not in frame.columns:
            raise ValueError(f"map is missing ID column: {id_column}")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between zero and one")
        if not 0.0 <= max_overlap_ratio <= 1.0:
            raise ValueError("max_overlap_ratio must be between zero and one")
        self._original = frame.copy(deep=True)
        self._current = frame.copy(deep=True)
        self.proposal = proposal
        self.id_column = id_column
        self.min_confidence = min_confidence
        self.max_overlap_ratio = max_overlap_ratio
        self.state = TransactionState.PROPOSED
        self.report: VerificationReport | None = None
        self.attempts: list[dict[str, Any]] = []

    def _matching(self) -> Any:
        object_id = self.proposal.edit.object_id
        return self._current[self.id_column].astype(str) == str(object_id)

    def verify(self) -> VerificationReport:
        if self.state != TransactionState.PROPOSED:
            raise RuntimeError(f"cannot verify transaction in state {self.state.value}")
        edit = self.proposal.edit
        matching = self._matching()
        checks = {
            "confidence": self.proposal.confidence >= self.min_confidence,
            "target_exists_once": True,
            "id_available": True,
            "geometry_present": True,
            "geometry_valid": True,
            "geometry_family": True,
            "topology": True,
        }
        hard_reasons = []
        revise_reasons = []

        if not checks["confidence"]:
            hard_reasons.append("confidence_below_threshold")
        if edit.op in {EditOperation.DELETE, EditOperation.RESHAPE}:
            checks["target_exists_once"] = int(matching.sum()) == 1
            if not checks["target_exists_once"]:
                hard_reasons.append("target_object_not_unique")
        if edit.op == EditOperation.ADD:
            checks["id_available"] = not bool(matching.any())
            if not checks["id_available"]:
                hard_reasons.append("object_id_already_exists")

        topology_valid: bool | None = None
        if edit.op in {EditOperation.ADD, EditOperation.RESHAPE}:
            checks["geometry_present"] = edit.geometry is not None
            geometry = None
            if edit.geometry is not None:
                try:
                    geometry = make_valid(shape(edit.geometry.model_dump(mode="json")))
                except Exception:
                    checks["geometry_valid"] = False
            if geometry is None or geometry.is_empty or not geometry.is_valid:
                checks["geometry_valid"] = False
                revise_reasons.append("invalid_geometry")
            else:
                map_family = _map_family(self._current)
                checks["geometry_family"] = (
                    map_family is None or _family(geometry) == map_family
                )
                if not checks["geometry_family"]:
                    revise_reasons.append("geometry_family_mismatch")
                other_geometries = [
                    item
                    for index, item in enumerate(self._current.geometry)
                    if not (edit.op == EditOperation.RESHAPE and bool(matching.iloc[index]))
                ]
                topology_valid = topology_is_valid(
                    geometry,
                    other_geometries=other_geometries,
                    max_overlap_ratio=self.max_overlap_ratio,
                )
                checks["topology"] = topology_valid
                if not topology_valid:
                    revise_reasons.append("topology_conflict")
        if not checks["geometry_present"]:
            revise_reasons.append("missing_geometry")

        if hard_reasons:
            decision = VerificationDecision.REJECT
            self.state = TransactionState.REJECTED
        elif revise_reasons:
            decision = VerificationDecision.REVISE
            self.state = TransactionState.REVISION_REQUIRED
        else:
            decision = VerificationDecision.APPROVE
            self.state = TransactionState.VERIFIED
        self.report = VerificationReport(
            proposal_id=self.proposal.proposal_id,
            decision=decision,
            checks=checks,
            reasons=hard_reasons + revise_reasons,
            topology_valid=topology_valid,
        )
        self.attempts.append(
            {
                "proposal": self.proposal.model_dump(mode="json"),
                "verification": self.report.model_dump(mode="json"),
            }
        )
        return self.report

    def revise(self, proposal: EditProposal) -> None:
        if self.state != TransactionState.REVISION_REQUIRED:
            raise RuntimeError(f"cannot revise transaction in state {self.state.value}")
        if proposal.proposal_id != self.proposal.proposal_id:
            raise ValueError("revision must retain proposal_id")
        self.proposal = proposal
        self.report = None
        self.state = TransactionState.PROPOSED

    def commit(self) -> gpd.GeoDataFrame:
        if self.state != TransactionState.VERIFIED or self.report is None:
            raise RuntimeError("only an approved verified proposal may commit")
        self._current = apply_edit(
            self._current, self.proposal.edit, id_column=self.id_column
        )
        self.state = TransactionState.COMMITTED
        return self._current.copy(deep=True)

    def rollback(self) -> gpd.GeoDataFrame:
        if self.state != TransactionState.COMMITTED:
            raise RuntimeError("only a committed transaction may roll back")
        self._current = self._original.copy(deep=True)
        self.state = TransactionState.ROLLED_BACK
        return self._current.copy(deep=True)

    def audit_record(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal.proposal_id,
            "state": self.state.value,
            "attempt_count": len(self.attempts),
            "attempts": self.attempts,
            "objects_before": len(self._original),
            "objects_after": len(self._current),
        }
