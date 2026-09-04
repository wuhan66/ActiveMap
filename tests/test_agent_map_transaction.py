import geopandas as gpd
import pytest
from shapely.geometry import LineString

from activemap.agent.map_transaction import (
    EditableMapTransaction,
    EditProposal,
    TransactionState,
    VerificationDecision,
)
from activemap.models import EditOperation, EditRecord, GeoJSONGeometry


def _roads() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"object_id": ["road-1", "road-2"]},
        geometry=[
            LineString([(0, 0), (2, 0)]),
            LineString([(5, 0), (7, 0)]),
        ],
        crs="EPSG:3857",
    )


def _line(coordinates: list[list[float]]) -> GeoJSONGeometry:
    return GeoJSONGeometry(type="LineString", coordinates=coordinates)


def test_verified_line_edit_commits_and_rolls_back() -> None:
    original = _roads()
    proposal = EditProposal(
        proposal_id="proposal-1",
        edit=EditRecord(
            op=EditOperation.RESHAPE,
            object_id="road-1",
            geometry=_line([[0, 0], [2, 1], [3, 1]]),
        ),
        confidence=0.9,
        evidence_ids=["evidence-1"],
    )
    transaction = EditableMapTransaction(original, proposal, min_confidence=0.7)

    report = transaction.verify()
    committed = transaction.commit()

    assert report.decision == VerificationDecision.APPROVE
    assert transaction.state == TransactionState.COMMITTED
    assert not committed.loc[0].geometry.equals(original.loc[0].geometry)
    assert transaction.audit_record()["objects_after"] == 2

    restored = transaction.rollback()
    assert transaction.state == TransactionState.ROLLED_BACK
    assert restored.geometry.equals(original.geometry)


def test_duplicate_add_is_rejected_and_cannot_commit() -> None:
    proposal = EditProposal(
        proposal_id="proposal-2",
        edit=EditRecord(
            op=EditOperation.ADD,
            object_id="road-1",
            geometry=_line([[10, 0], [12, 0]]),
        ),
        confidence=0.9,
    )
    transaction = EditableMapTransaction(_roads(), proposal)

    report = transaction.verify()

    assert report.decision == VerificationDecision.REJECT
    assert "object_id_already_exists" in report.reasons
    with pytest.raises(RuntimeError, match="approved verified"):
        transaction.commit()


def test_topology_conflict_requires_revision_before_commit() -> None:
    proposal = EditProposal(
        proposal_id="proposal-3",
        edit=EditRecord(
            op=EditOperation.ADD,
            object_id="road-3",
            geometry=_line([[0, 0], [2, 0]]),
        ),
        confidence=0.9,
    )
    transaction = EditableMapTransaction(_roads(), proposal)

    first = transaction.verify()
    assert first.decision == VerificationDecision.REVISE
    assert "topology_conflict" in first.reasons

    transaction.revise(
        proposal.model_copy(
            update={
                "edit": EditRecord(
                    op=EditOperation.ADD,
                    object_id="road-3",
                    geometry=_line([[10, 0], [12, 0]]),
                )
            }
        )
    )
    second = transaction.verify()
    committed = transaction.commit()

    assert second.decision == VerificationDecision.APPROVE
    assert len(committed) == 3
    assert transaction.audit_record()["attempt_count"] == 2


def test_low_confidence_proposal_is_rejected() -> None:
    transaction = EditableMapTransaction(
        _roads(),
        EditProposal(
            proposal_id="proposal-4",
            edit=EditRecord(op=EditOperation.DELETE, object_id="road-1"),
            confidence=0.2,
        ),
        min_confidence=0.7,
    )

    report = transaction.verify()

    assert report.decision == VerificationDecision.REJECT
    assert report.reasons == ["confidence_below_threshold"]
