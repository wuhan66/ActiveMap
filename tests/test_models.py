import pytest
from pydantic import ValidationError

from activemap.models import DecisionRecord, EditRecord, EpisodeRecord

POLYGON = {
    "type": "Polygon",
    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
}


def test_edit_operation_constraints() -> None:
    record = EditRecord.model_validate({"op": "RESHAPE", "object_id": "obj-1", "geometry": POLYGON})
    assert record.object_id == "obj-1"

    with pytest.raises(ValidationError):
        EditRecord.model_validate({"op": "ADD"})
    with pytest.raises(ValidationError):
        EditRecord.model_validate({"op": "DELETE", "object_id": "obj-1", "geometry": POLYGON})


def test_episode_rejects_duplicate_evidence_ids() -> None:
    evidence = {
        "evidence_id": "ev-1",
        "timestamp": "2019_01",
        "region": [0, 0, 32, 32],
        "scale": 1,
        "image_path": "image.tif",
        "clear_fraction": 1.0,
        "cost": 1.0,
    }
    payload = {
        "episode_id": "episode-1",
        "split": "train",
        "source_dataset": "spacenet7",
        "map_before": "before.geojson",
        "target_map": "target.geojson",
        "hypothesis": {
            "op": "RESHAPE",
            "object_id": "obj-1",
            "geometry": POLYGON,
            "source": "test",
        },
        "evidence_catalog": [evidence, evidence],
        "gt_edit": {"op": "RESHAPE", "object_id": "obj-1", "geometry": POLYGON},
        "is_synthetic": False,
        "derivation_version": "test-v1",
    }
    with pytest.raises(ValidationError, match="evidence_id"):
        EpisodeRecord.model_validate(payload)


def test_only_commit_may_carry_edit() -> None:
    with pytest.raises(ValidationError):
        DecisionRecord.model_validate(
            {
                "episode_id": "episode-1",
                "decision": "REJECT",
                "edit": {"op": "KEEP", "object_id": "obj-1"},
                "confidence": 0.8,
                "cost": 0,
                "provenance": {"policy": "test"},
            }
        )
