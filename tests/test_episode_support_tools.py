from __future__ import annotations

from pathlib import Path

from activemap.models import (
    CandidateHypothesis,
    EditOperation,
    EditRecord,
    EpisodeRecord,
    EvidenceItem,
    GeoJSONGeometry,
)
from scripts.audit_episode_support import audit_episode_support
from scripts.subsample_episodes import select_rows


def _episode(split: str, operation: EditOperation, aoi: str, index: int) -> EpisodeRecord:
    geometry = GeoJSONGeometry(
        type="Polygon",
        coordinates=[[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
    )
    object_id = None if operation == EditOperation.ADD else f"object-{index}"
    edit_geometry = geometry if operation in {EditOperation.ADD, EditOperation.RESHAPE} else None
    return EpisodeRecord(
        episode_id=f"{split}-{operation.value}-{aoi}-{index}",
        aoi_id=aoi,
        split=split,
        source_dataset="fixture",
        map_before="/unused/before.geojson",
        target_map="/unused/after.geojson",
        hypothesis=CandidateHypothesis(
            op=operation,
            object_id=object_id,
            geometry=edit_geometry,
            source="fixture",
        ),
        gt_edit=EditRecord(
            op=operation, object_id=object_id, geometry=edit_geometry
        ),
        evidence_catalog=[
            EvidenceItem(
                evidence_id=f"e-{index}",
                timestamp="2026_01",
                region=(0, 0, 16, 16),
                scale=1,
                image_path="/unused/image.tif",
                clear_fraction=1.0,
                cost=1.0,
            )
        ],
        is_synthetic=False,
        derivation_version="fixture-v1",
    )


def test_audit_and_subset_preserve_geographic_isolation(tmp_path: Path) -> None:
    rows = []
    for split, aois in (("train", ("a", "b")), ("val", ("c", "d"))):
        for operation in EditOperation:
            for index in range(4):
                rows.append(_episode(split, operation, aois[index % 2], index))
    source = tmp_path / "episodes.jsonl"
    source.write_text(
        "".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8"
    )

    audit = audit_episode_support(rows, source)
    assert audit["passed"] is True
    assert audit["aoi_overlap"] == {"train:val": 0}
    assert audit["test_assets_read"] is False

    first = select_rows(rows, per_operation=3, seed=7)
    second = select_rows(list(reversed(rows)), per_operation=3, seed=7)
    assert [row.episode_id for row in first] == [row.episode_id for row in second]
    assert len(first) == 2 * len(EditOperation) * 3
    counts = {}
    for row in first:
        key = (row.split, row.gt_edit.op.value)
        counts[key] = counts.get(key, 0) + 1
    assert set(counts.values()) == {3}


def test_audit_detects_aoi_overlap(tmp_path: Path) -> None:
    rows = [
        _episode("train", EditOperation.KEEP, "shared", 1),
        _episode("val", EditOperation.KEEP, "shared", 2),
    ]
    source = tmp_path / "episodes.jsonl"
    source.write_text("\n".join(row.model_dump_json() for row in rows), encoding="utf-8")
    summary = audit_episode_support(rows, source)
    assert summary["passed"] is False
    assert summary["aoi_overlap"]["train:val"] == 1
