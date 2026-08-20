"""Assemble validated ActiveMap episodes and write deterministic JSONL."""

from __future__ import annotations

import json
from pathlib import Path

from shapely.geometry import mapping

from activemap.data.edits import EditEvent
from activemap.models import CandidateHypothesis, EpisodeRecord, EvidenceItem, GeoJSONGeometry


def event_geometry(event: EditEvent) -> GeoJSONGeometry | None:
    if event.new_geometry is None:
        return None
    return GeoJSONGeometry.model_validate(mapping(event.new_geometry))


def build_episode(
    *,
    episode_id: str,
    split: str,
    source_dataset: str,
    map_before: str,
    target_map: str,
    event: EditEvent,
    evidence_catalog: list[EvidenceItem],
    hypothesis_source: str,
    hypothesis_confidence: float | None,
    is_synthetic: bool,
    derivation_version: str,
    aoi_id: str | None = None,
    anchor_timestamp: str | None = None,
) -> EpisodeRecord:
    hypothesis_geometry = None
    if event.op.value in {"ADD", "RESHAPE"}:
        hypothesis_geometry = event_geometry(event)
    hypothesis = CandidateHypothesis(
        op=event.op,
        object_id=event.object_id,
        geometry=hypothesis_geometry,
        source=hypothesis_source,
        confidence=hypothesis_confidence,
    )
    return EpisodeRecord(
        episode_id=episode_id,
        aoi_id=aoi_id,
        anchor_timestamp=anchor_timestamp,
        split=split,
        source_dataset=source_dataset,
        map_before=map_before,
        target_map=target_map,
        prior_geometry=(
            GeoJSONGeometry.model_validate(mapping(event.old_geometry))
            if event.old_geometry is not None
            else None
        ),
        target_geometry=(
            GeoJSONGeometry.model_validate(mapping(event.new_geometry))
            if event.new_geometry is not None
            else None
        ),
        hypothesis=hypothesis,
        evidence_catalog=evidence_catalog,
        gt_edit=event.to_edit_record(),
        is_synthetic=is_synthetic,
        derivation_version=derivation_version,
    )


def write_episodes(records: list[EpisodeRecord], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in sorted(records, key=lambda item: item.episode_id):
            payload = record.model_dump(mode="json", exclude_none=True)
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
