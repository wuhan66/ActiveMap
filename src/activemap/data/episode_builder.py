"""Build finite ActiveMap evidence episodes from adjacent SpaceNet 7 snapshots."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from activemap.data.episodes import build_episode
from activemap.data.evidence import build_evidence_catalog
from activemap.data.progress import write_progress
from activemap.data.sn7_pairs import iter_snapshot_pairs


def build_sn7_episodes(
    manifest: pd.DataFrame,
    output: Path,
    *,
    source_dataset: str = "SpaceNet7",
    id_column: str | None = None,
    keep_iou_min: float = 0.80,
    fallback_match_iou_min: float = 0.20,
    max_centroid_distance: float | None = None,
    min_area: float = 0.0,
    max_events_per_operation: int | None = None,
    max_month_gap: int | None = None,
    min_change_persistence: int = 1,
    sampling_seed: int = 20260710,
    context_units: tuple[float, ...] = (0.0, 32.0, 96.0),
    scales: tuple[int, ...] = (1, 2, 4),
    derivation_version: str = "sn7-adjacent-v3-distance-gated",
) -> dict[str, object]:
    if len(context_units) != len(scales):
        raise ValueError("context_units and scales must have equal lengths")
    operation_counts: Counter[str] = Counter()
    pair_count = 0
    episode_count = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.name}.tmp")
    progress_path = output.with_suffix(".progress.json")
    write_progress(
        progress_path,
        {"status": "running", "snapshot_pairs": 0, "episodes": 0},
    )
    with temporary_output.open("w", encoding="utf-8") as handle:
        for pair in iter_snapshot_pairs(
            manifest,
            id_column=id_column,
            keep_iou_min=keep_iou_min,
            fallback_match_iou_min=fallback_match_iou_min,
            max_centroid_distance=max_centroid_distance,
            min_area=min_area,
            max_events_per_operation=max_events_per_operation,
            max_month_gap=max_month_gap,
            min_change_persistence=min_change_persistence,
            sampling_seed=sampling_seed,
        ):
            pair_count += 1
            for event_index, event in enumerate(pair.events):
                reference = (
                    event.new_geometry
                    if event.new_geometry is not None
                    else event.old_geometry
                )
                if reference is None or reference.is_empty:
                    continue
                evidence = build_evidence_catalog(
                    manifest,
                    aoi_id=pair.aoi_id,
                    anchor_timestamp=pair.new_timestamp,
                    geometry=reference,
                    context_meters=context_units,
                    scales=scales,
                )
                evidence = [
                    item.model_copy(
                        update={
                            "prior_image_path": str(pair.old_image_path),
                            "prior_udm_path": (
                                str(pair.old_udm_path)
                                if pair.old_udm_path is not None
                                else None
                            ),
                            "prior_timestamp": pair.old_timestamp,
                        }
                    )
                    for item in evidence
                ]
                identity = (
                    f"{pair.aoi_id}|{pair.old_timestamp}|{pair.new_timestamp}|"
                    f"{event.op.value}|{event.object_id}|{event_index}"
                )
                digest = hashlib.sha1(identity.encode()).hexdigest()[:16]
                confidence = event.iou if event.iou is not None else 1.0
                episode = build_episode(
                    episode_id=f"sn7-{digest}",
                    split=pair.split,
                    source_dataset=source_dataset,
                    map_before=str(pair.old_label_path),
                    target_map=str(pair.new_label_path),
                    event=event,
                    evidence_catalog=evidence,
                    hypothesis_source="adjacent_snapshot_derivation",
                    hypothesis_confidence=float(confidence),
                    is_synthetic=False,
                    derivation_version=derivation_version,
                    aoi_id=pair.aoi_id,
                    anchor_timestamp=pair.new_timestamp,
                )
                payload = episode.model_dump(mode="json", exclude_none=True)
                handle.write(
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                episode_count += 1
                operation_counts[event.op.value] += 1
            write_progress(
                progress_path,
                {
                    "status": "running",
                    "snapshot_pairs": pair_count,
                    "episodes": episode_count,
                    "operations": dict(sorted(operation_counts.items())),
                    "current_aoi": pair.aoi_id,
                    "current_old_timestamp": pair.old_timestamp,
                    "current_new_timestamp": pair.new_timestamp,
                },
            )
    temporary_output.replace(output)
    summary: dict[str, object] = {
        "snapshot_pairs": pair_count,
        "episodes": episode_count,
        "operations": dict(sorted(operation_counts.items())),
        "output": str(output.resolve()),
        "derivation_version": derivation_version,
        "max_month_gap": max_month_gap,
        "min_change_persistence": min_change_persistence,
        "max_events_per_operation": max_events_per_operation,
        "sampling_seed": sampling_seed,
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_progress(progress_path, {"status": "complete", **summary})
    return summary
