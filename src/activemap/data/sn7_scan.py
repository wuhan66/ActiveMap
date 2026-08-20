"""Dry-run SpaceNet 7 edit derivation without writing image crops."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from time import perf_counter

import pandas as pd

from activemap.data.manifest import write_manifest
from activemap.data.sn7_pairs import iter_snapshot_pairs


def scan_sn7_edits(
    manifest: pd.DataFrame,
    output: Path,
    *,
    aoi_ids: set[str] | None = None,
    id_column: str | None = None,
    keep_iou_min: float = 0.80,
    fallback_match_iou_min: float = 0.20,
    max_centroid_distance: float | None = None,
    min_area: float = 0.0,
    max_events_per_operation: int | None = None,
    max_month_gap: int | None = None,
    min_change_persistence: int = 1,
    sampling_seed: int = 20260710,
) -> dict[str, object]:
    """Write pair-level event counts and return aggregate timing statistics."""

    selected = manifest
    if aoi_ids is not None:
        selected = manifest[manifest["aoi_id"].astype(str).isin(aoi_ids)].copy()
        missing = aoi_ids - set(selected["aoi_id"].astype(str))
        if missing:
            raise ValueError(f"AOIs not found in manifest: {sorted(missing)}")

    started = perf_counter()
    rows: list[dict[str, object]] = []
    operations: Counter[str] = Counter()
    for pair in iter_snapshot_pairs(
        selected,
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
        pair_counts = Counter(event.op.value for event in pair.events)
        operations.update(pair_counts)
        rows.append(
            {
                "aoi_id": pair.aoi_id,
                "split": pair.split,
                "old_timestamp": pair.old_timestamp,
                "new_timestamp": pair.new_timestamp,
                "month_gap": pair.month_gap,
                "events": len(pair.events),
                "keep": pair_counts["KEEP"],
                "add": pair_counts["ADD"],
                "delete": pair_counts["DELETE"],
                "reshape": pair_counts["RESHAPE"],
            }
        )

    columns = [
        "aoi_id",
        "split",
        "old_timestamp",
        "new_timestamp",
        "month_gap",
        "events",
        "keep",
        "add",
        "delete",
        "reshape",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    write_manifest(frame, output)
    summary: dict[str, object] = {
        "rows": len(selected),
        "aoi_count": int(selected["aoi_id"].nunique()),
        "snapshot_pairs": len(frame),
        "events": int(sum(operations.values())),
        "operations": dict(sorted(operations.items())),
        "elapsed_seconds": perf_counter() - started,
        "max_month_gap": max_month_gap,
        "min_change_persistence": min_change_persistence,
        "max_events_per_operation": max_events_per_operation,
        "sampling_seed": sampling_seed,
        "output": str(output.resolve()),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
