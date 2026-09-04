#!/usr/bin/env python3
"""Select one delta-component filter on cached validation writeback artifacts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from affine import Affine

from activemap.agent.writeback import EvidenceMaskPrediction, evaluate_typed_writeback
from activemap.models import EditOperation


def _rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                row["source_writeback"] = str(path.resolve())
                rows.append(row)
    if not rows:
        raise ValueError("no cached writeback rows")
    return rows


def _evaluate_cached(row: dict[str, Any], min_pixels: int) -> dict[str, Any]:
    artifact_path = Path(str(row["mask_artifact"]))
    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)
    with np.load(artifact_path) as artifact:
        committed = np.asarray(artifact["committed_mask"], dtype=np.float32)
        prior = np.asarray(artifact["prior_mask"], dtype=np.float32)
        target = np.asarray(artifact["target_mask"], dtype=np.float32)
        valid = np.asarray(artifact["valid_mask"], dtype=np.float32)
        transform_values = np.asarray(artifact["transform"], dtype=np.float64).tolist()
    metrics = evaluate_typed_writeback(
        [EvidenceMaskPrediction("cached-committed", committed, 1.0)],
        operation=EditOperation(str(row["operation"])),
        prior=prior,
        target=target,
        valid=valid,
        transform=Affine(*transform_values),
        min_delta_component_pixels=min_pixels,
    )
    return {
        "task_id": str(row["task_id"]),
        "budget": float(row["budget"]),
        "source_writeback": row["source_writeback"],
        **metrics,
    }


def sweep(
    paths: list[Path],
    candidates: list[int],
    *,
    raster_iou_tolerance: float,
    topology_floor: float,
    replay_floor: float,
) -> dict[str, Any]:
    if not candidates or 0 not in candidates or any(value < 0 for value in candidates):
        raise ValueError("candidates must be non-negative and include the raw baseline 0")
    source_rows = _rows(paths)
    evaluated = {
        value: [_evaluate_cached(row, value) for row in source_rows]
        for value in sorted(set(candidates))
    }
    summaries = []
    for value, rows in evaluated.items():
        summaries.append(
            {
                "min_delta_component_pixels": value,
                "sample_count": len(rows),
                "mean_raster_iou": float(np.mean([row["raster_iou"] for row in rows])),
                "mean_raster_iou_gain": float(
                    np.mean([row["raster_iou_gain"] for row in rows])
                ),
                "mean_component_count_absolute_error": float(
                    np.mean([row["component_count_absolute_error"] for row in rows])
                ),
                "mean_vector_replay_iou": float(
                    np.mean([row["vector_replay_iou"] for row in rows])
                ),
                "vector_delta_topology_valid_rate": float(
                    np.mean([row["vector_delta_topology_valid"] for row in rows])
                ),
            }
        )
    baseline = next(
        row for row in summaries if row["min_delta_component_pixels"] == 0
    )
    for row in summaries:
        row["raster_iou_delta_vs_raw"] = (
            row["mean_raster_iou"] - baseline["mean_raster_iou"]
        )
        row["component_error_delta_vs_raw"] = (
            row["mean_component_count_absolute_error"]
            - baseline["mean_component_count_absolute_error"]
        )
        row["feasible"] = bool(
            row["mean_raster_iou"]
            >= baseline["mean_raster_iou"] - raster_iou_tolerance
            and row["mean_raster_iou_gain"] >= 0.0
            and row["vector_delta_topology_valid_rate"] >= topology_floor
            and row["mean_vector_replay_iou"] >= replay_floor
        )
    feasible = [row for row in summaries if row["feasible"]]
    selected = min(
        feasible,
        key=lambda row: (
            row["mean_component_count_absolute_error"],
            -row["mean_raster_iou"],
            row["min_delta_component_pixels"],
        ),
    )
    by_task: dict[str, int] = defaultdict(int)
    for row in source_rows:
        by_task[str(row["task_id"])] += 1
    return {
        "schema_version": "writeback-delta-topology-sweep-v1",
        "selection_split": "validation",
        "source_writebacks": [str(path.resolve()) for path in paths],
        "sample_count": len(source_rows),
        "task_count": len(by_task),
        "selection_constraints": {
            "raster_iou_tolerance_vs_raw": raster_iou_tolerance,
            "mean_raster_iou_gain_floor": 0.0,
            "topology_valid_rate_floor": topology_floor,
            "vector_replay_iou_floor": replay_floor,
        },
        "candidates": summaries,
        "selected_min_delta_component_pixels": selected[
            "min_delta_component_pixels"
        ],
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--writeback", type=Path, action="append", required=True)
    parser.add_argument(
        "--min-pixels", type=int, action="append", default=[0, 2, 4, 8, 16, 32, 64]
    )
    parser.add_argument("--raster-iou-tolerance", type=float, default=0.001)
    parser.add_argument("--topology-floor", type=float, default=0.99)
    parser.add_argument("--replay-floor", type=float, default=0.98)
    args = parser.parse_args()
    result = sweep(
        args.writeback,
        args.min_pixels,
        raster_iou_tolerance=args.raster_iou_tolerance,
        topology_floor=args.topology_floor,
        replay_floor=args.replay_floor,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
