"""Dependency-free ArgoTweak annotation parsing for the legacy runtime."""

from __future__ import annotations

import math
from typing import Any


def _xy_points(value: Any) -> list[tuple[float, float]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        return []
    return [
        (float(row[0]), float(row[1]))
        for row in value
        if isinstance(row, (list, tuple)) and len(row) >= 2
    ]


def _mean_nearest_distance(
    source: list[tuple[float, float]], target: list[tuple[float, float]]
) -> float:
    if not source or not target:
        return math.inf
    return sum(min(math.hypot(x - tx, y - ty) for tx, ty in target) for x, y in source) / len(
        source
    )


def _symmetric_point_distance(
    first: list[tuple[float, float]], second: list[tuple[float, float]]
) -> float:
    return 0.5 * (_mean_nearest_distance(first, second) + _mean_nearest_distance(second, first))


def _frame_objects(frame: dict[str, Any]) -> list[dict[str, Any]]:
    annotation = frame.get("annotation")
    if not isinstance(annotation, dict):
        raise ValueError("ArgoTweak frame lacks annotation")
    objects: list[dict[str, Any]] = []
    for row in annotation.get("lane_segment") or []:
        points: list[tuple[float, float]] = []
        for key in ("centerline", "left_laneline", "right_laneline"):
            points.extend(_xy_points(row.get(key)))
        objects.append(
            {
                "object_id": str(row["id"]),
                "feature_class": "lane_segment",
                "points": points,
                "target_operation": _change_operation(
                    row.get("change_score", row.get("changes", [0]))
                ),
            }
        )
    for row in annotation.get("area") or []:
        if int(row.get("category", -1)) == 1:
            objects.append(
                {
                    "object_id": str(row["id"]),
                    "feature_class": "pedestrian_crossing",
                    "points": _xy_points(row.get("points")),
                    "target_operation": _change_operation(
                        row.get("change_score", row.get("changes", [0]))
                    ),
                }
            )
    return objects


def assign_argotweak_proposal_objects(
    proposals: list[dict[str, Any]],
    frame: dict[str, Any],
    *,
    max_distance: float = 1.5,
) -> dict[str, int]:
    """Associate frame-local predictions to official objects with fail-closed matching."""

    if max_distance <= 0.0:
        raise ValueError("max_distance must be positive")
    objects = _frame_objects(frame)
    candidates: list[tuple[float, int, int]] = []
    for proposal_index, proposal in enumerate(proposals):
        points = _xy_points(proposal.get("geometry"))
        for object_index, obj in enumerate(objects):
            if proposal.get("feature_class") != obj["feature_class"]:
                continue
            distance = _symmetric_point_distance(points, obj["points"])
            if distance <= max_distance:
                candidates.append((distance, proposal_index, object_index))
    assigned_proposals: set[int] = set()
    assigned_objects: set[int] = set()
    for distance, proposal_index, object_index in sorted(candidates):
        if proposal_index in assigned_proposals or object_index in assigned_objects:
            continue
        proposal = proposals[proposal_index]
        proposal["object_id"] = objects[object_index]["object_id"]
        proposal["object_match_status"] = "matched_official_local_geometry"
        proposal["object_match_distance"] = float(distance)
        proposal["matched_target_operation"] = objects[object_index]["target_operation"]
        assigned_proposals.add(proposal_index)
        assigned_objects.add(object_index)
    for index, proposal in enumerate(proposals):
        if index not in assigned_proposals:
            proposal["object_id"] = None
            proposal["object_match_status"] = "unmatched_fail_closed"
            proposal["object_match_distance"] = None
            proposal["matched_target_operation"] = None
    return {
        "matched": len(assigned_proposals),
        "unmatched": len(proposals) - len(assigned_proposals),
        "official_objects": len(objects),
    }


def _change_operation(value: Any) -> str:
    codes = value if isinstance(value, list) else [value]
    normalized = {int(code) for code in codes if code is not None}
    if not normalized or normalized == {0}:
        return "KEEP"
    if 2 in normalized:
        return "ADD"
    if 1 in normalized:
        return "DELETE"
    if normalized & {3, 5}:
        return "RESHAPE"
    raise ValueError(f"unsupported ArgoTweak change code: {sorted(normalized)}")


def argotweak_frame_operation_counts(frame: dict[str, Any]) -> dict[str, int]:
    """Extract official frame-local operation supervision without geometry matching."""

    annotation = frame.get("annotation")
    if not isinstance(annotation, dict):
        raise ValueError("ArgoTweak frame lacks annotation")
    rows = list(annotation.get("lane_segment") or [])
    rows.extend(row for row in annotation.get("area") or [] if int(row.get("category", -1)) == 1)
    counts = {operation: 0 for operation in ("KEEP", "ADD", "DELETE", "RESHAPE")}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid ArgoTweak frame annotation element")
        operation = _change_operation(row.get("change_score", row.get("changes", [0])))
        counts[operation] += 1
    return counts
