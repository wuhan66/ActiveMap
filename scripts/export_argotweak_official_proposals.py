from __future__ import annotations

import argparse
import json
from pathlib import Path

import mmcv
import numpy as np

from activemap.data.argotweak_native_export import (
    argotweak_frame_operation_counts,
    assign_argotweak_proposal_objects,
)

CHANGE_TO_OPERATION = {0: "KEEP", 1: "DELETE", 2: "ADD", 3: "RESHAPE"}
CLASS_NAMES = {0: "lane_segment", 1: "pedestrian_crossing"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export official ArgoTweak predictions as portable atomic-edit proposals."
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--object-threshold", type=float, default=0.3)
    parser.add_argument("--object-match-distance", type=float, default=1.5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite proposal export: {args.output}")

    results = mmcv.load(str(args.results), file_format="pkl")
    annotations = mmcv.load(str(args.annotations), file_format="pkl")
    frame_rows = (
        list(annotations.items())
        if isinstance(annotations, dict)
        else [(None, row) for row in annotations]
    )
    if len(results) != len(frame_rows):
        raise ValueError(
            f"prediction/annotation length mismatch: {len(results)} != {len(frame_rows)}"
        )

    rows = []
    operation_counts = {operation: 0 for operation in CHANGE_TO_OPERATION.values()}
    assignment_counts = {"matched": 0, "unmatched": 0, "official_objects": 0}
    assignment_by_operation = {
        operation: {"matched": 0, "unmatched": 0} for operation in CHANGE_TO_OPERATION.values()
    }
    assignment_operation_confusion = {
        predicted: {target: 0 for target in CHANGE_TO_OPERATION.values()}
        for predicted in CHANGE_TO_OPERATION.values()
    }
    # The isolated official runtime is Python 3.8, before zip(strict=...).
    for (identifier, frame), result in zip(frame_rows, results):  # noqa: B905
        lane_results = result["lane_results"]
        geometry = lane_results[0].reshape(-1, 30, 2)
        object_scores = lane_results[1]
        class_labels = lane_results[2]
        change_scores = lane_results[7]
        change_labels = lane_results[8]
        geometry_scores = lane_results[9]
        geometry_labels = lane_results[10]
        mark_scores = lane_results[11]
        mark_labels = lane_results[12]
        proposals = []
        for query_index in np.flatnonzero(object_scores >= args.object_threshold):
            class_label = int(class_labels[query_index])
            change_label = int(change_labels[query_index])
            if class_label not in CLASS_NAMES or change_label not in CHANGE_TO_OPERATION:
                continue
            operation = CHANGE_TO_OPERATION[change_label]
            operation_counts[operation] += 1
            proposals.append(
                {
                    "proposal_id": f"query-{int(query_index):03d}",
                    "object_id": None,
                    "object_match_status": "pending_geometry_assignment",
                    "feature_class": CLASS_NAMES[class_label],
                    "operation": operation,
                    "geometry": geometry[query_index].tolist(),
                    "confidence": {
                        "object": float(object_scores[query_index]),
                        "change": float(change_scores[query_index]),
                        "joint": float(object_scores[query_index] * change_scores[query_index]),
                        "geometry": float(geometry_scores[query_index]),
                        "mark": float(mark_scores[query_index]),
                    },
                    "auxiliary_labels": {
                        "change": change_label,
                        "geometry": int(geometry_labels[query_index]),
                        "mark": int(mark_labels[query_index]),
                    },
                }
            )
        frame_assignment = assign_argotweak_proposal_objects(
            proposals, frame, max_distance=args.object_match_distance
        )
        for key, value in frame_assignment.items():
            assignment_counts[key] += value
        for proposal in proposals:
            assignment = (
                "matched" if proposal["object_match_status"].startswith("matched_") else "unmatched"
            )
            assignment_by_operation[proposal["operation"]][assignment] += 1
            target_operation = proposal.get("matched_target_operation")
            if target_operation is not None:
                assignment_operation_confusion[proposal["operation"]][target_operation] += 1
        rows.append(
            {
                "schema_version": "activemap-argotweak-official-proposals-v1",
                "split": identifier[0] if identifier is not None else "unknown",
                "segment_id": frame["segment_id"],
                "timestamp": str(frame["timestamp"]),
                "object_threshold": args.object_threshold,
                "proposals": proposals,
                "gt_operation_counts": argotweak_frame_operation_counts(frame),
                "supervision_unit": "official_frame_local_transaction",
                "object_identity_available": frame_assignment["matched"] > 0,
                "object_assignment": frame_assignment,
                "city_se3_egovehicle": {
                    "rotation": np.asarray(frame["pose"]["rotation"]).tolist(),
                    "translation": np.asarray(frame["pose"]["translation"]).tolist(),
                },
                "test_assets_read": False,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "activemap-argotweak-official-proposal-summary-v1",
        "frames": len(rows),
        "operation_counts": operation_counts,
        "object_threshold": args.object_threshold,
        "object_assignment": "official_local_geometry_fail_closed",
        "object_assignment_policy": "official_local_geometry_fail_closed",
        "object_match_distance": args.object_match_distance,
        "object_assignment_counts": assignment_counts,
        "object_assignment_by_operation": assignment_by_operation,
        "matched_operation_confusion": assignment_operation_confusion,
        "test_assets_read": False,
    }
    args.output.with_suffix(args.output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(summary)


if __name__ == "__main__":
    main()
