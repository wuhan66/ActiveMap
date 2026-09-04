#!/usr/bin/env python3
"""Train-only calibration of operation-conditioned SN7 Safe Commit gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from activemap.models import EditOperation
from scripts.calibrate_sn7_v5_safe_commit import (
    POLICIES,
    apply_safe_commit,
    bool_value,
    load_rows,
    macro_aoi_summary,
    parse_record,
    sha256,
    target_operation,
)

EDIT_OPERATIONS = (
    EditOperation.ADD.value,
    EditOperation.DELETE.value,
    EditOperation.RESHAPE.value,
)


def _validate_paired_support(
    by_policy: dict[str, dict[tuple[str, float], dict[str, Any]]],
) -> None:
    if tuple(by_policy) != POLICIES:
        raise ValueError(f"records must appear once in the fixed order: {POLICIES}")
    reference = by_policy[POLICIES[0]]
    reference_keys = set(reference)
    for policy in POLICIES[1:]:
        candidate = by_policy[policy]
        if set(candidate) != reference_keys:
            raise ValueError(f"{policy} lacks paired train support")
        for key in reference_keys:
            if (
                str(candidate[key]["aoi_id"]) != str(reference[key]["aoi_id"])
                or str(candidate[key]["target"]) != str(reference[key]["target"])
            ):
                raise ValueError(f"policy provenance mismatch at {key}")


def apply_operation_safe_commit(
    row: dict[str, Any],
    *,
    confidence_thresholds: dict[str, float],
    replay_iou_threshold: float,
    require_topology: bool,
) -> dict[str, Any]:
    operation = str(row["effective_operation"])
    threshold = (
        float(confidence_thresholds[operation])
        if bool_value(row["writeback_changed"])
        else 0.0
    )
    result = apply_safe_commit(
        row,
        confidence_threshold=threshold,
        replay_iou_threshold=replay_iou_threshold,
        require_topology=require_topology,
    )
    result.update(
        {
            "safe_commit_gate_mode": "operation_conditioned",
            "safe_commit_proposed_operation": operation,
            "safe_commit_operation_thresholds": dict(confidence_thresholds),
        }
    )
    return result


def operation_threshold_candidates(
    rows: list[dict[str, Any]],
    *,
    operation: str,
    thresholds: np.ndarray,
    replay_iou_threshold: float,
    require_topology: bool,
) -> list[dict[str, Any]]:
    """Vectorize the unchanged V5 Safe Commit metric contract over a grid."""

    if not rows:
        raise ValueError("operation threshold grid requires rows")
    aoi_names = sorted({str(row["aoi_id"]) for row in rows})
    aoi_lookup = {name: index for index, name in enumerate(aoi_names)}
    aoi_index = np.asarray(
        [aoi_lookup[str(row["aoi_id"])] for row in rows], dtype=np.int64
    )
    aoi_counts = np.bincount(aoi_index, minlength=len(aoi_names)).astype(np.float64)

    confidence = np.asarray([float(row["fused_confidence"]) for row in rows])
    replay_valid = np.asarray(
        [float(row["vector_replay_iou"]) >= replay_iou_threshold for row in rows]
    )
    topology_valid = np.asarray(
        [bool_value(row["vector_delta_topology_valid"]) for row in rows]
    )
    hard_valid = replay_valid & (topology_valid if require_topology else True)
    prior = np.asarray([float(row["prior_raster_iou"]) for row in rows])
    raw_final = np.asarray([float(row["raster_iou"]) for row in rows])
    spent_cost = np.asarray([float(row["spent_cost"]) for row in rows])
    budget = np.asarray([float(row["budget"]) for row in rows])
    normalized_cost = np.minimum(spent_cost / budget, 1.0)
    tool_called = np.asarray(
        [bool_value(row.get("semantic_tool_called", False)) for row in rows],
        dtype=np.float64,
    )
    target = np.asarray([target_operation(str(row["target"])).value for row in rows])
    target_keep = target == EditOperation.KEEP.value
    target_matches = target == operation
    prior_topology_valid = np.asarray(
        [bool_value(row.get("topology_quality_before", True)) for row in rows]
    )

    def macro(values: np.ndarray) -> float:
        totals = np.bincount(
            aoi_index, weights=values.astype(np.float64), minlength=len(aoi_names)
        )
        return float(np.mean(totals / aoi_counts))

    candidates = []
    for threshold in thresholds:
        accepted = hard_valid & (confidence >= float(threshold))
        final = np.where(accepted, raw_final, prior)
        quality_gain = final - prior
        false_edit = target_keep & accepted
        missed_edit = (~target_keep) & (~accepted)
        wrong_edit = (~target_keep) & accepted & (~target_matches)
        final_topology_valid = np.where(
            accepted, topology_valid, prior_topology_valid
        )
        balanced_utility = (
            quality_gain
            - 0.10 * normalized_cost
            - 0.50 * false_edit
            - 0.25 * missed_edit
            - 0.25 * wrong_edit
            - 0.25 * (~final_topology_valid)
        )
        candidates.append(
            {
                "confidence_threshold": float(threshold),
                "metrics": {
                    "map_quality_after": macro(final),
                    "map_quality_gain": macro(quality_gain),
                    "balanced_utility": macro(balanced_utility),
                    "false_edit_rate": macro(false_edit),
                    "missed_edit_rate": macro(missed_edit),
                    "commit_rate": macro(accepted),
                    "additional_evidence_rate": macro(tool_called),
                    "mean_additional_cost": macro(spent_cost),
                },
            }
        )
    return candidates


def choose_operation_thresholds(
    by_policy: dict[str, dict[tuple[str, float], dict[str, Any]]],
    *,
    replay_iou_threshold: float,
    require_topology: bool,
    threshold_count: int,
    maximum_false_edit_rate: float,
) -> dict[str, Any]:
    """Choose one train-only confidence threshold per proposed edit operation."""

    _validate_paired_support(by_policy)
    pooled = [row for records in by_policy.values() for row in records.values()]
    thresholds = np.linspace(0.0, 1.0, threshold_count)
    operation_results: dict[str, Any] = {}
    selected_thresholds: dict[str, float] = {}
    for operation in EDIT_OPERATIONS:
        rows = [
            row
            for row in pooled
            if bool(row["writeback_changed"])
            and str(row["effective_operation"]) == operation
        ]
        if not rows:
            raise ValueError(f"no changed train proposals for {operation}")
        candidates = operation_threshold_candidates(
            rows,
            operation=operation,
            thresholds=thresholds,
            replay_iou_threshold=replay_iou_threshold,
            require_topology=require_topology,
        )
        feasible = [
            item
            for item in candidates
            if item["metrics"]["false_edit_rate"]
            <= maximum_false_edit_rate + 1e-12
        ]
        if not feasible:
            raise RuntimeError(
                f"no {operation} threshold satisfies false-edit cap "
                f"{maximum_false_edit_rate}"
            )
        selected = max(
            feasible,
            key=lambda item: (
                item["metrics"]["map_quality_after"],
                -item["metrics"]["false_edit_rate"],
                -item["metrics"]["missed_edit_rate"],
                -item["confidence_threshold"],
            ),
        )
        selected_thresholds[operation] = selected["confidence_threshold"]
        operation_results[operation] = {
            "changed_proposal_count": len(rows),
            "aoi_count": len({str(row["aoi_id"]) for row in rows}),
            "raw_train_macro_aoi": macro_aoi_summary(rows),
            "threshold_candidates": candidates,
            "selected": selected,
        }
    return {
        "policies": list(POLICIES),
        "confidence_thresholds": selected_thresholds,
        "operations": operation_results,
        "selection_rule": (
            "For each proposed edit operation, maximize train-AOI-macro final "
            "map quality subject to the fixed operation-slice false-edit cap; "
            "ties prefer lower false edit, lower missed edit, then lower threshold."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--record", action="append", type=parse_record, required=True)
    parser.add_argument("--maximum-false-edit-rate", type=float, default=0.01)
    parser.add_argument("--replay-iou-threshold", type=float, default=0.99)
    parser.add_argument("--allow-invalid-topology", action="store_true")
    parser.add_argument("--threshold-count", type=int, default=101)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite calibration: {args.output}")
    if not 0.0 <= args.maximum_false_edit_rate <= 1.0:
        raise ValueError("maximum false-edit rate must be between zero and one")
    if not 0.0 <= args.replay_iou_threshold <= 1.0 or args.threshold_count < 2:
        raise ValueError("invalid operation-conditioned calibration grid")
    records = dict(args.record)
    if len(records) != len(args.record):
        raise ValueError("duplicate Safe Commit policy record")
    by_policy = {
        policy: load_rows(path, expected_split="train")
        for policy, path in records.items()
    }
    result = choose_operation_thresholds(
        by_policy,
        replay_iou_threshold=args.replay_iou_threshold,
        require_topology=not args.allow_invalid_topology,
        threshold_count=args.threshold_count,
        maximum_false_edit_rate=args.maximum_false_edit_rate,
    )
    result.update(
        {
            "schema_version": "sn7-operation-safe-commit-calibration-v1",
            "split": "train",
            "policy_blind": True,
            "maximum_false_edit_rate": args.maximum_false_edit_rate,
            "inputs": {
                policy: {"path": str(path.resolve()), "sha256": sha256(path)}
                for policy, path in records.items()
            },
            "gate": {
                "confidence_thresholds": result["confidence_thresholds"],
                "replay_iou_threshold": args.replay_iou_threshold,
                "require_topology": not args.allow_invalid_topology,
            },
            "test_assets_read": False,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
