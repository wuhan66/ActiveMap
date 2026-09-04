#!/usr/bin/env python3
"""Train-only calibration of the V5 terminal Safe Commit threshold.

The calibration consumes real typed writebacks from the paired V5 ``direct``
and ``selected`` train rollouts.  It never sees validation or frozen-test
rows.  A single confidence threshold is selected over both evidence policies
to preserve the factorial comparison; vector replay and topology validity are
hard constraints for every changed writeback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from activemap.evaluation.episode_utility import score_episode_profiles
from activemap.models import EditOperation


POLICIES = ("direct", "selected")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_record(value: str) -> tuple[str, Path]:
    policy, separator, raw_path = value.partition("=")
    if not separator or policy not in POLICIES or not raw_path:
        raise argparse.ArgumentTypeError(
            "record must be direct=WRITEBACK_JSONL or selected=WRITEBACK_JSONL"
        )
    return policy, Path(raw_path)


def bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def target_operation(value: str) -> EditOperation:
    if value == "REJECT":
        return EditOperation.KEEP
    if not value.startswith("COMMIT:"):
        raise ValueError(f"unsupported terminal target {value!r}")
    return EditOperation(value.removeprefix("COMMIT:"))


def executable_errors(
    target: EditOperation, executed: EditOperation
) -> tuple[bool, bool, bool]:
    false_edit = target == EditOperation.KEEP and executed != EditOperation.KEEP
    missed_edit = target != EditOperation.KEEP and executed == EditOperation.KEEP
    wrong_edit = (
        target != EditOperation.KEEP
        and executed != EditOperation.KEEP
        and executed != target
    )
    return false_edit, missed_edit, wrong_edit


def row_errors(row: dict[str, Any]) -> tuple[bool, bool, bool]:
    """Use recorded error flags when available, otherwise derive executable errors."""

    names = ("false_edit", "missed_edit", "wrong_edit")
    if all(name in row for name in names):
        return tuple(bool_value(row[name]) for name in names)  # type: ignore[return-value]
    return executable_errors(
        target_operation(str(row["target"])),
        EditOperation(str(row["effective_operation"])),
    )


def validate_row(row: dict[str, Any], *, expected_split: str) -> None:
    required = {
        "task_id",
        "aoi_id",
        "budget",
        "target",
        "effective_operation",
        "writeback_changed",
        "fused_confidence",
        "vector_replay_iou",
        "vector_delta_topology_valid",
        "prior_raster_iou",
        "raster_iou",
        "spent_cost",
        "split",
        "test_assets_read",
    }
    missing = sorted(required - row.keys())
    if missing:
        raise ValueError(f"writeback row is missing fields {missing}")
    if row["split"] != expected_split or row["test_assets_read"] is not False:
        raise ValueError("V5 Safe Commit accepts only the requested non-test split")
    target_operation(str(row["target"]))
    EditOperation(str(row["effective_operation"]))


def load_rows(path: Path, *, expected_split: str) -> dict[tuple[str, float], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    result: dict[tuple[str, float], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            validate_row(row, expected_split=expected_split)
            key = (str(row["task_id"]), float(row["budget"]))
            if key in result:
                raise ValueError(f"duplicate task-budget row in {path}:{line_number}")
            result[key] = row
    if not result:
        raise ValueError(f"no writeback rows in {path}")
    return result


def safe_commit_accepts(
    row: dict[str, Any],
    *,
    confidence_threshold: float,
    replay_iou_threshold: float,
    require_topology: bool,
) -> bool:
    if not bool_value(row["writeback_changed"]):
        return True
    return bool(
        float(row["fused_confidence"]) >= confidence_threshold
        and float(row["vector_replay_iou"]) >= replay_iou_threshold
        and (
            not require_topology
            or bool_value(row["vector_delta_topology_valid"])
        )
    )


def apply_safe_commit(
    row: dict[str, Any],
    *,
    confidence_threshold: float,
    replay_iou_threshold: float,
    require_topology: bool,
) -> dict[str, Any]:
    """Return the terminal map state after accepting or reverting a raw delta."""

    accepted = safe_commit_accepts(
        row,
        confidence_threshold=confidence_threshold,
        replay_iou_threshold=replay_iou_threshold,
        require_topology=require_topology,
    )
    result = dict(row)
    target = target_operation(str(row["target"]))
    executed = EditOperation(str(row["effective_operation"])) if accepted else EditOperation.KEEP
    false_edit, missed_edit, wrong_edit = executable_errors(target, executed)
    prior = float(row["prior_raster_iou"])
    final = float(row["raster_iou"]) if accepted else prior
    utility = score_episode_profiles(
        final_map_quality=final,
        prior_map_quality=prior,
        spent_cost=float(row["spent_cost"]),
        budget=float(row["budget"]),
        false_edit=false_edit,
        missed_edit=missed_edit,
        wrong_edit=wrong_edit,
        topology_valid=(
            bool_value(row["vector_delta_topology_valid"])
            if accepted
            else bool(row.get("topology_quality_before", True))
        ),
    )
    result.update(
        {
            "safe_commit_applied": True,
            "safe_commit_accepted": accepted,
            "safe_commit_confidence_threshold": confidence_threshold,
            "safe_commit_replay_iou_threshold": replay_iou_threshold,
            "safe_commit_require_topology": require_topology,
            "safe_commit_rejection_reason": (
                None
                if accepted
                else "confidence_or_vector_validity_gate"
            ),
            "effective_operation": executed.value,
            "writeback_changed": executed != EditOperation.KEEP,
            "false_edit": false_edit,
            "missed_edit": missed_edit,
            "wrong_edit": wrong_edit,
            "raster_iou": final,
            "map_quality_after": final,
            "map_quality_before": prior,
            "raster_iou_gain": final - prior,
            "episode_utility_v2": utility,
            "episode_utility_v2_balanced": utility["balanced"]["value"],
            "episode_utility_v2_safety": utility["safety"]["value"],
            "episode_utility_v2_cost_aware": utility["cost_aware"]["value"],
        }
    )
    if not accepted:
        result.update(
            {
                "topology_quality_after": float(row.get("topology_quality_before", 1.0)),
                "predicted_add_geometry": None,
                "predicted_remove_geometry": None,
                "safe_commit_mask_source": "prior_mask",
            }
        )
    return result


def balanced_utility(row: dict[str, Any]) -> float:
    """Read a recorded utility or derive it for minimally stored raw rows."""

    recorded = row.get("episode_utility_v2_balanced")
    if recorded is not None:
        return float(recorded)
    profile = row.get("episode_utility_v2")
    if isinstance(profile, dict):
        balanced = profile.get("balanced")
        if isinstance(balanced, dict) and "value" in balanced:
            return float(balanced["value"])
    false_edit, missed_edit, wrong_edit = row_errors(row)
    utility = score_episode_profiles(
        final_map_quality=float(row["raster_iou"]),
        prior_map_quality=float(row["prior_raster_iou"]),
        spent_cost=float(row["spent_cost"]),
        budget=float(row["budget"]),
        false_edit=false_edit,
        missed_edit=missed_edit,
        wrong_edit=wrong_edit,
        topology_valid=bool_value(row["vector_delta_topology_valid"]),
    )
    return float(utility["balanced"]["value"])


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot summarize an empty Safe Commit cell")
    return {
        "map_quality_after": float(np.mean([float(row["raster_iou"]) for row in rows])),
        "map_quality_gain": float(np.mean([float(row["raster_iou_gain"]) for row in rows])),
        "balanced_utility": float(np.mean([balanced_utility(row) for row in rows])),
        "false_edit_rate": float(np.mean([row_errors(row)[0] for row in rows])),
        "missed_edit_rate": float(np.mean([row_errors(row)[1] for row in rows])),
        "commit_rate": float(np.mean([bool_value(row["writeback_changed"]) for row in rows])),
        "additional_evidence_rate": float(
            np.mean([bool_value(row.get("semantic_tool_called", False)) for row in rows])
        ),
        "mean_additional_cost": float(np.mean([float(row["spent_cost"]) for row in rows])),
    }


def macro_aoi_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    by_aoi: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_aoi[str(row["aoi_id"])].append(row)
    if len(by_aoi) < 2:
        raise ValueError("Safe Commit calibration requires multiple train AOIs")
    by_aoi_metrics = [summarize(group) for group in by_aoi.values()]
    return {
        name: float(np.mean([group[name] for group in by_aoi_metrics]))
        for name in by_aoi_metrics[0]
    }


def choose_common_threshold(
    by_policy: dict[str, dict[tuple[str, float], dict[str, Any]]],
    *,
    expected_policies: tuple[str, ...],
    replay_iou_threshold: float,
    require_topology: bool,
    threshold_count: int,
) -> dict[str, Any]:
    """Fit one policy-blind gate on a predeclared matched-policy union."""

    if tuple(by_policy) != expected_policies:
        raise ValueError(
            "Safe Commit calibration requires policies in this exact order: "
            f"{expected_policies}"
        )
    reference_policy = expected_policies[0]
    reference_rows = by_policy[reference_policy]
    reference_keys = set(reference_rows)
    for policy in expected_policies[1:]:
        candidate_rows = by_policy[policy]
        if reference_keys != set(candidate_rows):
            raise ValueError(
                f"{reference_policy} and {policy} train writebacks lack paired support"
            )
        for key in reference_keys:
            if (
                str(reference_rows[key]["aoi_id"])
                != str(candidate_rows[key]["aoi_id"])
                or str(reference_rows[key]["target"])
                != str(candidate_rows[key]["target"])
            ):
                raise ValueError(f"policy provenance mismatch at {key}")
    pooled_raw = [row for records in by_policy.values() for row in records.values()]
    raw_metrics = macro_aoi_summary(pooled_raw)
    thresholds = np.linspace(0.0, 1.0, threshold_count)
    candidates: list[dict[str, Any]] = []
    for threshold in thresholds:
        gated = [
            apply_safe_commit(
                row,
                confidence_threshold=float(threshold),
                replay_iou_threshold=replay_iou_threshold,
                require_topology=require_topology,
            )
            for row in pooled_raw
        ]
        metrics = macro_aoi_summary(gated)
        candidates.append({"confidence_threshold": float(threshold), "metrics": metrics})
    feasible = [
        item
        for item in candidates
        if item["metrics"]["false_edit_rate"] <= raw_metrics["false_edit_rate"] + 1e-12
    ]
    if not feasible:
        raise RuntimeError("no train threshold preserves the raw false-edit rate")
    selected = max(
        feasible,
        key=lambda item: (
            item["metrics"]["map_quality_after"],
            -item["metrics"]["false_edit_rate"],
            -item["metrics"]["missed_edit_rate"],
            -item["confidence_threshold"],
        ),
    )
    return {
        "policies": list(expected_policies),
        "raw_train_macro_aoi": raw_metrics,
        "threshold_candidates": candidates,
        "selected": selected,
        "selection_rule": (
            "maximize train-AOI-macro final map quality subject to false-edit rate "
            "not exceeding the paired raw-writeback reference; ties prefer lower "
            "false-edit, lower missed-edit, then the smaller threshold"
        ),
    }


def choose_threshold(
    by_policy: dict[str, dict[tuple[str, float], dict[str, Any]]],
    *,
    replay_iou_threshold: float,
    require_topology: bool,
    threshold_count: int,
) -> dict[str, Any]:
    """Backwards-compatible two-policy V5 calibration wrapper."""

    return choose_common_threshold(
        by_policy,
        expected_policies=POLICIES,
        replay_iou_threshold=replay_iou_threshold,
        require_topology=require_topology,
        threshold_count=threshold_count,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--record", action="append", type=parse_record, required=True)
    parser.add_argument("--replay-iou-threshold", type=float, default=0.99)
    parser.add_argument("--allow-invalid-topology", action="store_true")
    parser.add_argument("--threshold-count", type=int, default=101)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite calibration: {args.output}")
    if not 0.0 <= args.replay_iou_threshold <= 1.0 or args.threshold_count < 2:
        raise ValueError("invalid Safe Commit calibration grid")
    records = dict(args.record)
    if len(records) != len(args.record):
        raise ValueError("duplicate Safe Commit policy record")
    by_policy = {
        policy: load_rows(path, expected_split="train")
        for policy, path in records.items()
    }
    result = choose_threshold(
        by_policy,
        replay_iou_threshold=args.replay_iou_threshold,
        require_topology=not args.allow_invalid_topology,
        threshold_count=args.threshold_count,
    )
    result.update(
        {
            "schema_version": "sn7-v5-safe-commit-calibration-v1",
            "split": "train",
            "inputs": {
                policy: {"path": str(path.resolve()), "sha256": sha256(path)}
                for policy, path in records.items()
            },
            "gate": {
                "confidence_threshold": result["selected"]["confidence_threshold"],
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
