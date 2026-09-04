#!/usr/bin/env python3
"""Trace SN7 operation support from candidates through safe writeback.

This validation-only diagnostic joins immutable selector states, selected-policy
rollouts, raw executable writebacks, and Safe Commit writebacks. It identifies
where ADD, DELETE, and RESHAPE support is lost without training a new model.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import statistics
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

OPERATIONS = ("KEEP", "ADD", "DELETE", "RESHAPE")
NONKEEP_OPERATIONS = ("ADD", "DELETE", "RESHAPE")


@dataclass(frozen=True)
class Bundle:
    seed: int
    states: Path
    rollout: Path
    raw_writeback: Path
    safe_writeback: Path


def operation(value: Any) -> str:
    text = str(value or "KEEP").upper().replace("-", "_").replace(" ", "")
    text = text.removeprefix("COMMIT:")
    if text in {"", "STOP", "REJECT", "KEEP"}:
        return "KEEP"
    if text in NONKEEP_OPERATIONS:
        return text
    raise ValueError(f"unsupported operation: {value!r}")


def bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def state_key(source_episode: Any, budget: Any) -> tuple[str, float]:
    return str(source_episode), round(float(budget), 6)


def _require_validation(row: dict[str, Any], *, source: Path, line_number: int) -> None:
    if row.get("split") != "val":
        raise ValueError(f"{source}:{line_number}: expected validation-only record")
    metadata = row.get("metadata", {})
    if row.get("test_assets_read") is True or (
        isinstance(metadata, dict) and metadata.get("test_assets_read") is True
    ):
        raise ValueError(f"{source}:{line_number}: test provenance is forbidden")


def _stream_jsonl(
    path: Path, compact: Callable[[dict[str, Any], int], dict[str, Any] | None]
) -> tuple[list[dict[str, Any]], str, int]:
    digest = hashlib.sha256()
    rows: list[dict[str, Any]] = []
    line_count = 0
    with path.open("rb") as handle:
        for line_count, line in enumerate(handle, start=1):
            digest.update(line)
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except Exception as exc:
                raise ValueError(f"{path}:{line_count}: invalid JSON") from exc
            item = compact(payload, line_count)
            if item is not None:
                rows.append(item)
    return rows, digest.hexdigest(), line_count


def _safe(candidate: dict[str, Any], anchor: dict[str, Any]) -> bool:
    return (
        bool_value(candidate.get("false_edit"))
        <= bool_value(anchor.get("false_edit"))
        and bool_value(candidate.get("missed_edit"))
        <= bool_value(anchor.get("missed_edit"))
    )


def compact_state(row: dict[str, Any], *, source: Path, line_number: int) -> dict[str, Any] | None:
    _require_validation(row, source=source, line_number=line_number)
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{source}:{line_number}: missing state metadata")
    if int(metadata.get("oracle_step", -1)) != 0:
        return None
    target_operation = operation(metadata.get("gt_edit"))
    evidence_ids = [str(value) for value in row.get("evidence_ids", [])]
    utilities = [float(value) for value in row.get("oracle_utilities", [])]
    if not evidence_ids or len(evidence_ids) != len(utilities):
        raise ValueError(f"{source}:{line_number}: malformed visible candidate utilities")
    stop_utility = float(row.get("stop_utility", 0.0))
    best_visible_index = max(range(len(utilities)), key=utilities.__getitem__)
    best_visible_id = evidence_ids[best_visible_index]
    best_visible_utility = utilities[best_visible_index]

    outcomes = metadata.get("executable_outcomes")
    predictions = metadata.get("evidence_predictions")
    anchor_id = metadata.get("initial_evidence_id")
    if not isinstance(outcomes, dict) or not isinstance(predictions, dict):
        raise ValueError(f"{source}:{line_number}: missing candidate outcome registry")
    if not isinstance(anchor_id, str) or anchor_id not in outcomes:
        raise ValueError(f"{source}:{line_number}: invalid direct anchor")
    anchor = outcomes[anchor_id]
    if not isinstance(anchor, dict) or "final_raster_iou" not in anchor:
        raise ValueError(f"{source}:{line_number}: malformed anchor outcome")

    candidates = []
    target_candidate_count = 0
    signatures = set()
    for evidence_id, candidate in outcomes.items():
        if not isinstance(candidate, dict) or "final_raster_iou" not in candidate:
            raise ValueError(f"{source}:{line_number}: malformed outcome {evidence_id}")
        prediction = predictions.get(evidence_id, {})
        predicted_operation = operation(
            prediction.get("gated_edit", candidate.get("predicted_operation", "KEEP"))
            if isinstance(prediction, dict)
            else candidate.get("predicted_operation", "KEEP")
        )
        target_candidate_count += int(predicted_operation == target_operation)
        compact_candidate = {
            "evidence_id": str(evidence_id),
            "final_raster_iou": float(candidate["final_raster_iou"]),
            "false_edit": bool_value(candidate.get("false_edit")),
            "missed_edit": bool_value(candidate.get("missed_edit")),
            "wrong_edit": bool_value(candidate.get("wrong_edit")),
            "predicted_operation": predicted_operation,
        }
        signatures.add(
            (
                predicted_operation,
                round(compact_candidate["final_raster_iou"], 8),
                compact_candidate["false_edit"],
                compact_candidate["missed_edit"],
                compact_candidate["wrong_edit"],
            )
        )
        candidates.append(compact_candidate)

    safe_candidates = [candidate for candidate in candidates if _safe(candidate, anchor)]
    if not safe_candidates:
        raise ValueError(f"{source}:{line_number}: direct anchor missing from safe bank")
    safe_oracle = max(
        safe_candidates,
        key=lambda item: (
            item["final_raster_iou"],
            -item["missed_edit"],
            item["evidence_id"],
        ),
    )
    anchor_iou = float(anchor["final_raster_iou"])
    candidate_raster_iou_gains = {
        candidate["evidence_id"]: candidate["final_raster_iou"] - anchor_iou
        for candidate in candidates
    }
    safe_headroom = safe_oracle["final_raster_iou"] - anchor_iou
    source_episode = str(metadata.get("source_episode", row.get("sample_id")))
    budget = float(metadata.get("budget"))
    return {
        "sample_id": str(row.get("sample_id")),
        "key": state_key(source_episode, budget),
        "source_episode": source_episode,
        "aoi_id": str(metadata.get("aoi_id", "unknown")),
        "budget": budget,
        "target_operation": target_operation,
        "candidate_count": len(candidates),
        "distinct_outcome_count": len(signatures),
        "target_candidate_count": target_candidate_count,
        "target_candidate_covered": target_candidate_count > 0,
        "safe_candidate_count": len(safe_candidates),
        "safe_map_headroom": safe_headroom,
        "safe_map_recovery_possible": safe_headroom > 1e-6,
        "missed_edit_recovery_possible": bool_value(anchor.get("missed_edit"))
        and not safe_oracle["missed_edit"],
        "safe_oracle_evidence_id": safe_oracle["evidence_id"],
        "safe_oracle_predicted_operation": safe_oracle["predicted_operation"],
        "stop_utility": stop_utility,
        "best_visible_evidence_id": best_visible_id,
        "best_visible_utility": best_visible_utility,
        "oracle_acquire": best_visible_utility > stop_utility,
        "visible_utilities": dict(zip(evidence_ids, utilities, strict=True)),
        "candidate_raster_iou_gains": candidate_raster_iou_gains,
    }


def compact_rollout(row: dict[str, Any], *, source: Path, line_number: int) -> dict[str, Any]:
    _require_validation(row, source=source, line_number=line_number)
    source_episode = str(row.get("source_episode"))
    budget = float(row.get("budget"))
    return {
        "key": state_key(source_episode, budget),
        "task_id": str(row.get("task_id")),
        "source_episode": source_episode,
        "aoi_id": str(row.get("aoi_id", "unknown")),
        "budget": budget,
        "target_operation": operation(row.get("target")),
        "policy_acquire": bool_value(row.get("selected_extra_evidence")),
        "selected_evidence_id": (
            str(row.get("selected_extra_evidence_id"))
            if row.get("selected_extra_evidence_id") is not None
            else None
        ),
        "prediction": str(row.get("prediction")),
        "spent_cost": float(row.get("spent_cost", 0.0)),
    }


def compact_writeback(row: dict[str, Any], *, source: Path, line_number: int) -> dict[str, Any]:
    _require_validation(row, source=source, line_number=line_number)
    task_id = str(row.get("task_id"))
    budget = float(row.get("budget"))
    return {
        "key": (task_id, round(budget, 6)),
        "task_id": task_id,
        "budget": budget,
        "target_operation": operation(row.get("target")),
        "predicted_operation": operation(row.get("operation", row.get("prediction"))),
        "effective_operation": operation(row.get("effective_operation")),
        "writeback_changed": bool_value(row.get("writeback_changed")),
        "raster_iou_gain": float(row.get("raster_iou_gain", 0.0)),
        "false_edit": bool_value(row.get("false_edit")),
        "missed_edit": bool_value(row.get("missed_edit")),
        "wrong_edit": bool_value(row.get("wrong_edit")),
        "safe_commit_applied": bool_value(row.get("safe_commit_applied")),
        "safe_commit_accepted": bool_value(row.get("safe_commit_accepted")),
        "safe_commit_rejection_reason": row.get("safe_commit_rejection_reason"),
    }


def _unique_index(
    rows: Iterable[dict[str, Any]], key_name: str, *, source: Path
) -> dict[Any, dict[str, Any]]:
    index: dict[Any, dict[str, Any]] = {}
    for row in rows:
        key = row[key_name]
        if key in index:
            raise ValueError(f"{source}: duplicate {key_name}={key!r}")
        index[key] = row
    return index


def load_bundle(bundle: Bundle) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    state_rows, states_sha, state_lines = _stream_jsonl(
        bundle.states,
        lambda row, line: compact_state(
            row, source=bundle.states, line_number=line
        ),
    )
    rollout_rows, rollout_sha, rollout_lines = _stream_jsonl(
        bundle.rollout,
        lambda row, line: compact_rollout(
            row, source=bundle.rollout, line_number=line
        ),
    )
    raw_rows, raw_sha, raw_lines = _stream_jsonl(
        bundle.raw_writeback,
        lambda row, line: compact_writeback(
            row, source=bundle.raw_writeback, line_number=line
        ),
    )
    safe_rows, safe_sha, safe_lines = _stream_jsonl(
        bundle.safe_writeback,
        lambda row, line: compact_writeback(
            row, source=bundle.safe_writeback, line_number=line
        ),
    )
    states = _unique_index(state_rows, "key", source=bundle.states)
    rollouts = _unique_index(rollout_rows, "key", source=bundle.rollout)
    raw = _unique_index(raw_rows, "key", source=bundle.raw_writeback)
    safe = _unique_index(safe_rows, "key", source=bundle.safe_writeback)
    if set(states) != set(rollouts):
        raise ValueError(
            f"seed {bundle.seed}: state/rollout keys differ "
            f"({len(states)} states, {len(rollouts)} rollouts)"
        )
    if set(raw) != set(safe):
        raise ValueError(
            f"seed {bundle.seed}: raw/safe task ids differ "
            f"({len(raw)} raw, {len(safe)} safe)"
        )

    joined = []
    for key, state in states.items():
        rollout = rollouts[key]
        task_id = rollout["task_id"]
        writeback_key = (task_id, round(float(rollout["budget"]), 6))
        if writeback_key not in raw or writeback_key not in safe:
            raise ValueError(
                f"seed {bundle.seed}: missing writeback for {writeback_key}"
            )
        raw_row = raw[writeback_key]
        safe_row = safe[writeback_key]
        operations = {
            state["target_operation"],
            rollout["target_operation"],
            raw_row["target_operation"],
            safe_row["target_operation"],
        }
        if len(operations) != 1:
            raise ValueError(f"seed {bundle.seed}: target mismatch for {task_id}")
        selected_id = rollout["selected_evidence_id"]
        selected_utility = (
            state["visible_utilities"].get(selected_id)
            if selected_id is not None
            else None
        )
        selected_expected_raster_gain = (
            state["candidate_raster_iou_gains"].get(selected_id)
            if selected_id is not None
            else None
        )
        if rollout["policy_acquire"] and selected_utility is None:
            raise ValueError(
                f"seed {bundle.seed}: selected evidence absent from state for {task_id}"
            )
        joined.append(
            {
                "seed": bundle.seed,
                "task_id": task_id,
                "source_episode": state["source_episode"],
                "aoi_id": state["aoi_id"],
                "budget": state["budget"],
                "target_operation": state["target_operation"],
                "candidate_count": state["candidate_count"],
                "distinct_outcome_count": state["distinct_outcome_count"],
                "target_candidate_count": state["target_candidate_count"],
                "target_candidate_covered": state["target_candidate_covered"],
                "safe_candidate_count": state["safe_candidate_count"],
                "safe_map_headroom": state["safe_map_headroom"],
                "safe_map_recovery_possible": state["safe_map_recovery_possible"],
                "missed_edit_recovery_possible": state[
                    "missed_edit_recovery_possible"
                ],
                "safe_oracle_predicted_operation": state[
                    "safe_oracle_predicted_operation"
                ],
                "oracle_acquire": state["oracle_acquire"],
                "best_visible_utility_gain": state["best_visible_utility"]
                - state["stop_utility"],
                "policy_acquire": rollout["policy_acquire"],
                "selected_utility_gain": (
                    selected_utility - state["stop_utility"]
                    if selected_utility is not None
                    else 0.0
                ),
                "selected_expected_raster_gain": (
                    selected_expected_raster_gain
                    if selected_expected_raster_gain is not None
                    else 0.0
                ),
                "policy_true_call": bool(
                    rollout["policy_acquire"] and state["oracle_acquire"]
                ),
                "policy_false_call": bool(
                    rollout["policy_acquire"] and not state["oracle_acquire"]
                ),
                "policy_harmful_call": bool(
                    rollout["policy_acquire"]
                    and selected_utility is not None
                    and selected_utility <= state["stop_utility"]
                ),
                "exact_oracle_candidate": bool(
                    rollout["policy_acquire"]
                    and selected_id == state["best_visible_evidence_id"]
                ),
                "spent_cost": rollout["spent_cost"],
                "raw_predicted_operation": raw_row["predicted_operation"],
                "raw_effective_operation": raw_row["effective_operation"],
                "raw_writeback_changed": raw_row["writeback_changed"],
                "raw_raster_iou_gain": raw_row["raster_iou_gain"],
                "raw_false_edit": raw_row["false_edit"],
                "raw_missed_edit": raw_row["missed_edit"],
                "raw_wrong_edit": raw_row["wrong_edit"],
                "safe_commit_accepted": safe_row["safe_commit_accepted"],
                "safe_commit_rejection_reason": safe_row[
                    "safe_commit_rejection_reason"
                ],
                "safe_effective_operation": safe_row["effective_operation"],
                "safe_writeback_changed": safe_row["writeback_changed"],
                "safe_raster_iou_gain": safe_row["raster_iou_gain"],
                "safe_false_edit": safe_row["false_edit"],
                "safe_missed_edit": safe_row["missed_edit"],
                "safe_wrong_edit": safe_row["wrong_edit"],
                "test_assets_read": False,
            }
        )
    receipt = {
        "seed": bundle.seed,
        "states": str(bundle.states.resolve()),
        "states_sha256": states_sha,
        "state_line_count": state_lines,
        "initial_state_count": len(states),
        "rollout": str(bundle.rollout.resolve()),
        "rollout_sha256": rollout_sha,
        "rollout_line_count": rollout_lines,
        "raw_writeback": str(bundle.raw_writeback.resolve()),
        "raw_writeback_sha256": raw_sha,
        "raw_writeback_line_count": raw_lines,
        "safe_writeback": str(bundle.safe_writeback.resolve()),
        "safe_writeback_sha256": safe_sha,
        "safe_writeback_line_count": safe_lines,
        "joined_state_count": len(joined),
    }
    return joined, receipt


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    if not rows:
        return 0.0
    return float(statistics.fmean(float(row[field]) for row in rows))


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "decision_state_count": 0,
            "source_episode_count": 0,
        }
    source_rows: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        key = int(row["seed"]), str(row["source_episode"])
        previous = source_rows.setdefault(key, row)
        stable_fields = (
            "target_operation",
            "candidate_count",
            "distinct_outcome_count",
            "target_candidate_count",
            "safe_candidate_count",
            "safe_map_headroom",
            "safe_map_recovery_possible",
            "missed_edit_recovery_possible",
        )
        if any(previous[field] != row[field] for field in stable_fields):
            raise ValueError(f"candidate bank varies across budgets for {key}")
    candidate_rows = list(source_rows.values())
    calls = [row for row in rows if row["policy_acquire"]]
    oracle_calls = [row for row in rows if row["oracle_acquire"]]
    return {
        "decision_state_count": len(rows),
        "source_episode_count": len(candidate_rows),
        "aoi_count": len({str(row["aoi_id"]) for row in rows}),
        "mean_candidate_count": _mean(candidate_rows, "candidate_count"),
        "candidate_outcome_diversity_fraction": _mean(
            [
                {"value": int(row["distinct_outcome_count"] > 1)}
                for row in candidate_rows
            ],
            "value",
        ),
        "target_operation_candidate_coverage": _mean(
            candidate_rows, "target_candidate_covered"
        ),
        "mean_safe_map_headroom": _mean(candidate_rows, "safe_map_headroom"),
        "safe_map_recovery_fraction": _mean(
            candidate_rows, "safe_map_recovery_possible"
        ),
        "missed_edit_recovery_fraction": _mean(
            candidate_rows, "missed_edit_recovery_possible"
        ),
        "oracle_acquire_count": len(oracle_calls),
        "oracle_acquire_rate": len(oracle_calls) / len(rows),
        "policy_acquire_count": len(calls),
        "policy_acquire_rate": len(calls) / len(rows),
        "policy_acquire_recall": sum(row["policy_true_call"] for row in rows)
        / max(len(oracle_calls), 1),
        "policy_false_call_rate": sum(row["policy_false_call"] for row in rows)
        / len(rows),
        "policy_harmful_call_fraction": sum(
            row["policy_harmful_call"] for row in rows
        )
        / max(len(calls), 1),
        "exact_oracle_candidate_recall": sum(
            row["exact_oracle_candidate"] for row in rows
        )
        / max(len(oracle_calls), 1),
        "raw_writeback_change_rate": _mean(rows, "raw_writeback_changed"),
        "raw_positive_gain_rate": sum(row["raw_raster_iou_gain"] > 1e-6 for row in rows)
        / len(rows),
        "raw_mean_raster_iou_gain": _mean(rows, "raw_raster_iou_gain"),
        "raw_positive_gain_given_acquire": sum(
            row["raw_raster_iou_gain"] > 1e-6 for row in calls
        )
        / max(len(calls), 1),
        "raw_mean_gain_given_acquire": _mean(calls, "raw_raster_iou_gain"),
        "safe_commit_accept_rate": _mean(rows, "safe_commit_accepted"),
        "safe_commit_accept_given_acquire": _mean(calls, "safe_commit_accepted"),
        "safe_writeback_change_rate": _mean(rows, "safe_writeback_changed"),
        "safe_positive_gain_rate": sum(
            row["safe_raster_iou_gain"] > 1e-6 for row in rows
        )
        / len(rows),
        "safe_mean_raster_iou_gain": _mean(rows, "safe_raster_iou_gain"),
        "safe_positive_gain_given_acquire": sum(
            row["safe_raster_iou_gain"] > 1e-6 for row in calls
        )
        / max(len(calls), 1),
        "safe_mean_gain_given_acquire": _mean(calls, "safe_raster_iou_gain"),
        "raw_false_edit_rate": _mean(rows, "raw_false_edit"),
        "safe_false_edit_rate": _mean(rows, "safe_false_edit"),
        "raw_missed_edit_rate": _mean(rows, "raw_missed_edit"),
        "safe_missed_edit_rate": _mean(rows, "safe_missed_edit"),
    }


def bottleneck_flags(
    summary: dict[str, Any],
    *,
    minimum_candidate_recovery: float,
    minimum_label_rate: float,
    minimum_policy_recall: float,
    minimum_conversion: float,
) -> dict[str, bool]:
    if not summary.get("decision_state_count"):
        return {"missing_support": True}
    candidate_weak = (
        float(summary["safe_map_recovery_fraction"]) < minimum_candidate_recovery
        or float(summary["mean_safe_map_headroom"]) <= 1e-6
    )
    label_sparse = (
        not candidate_weak
        and float(summary["oracle_acquire_rate"]) < minimum_label_rate
    )
    policy_weak = (
        not candidate_weak
        and not label_sparse
        and float(summary["policy_acquire_recall"]) < minimum_policy_recall
    )
    raw_conversion_weak = (
        int(summary["policy_acquire_count"]) > 0
        and float(summary["raw_positive_gain_given_acquire"]) < minimum_conversion
    )
    safe_commit_blocking = (
        float(summary["raw_positive_gain_given_acquire"])
        > float(summary["safe_positive_gain_given_acquire"]) + 1e-9
    )
    return {
        "candidate_interface_weak": candidate_weak,
        "oracle_label_support_sparse": label_sparse,
        "policy_acquire_recall_weak": policy_weak,
        "raw_writeback_conversion_weak": raw_conversion_weak,
        "safe_commit_blocks_positive_raw_gain": safe_commit_blocking,
    }


def parse_bundle(value: str) -> Bundle:
    parts = value.split(":", 4)
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            "bundle must be seed:states:rollout:raw_writeback:safe_writeback"
        )
    seed, *paths = parts
    try:
        parsed_seed = int(seed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("bundle seed must be an integer") from exc
    bundle = Bundle(parsed_seed, *(Path(path) for path in paths))
    for path in (bundle.states, bundle.rollout, bundle.raw_writeback, bundle.safe_writeback):
        if not path.is_file():
            raise argparse.ArgumentTypeError(f"bundle input does not exist: {path}")
    return bundle


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--bundle", action="append", type=parse_bundle, required=True)
    parser.add_argument("--minimum-candidate-recovery", type=float, default=0.10)
    parser.add_argument("--minimum-label-rate", type=float, default=0.05)
    parser.add_argument("--minimum-policy-recall", type=float, default=0.50)
    parser.add_argument("--minimum-conversion", type=float, default=0.10)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if len({bundle.seed for bundle in args.bundle}) != len(args.bundle):
        raise ValueError("bundle seeds must be unique")
    thresholds = {
        "minimum_candidate_recovery": args.minimum_candidate_recovery,
        "minimum_label_rate": args.minimum_label_rate,
        "minimum_policy_recall": args.minimum_policy_recall,
        "minimum_conversion": args.minimum_conversion,
    }
    if any(not 0.0 <= value <= 1.0 for value in thresholds.values()):
        raise ValueError("diagnostic thresholds must be between zero and one")

    all_rows: list[dict[str, Any]] = []
    receipts = []
    for bundle in args.bundle:
        rows, receipt = load_bundle(bundle)
        all_rows.extend(rows)
        receipts.append(receipt)

    aggregate_rows = []
    summaries: dict[str, Any] = {}
    for operation_name in OPERATIONS:
        operation_rows = [
            row for row in all_rows if row["target_operation"] == operation_name
        ]
        summary = summarize_rows(operation_rows)
        summary["bottleneck_flags"] = bottleneck_flags(summary, **thresholds)
        summaries[operation_name] = summary
        aggregate_rows.append(
            {
                "scope": "all_seeds",
                "seed": "all",
                "operation": operation_name,
                **{key: value for key, value in summary.items() if key != "bottleneck_flags"},
                **summary["bottleneck_flags"],
            }
        )
        for seed in sorted({int(row["seed"]) for row in operation_rows}):
            seed_summary = summarize_rows(
                [row for row in operation_rows if int(row["seed"]) == seed]
            )
            aggregate_rows.append(
                {
                    "scope": "seed",
                    "seed": seed,
                    "operation": operation_name,
                    **seed_summary,
                    **bottleneck_flags(seed_summary, **thresholds),
                }
            )

    args.output_dir.mkdir(parents=True)
    with gzip.open(
        args.output_dir / "per_state.jsonl.gz", "xt", encoding="utf-8"
    ) as handle:
        for row in all_rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    _write_csv(args.output_dir / "operation_layers.csv", aggregate_rows)
    result = {
        "schema_version": "sn7-operation-support-layers-v1",
        "split": "val",
        "test_assets_read": False,
        "seed_count": len(args.bundle),
        "seeds": [bundle.seed for bundle in args.bundle],
        "joined_state_count": len(all_rows),
        "thresholds": thresholds,
        "operations": summaries,
        "input_receipts": receipts,
        "interpretation": (
            "Diagnostic only. Candidate and label fields use frozen executable "
            "counterfactuals; policy and writeback fields use model-selected "
            "validation rollouts. Flags localize support loss and are not paper claims."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
