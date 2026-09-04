#!/usr/bin/env python3
"""Audit a seed-matched MUNO21 selector comparison without rerunning inference."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def key(row: dict[str, Any]) -> tuple[str, float]:
    return str(row["task_id"]), float(row["budget"])


def compact(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "prediction": row.get("prediction"),
        "target": row.get("target"),
        "terminal_correct": row.get("terminal_correct"),
        "false_edit": row.get("false_edit"),
        "missed_edit": row.get("missed_edit"),
        "wrong_edit": row.get("wrong_edit"),
        "acquisitions": row.get("acquisitions"),
        "spent_cost": row.get("spent_cost"),
        "selected_evidence_ids": row.get("selected_evidence_ids"),
        "writeback_changed": row.get("writeback_changed"),
        "effective_operation": row.get("effective_operation"),
        "raster_iou_gain": row.get("raster_iou_gain"),
        "added_polygon_iou": row.get("added_polygon_iou"),
        "removed_polygon_iou": row.get("removed_polygon_iou"),
        "component_count_absolute_error": row.get("component_count_absolute_error"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("generic_rollout", type=Path)
    parser.add_argument("candidate_rollout", type=Path)
    parser.add_argument("generic_writeback", type=Path)
    parser.add_argument("candidate_writeback", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    generic_rollout = {key(row): row for row in read_jsonl(args.generic_rollout)}
    candidate_rollout = {key(row): row for row in read_jsonl(args.candidate_rollout)}
    generic_writeback = {key(row): row for row in read_jsonl(args.generic_writeback)}
    candidate_writeback = {key(row): row for row in read_jsonl(args.candidate_writeback)}
    keys = sorted(set(generic_rollout) & set(candidate_rollout) & set(generic_writeback) & set(candidate_writeback))
    if not keys:
        raise ValueError("no complete paired task-budget records")

    rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for item in keys:
        generic_policy = generic_rollout[item]
        candidate_policy = candidate_rollout[item]
        generic_map = generic_writeback[item]
        candidate_map = candidate_writeback[item]
        action_changed = generic_policy.get("prediction") != candidate_policy.get("prediction")
        evidence_changed = generic_policy.get("selected_evidence_ids") != candidate_policy.get("selected_evidence_ids")
        false_edit_delta = int(bool(candidate_map.get("false_edit"))) - int(bool(generic_map.get("false_edit")))
        missed_edit_delta = int(bool(candidate_map.get("missed_edit"))) - int(bool(generic_map.get("missed_edit")))
        wrong_edit_delta = int(bool(candidate_map.get("wrong_edit"))) - int(bool(generic_map.get("wrong_edit")))
        iou_delta = float(candidate_map.get("raster_iou_gain", 0.0)) - float(generic_map.get("raster_iou_gain", 0.0))
        category = "unchanged"
        if false_edit_delta > 0:
            category = "introduced_false_edit"
        elif missed_edit_delta > 0:
            category = "introduced_missed_edit"
        elif wrong_edit_delta > 0:
            category = "introduced_wrong_edit"
        elif action_changed:
            category = "action_changed"
        elif evidence_changed:
            category = "evidence_changed"
        counters[category] += 1
        rows.append(
            {
                "task_id": item[0],
                "budget": item[1],
                "category": category,
                "action_changed": action_changed,
                "evidence_changed": evidence_changed,
                "false_edit_delta": false_edit_delta,
                "missed_edit_delta": missed_edit_delta,
                "wrong_edit_delta": wrong_edit_delta,
                "raster_iou_gain_delta": iou_delta,
                "generic_policy": compact(generic_policy),
                "candidate_policy": compact(candidate_policy),
                "generic_writeback": compact(generic_map),
                "candidate_writeback": compact(candidate_map),
            }
        )

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "paired_audit.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    harmful = [row for row in rows if row["category"].startswith("introduced_")]
    (args.output / "harmful_differences.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in harmful), encoding="utf-8"
    )
    harmful_transitions = Counter(
        " | ".join(
            (
                str(row["generic_policy"].get("target")),
                str(row["generic_policy"].get("prediction")),
                str(row["candidate_policy"].get("prediction")),
                str(row["candidate_writeback"].get("effective_operation")),
                row["category"],
            )
        )
        for row in harmful
    )
    harmful_summary = {
        "schema_version": "muno21-p50-paired-operation-audit-v1",
        "transitions": [
            {"transition": transition, "count": count}
            for transition, count in harmful_transitions.most_common()
        ],
        "examples": [
            {
                "task_id": row["task_id"],
                "budget": row["budget"],
                "category": row["category"],
                "target": row["generic_policy"].get("target"),
                "generic_prediction": row["generic_policy"].get("prediction"),
                "candidate_prediction": row["candidate_policy"].get("prediction"),
                "generic_evidence": row["generic_policy"].get("selected_evidence_ids"),
                "candidate_evidence": row["candidate_policy"].get("selected_evidence_ids"),
                "candidate_operation": row["candidate_writeback"].get("effective_operation"),
            }
            for row in harmful
        ],
    }
    (args.output / "harmful_summary.json").write_text(
        json.dumps(harmful_summary, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "schema_version": "muno21-p50-paired-operation-audit-v1",
        "paired_records": len(rows),
        "category_counts": dict(sorted(counters.items())),
        "harmful_difference_count": len(harmful),
        "mean_raster_iou_gain_delta": sum(row["raster_iou_gain_delta"] for row in rows) / len(rows),
        "test_assets_read": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
