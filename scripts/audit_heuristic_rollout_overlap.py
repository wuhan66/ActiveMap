#!/usr/bin/env python3
"""Audit whether nominally different rollout heuristics select the same evidence."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {
    "task_id",
    "budget",
    "prediction",
    "acquisitions",
    "selected_evidence_ids",
}


def _load(path: Path) -> dict[tuple[str, float], dict[str, Any]]:
    rows: dict[tuple[str, float], dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = sorted(REQUIRED_FIELDS - row.keys())
        if missing:
            raise ValueError(f"{path}:{line_number} missing fields: {missing}")
        key = (str(row["task_id"]), float(row["budget"]))
        if key in rows:
            raise ValueError(f"{path}:{line_number} duplicate rollout key: {key}")
        selected = [str(value) for value in row["selected_evidence_ids"]]
        acquisitions = int(row["acquisitions"])
        if acquisitions < 0 or acquisitions > len(selected):
            raise ValueError(
                f"{path}:{line_number} invalid acquisitions={acquisitions} "
                f"for {len(selected)} selected items"
            )
        row["_acquired_sequence"] = tuple(selected[-acquisitions:]) if acquisitions else ()
        rows[key] = row
    if not rows:
        raise ValueError(f"no rollout rows in {path}")
    return rows


def _jaccard(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def audit_rollout_overlap(
    method_paths: dict[str, Path],
    *,
    primary_budget: float = 3.0,
    collapse_threshold: float = 0.95,
) -> dict[str, Any]:
    if len(method_paths) < 2:
        raise ValueError("at least two methods are required")
    methods = {name: _load(path) for name, path in sorted(method_paths.items())}
    first_name = next(iter(methods))
    expected_keys = methods[first_name].keys()
    for name, rows in methods.items():
        if rows.keys() != expected_keys:
            missing = sorted(expected_keys - rows.keys())[:10]
            extra = sorted(rows.keys() - expected_keys)[:10]
            raise ValueError(f"{name} rollout keys differ; missing={missing}, extra={extra}")

    budgets = sorted({key[1] for key in expected_keys})
    pairwise: list[dict[str, Any]] = []
    for left_name, right_name in itertools.combinations(methods, 2):
        for budget in budgets:
            keys = sorted(key for key in expected_keys if key[1] == budget)
            left_rows, right_rows = methods[left_name], methods[right_name]
            exact = [
                left_rows[key]["_acquired_sequence"] == right_rows[key]["_acquired_sequence"]
                for key in keys
            ]
            count_agreement = [
                int(left_rows[key]["acquisitions"]) == int(right_rows[key]["acquisitions"])
                for key in keys
            ]
            prediction_agreement = [
                str(left_rows[key]["prediction"]) == str(right_rows[key]["prediction"])
                for key in keys
            ]
            jaccards = [
                _jaccard(
                    left_rows[key]["_acquired_sequence"],
                    right_rows[key]["_acquired_sequence"],
                )
                for key in keys
            ]
            pairwise.append(
                {
                    "method_a": left_name,
                    "method_b": right_name,
                    "budget": budget,
                    "sample_count": len(keys),
                    "acquisition_sequence_agreement": sum(exact) / len(exact),
                    "acquisition_count_agreement": sum(count_agreement) / len(count_agreement),
                    "acquisition_set_jaccard": sum(jaccards) / len(jaccards),
                    "prediction_agreement": sum(prediction_agreement) / len(prediction_agreement),
                }
            )

    all_method_agreement = []
    for budget in budgets:
        keys = sorted(key for key in expected_keys if key[1] == budget)
        agreed = 0
        for key in keys:
            sequences = {rows[key]["_acquired_sequence"] for rows in methods.values()}
            agreed += len(sequences) == 1
        all_method_agreement.append(
            {
                "budget": budget,
                "sample_count": len(keys),
                "all_method_acquisition_sequence_agreement": agreed / len(keys),
            }
        )

    warnings = [
        {
            "type": "strategy_selection_collapse",
            "method_a": row["method_a"],
            "method_b": row["method_b"],
            "budget": row["budget"],
            "acquisition_sequence_agreement": row["acquisition_sequence_agreement"],
            "threshold": collapse_threshold,
        }
        for row in pairwise
        if abs(float(row["budget"]) - primary_budget) < 1e-9
        and float(row["acquisition_sequence_agreement"]) >= collapse_threshold
    ]
    return {
        "schema_version": "activemap-heuristic-overlap-audit-v1",
        "methods": sorted(methods),
        "sample_count": len(expected_keys),
        "task_count": len({key[0] for key in expected_keys}),
        "budgets": budgets,
        "primary_budget": primary_budget,
        "collapse_threshold": collapse_threshold,
        "pairwise": pairwise,
        "all_method_agreement": all_method_agreement,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--methods", default="")
    parser.add_argument("--primary-budget", type=float, default=3.0)
    parser.add_argument("--collapse-threshold", type=float, default=0.95)
    args = parser.parse_args()

    requested = [value.strip() for value in args.methods.split(",") if value.strip()]
    paths = {
        path.stem: path
        for path in sorted(args.input_dir.glob("*.jsonl"))
        if not requested or path.stem in requested
    }
    result = audit_rollout_overlap(
        paths,
        primary_budget=args.primary_budget,
        collapse_threshold=args.collapse_threshold,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result["pairwise"][0]))
            writer.writeheader()
            writer.writerows(result["pairwise"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
