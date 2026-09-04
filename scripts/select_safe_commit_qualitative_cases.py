#!/usr/bin/env python3
"""Select reproducible Safe Commit cases from out-of-fold traces."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

Row = dict[str, Any]


def _read_jsonl(path: Path) -> list[Row]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _gate_margin(row: Row) -> float:
    gate = row.get("safe_commit") or {}
    return abs(float(gate.get("observed_value", 0.0)) - float(gate.get("threshold", 0.0)))


def _select_diverse(
    rows: list[Row], predicate: Callable[[Row], bool], limit: int
) -> list[Row]:
    candidates = sorted(
        (row for row in rows if predicate(row)),
        key=lambda row: (_gate_margin(row), str(row.get("sample_id", ""))),
        reverse=True,
    )
    selected: list[Row] = []
    used_aois: set[str] = set()
    used_targets: set[str] = set()
    for require_new_target in (True, False):
        for row in candidates:
            aoi = str(row.get("aoi_id", ""))
            target = str(row.get("target_edit", ""))
            if aoi in used_aois or (require_new_target and target in used_targets):
                continue
            selected.append(row)
            used_aois.add(aoi)
            used_targets.add(target)
            if len(selected) == limit:
                return selected
    return selected


def select_cases(rows: list[Row], per_category: int) -> list[tuple[str, Row]]:
    categories: tuple[tuple[str, Callable[[Row], bool]], ...] = (
        (
            "corrected_false_edit",
            lambda row: bool((row.get("safe_commit") or {}).get("terminal_suppressed"))
            and bool(row.get("terminal_correct"))
            and str(row.get("target_edit")) == "KEEP",
        ),
        (
            "preserved_correct_edit",
            lambda row: not bool((row.get("safe_commit") or {}).get("terminal_suppressed"))
            and bool(row.get("terminal_correct"))
            and str(row.get("target_edit")) != "KEEP",
        ),
        (
            "residual_false_edit",
            lambda row: not bool((row.get("safe_commit") or {}).get("terminal_suppressed"))
            and bool(row.get("false_edit")),
        ),
        (
            "induced_missed_edit",
            lambda row: bool((row.get("safe_commit") or {}).get("terminal_suppressed"))
            and bool(row.get("missed_edit")),
        ),
    )
    selected: list[tuple[str, Row]] = []
    used_ids: set[str] = set()
    for category, predicate in categories:
        for row in _select_diverse(rows, predicate, per_category):
            sample_id = str(row["sample_id"])
            if sample_id not in used_ids:
                selected.append((category, row))
                used_ids.add(sample_id)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--per-category", type=int, default=3)
    args = parser.parse_args()
    if args.per_category <= 0:
        raise ValueError("--per-category must be positive")

    rows = [row for path in args.traces for row in _read_jsonl(path)]
    selected = select_cases(rows, args.per_category)
    if not selected:
        raise ValueError("no qualifying Safe Commit cases found")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    traces_path = args.output_dir / "selected_traces.jsonl"
    traces_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for _, row in selected),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "safe-commit-qualitative-selection-v1",
        "selection_source": "out_of_fold_validation_only",
        "selection_rule": "largest absolute gate margin with AOI and edit-type diversity",
        "per_category_limit": args.per_category,
        "cases": [
            {
                "category": category,
                "sample_id": row["sample_id"],
                "aoi_id": row.get("aoi_id"),
                "target_edit": row.get("target_edit"),
                "prediction": row.get("predicted_edit"),
                "original_prediction": (row.get("safe_commit") or {}).get(
                    "original_prediction"
                ),
                "gate_margin": _gate_margin(row),
            }
            for category, row in selected
        ],
    }
    (args.output_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output_dir), "cases": len(selected)}))


if __name__ == "__main__":
    main()
