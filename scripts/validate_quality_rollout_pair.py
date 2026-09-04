#!/usr/bin/env python3
"""Fail fast unless paired rollouts support the frozen quality-cost protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {
    "task_id",
    "budget",
    "final_evidence_quality",
    "evidence_quality_gain",
    "quality_cost_utility",
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
        rows[key] = row
    if not rows:
        raise ValueError(f"no rollout rows in {path}")
    return rows


def validate_pair(baseline_path: Path, candidate_path: Path) -> dict[str, Any]:
    baseline = _load(baseline_path)
    candidate = _load(candidate_path)
    if baseline.keys() != candidate.keys():
        missing = sorted(baseline.keys() - candidate.keys())[:10]
        extra = sorted(candidate.keys() - baseline.keys())[:10]
        raise ValueError(f"paired rollout keys differ; missing={missing}, extra={extra}")
    return {
        "protocol": "quality-cost-v2",
        "paired": True,
        "sample_count": len(baseline),
        "task_count": len({key[0] for key in baseline}),
        "budgets": sorted({key[1] for key in baseline}),
        "required_fields": sorted(REQUIRED_FIELDS),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = validate_pair(args.baseline, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
