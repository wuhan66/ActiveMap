#!/usr/bin/env python3
"""Audit candidate-level counterfactual false-edit support for C5 risk learning."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError("empty state file")
    if any(str(row.get("split")) == "test" for row in rows):
        raise ValueError("C5 candidate-risk audit forbids test records")
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    groups: dict[str, set[str]] = defaultdict(set)
    risk_buckets: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        split = str(row["split"])
        groups[split].add(str(row["metadata"].get("source_episode", row["sample_id"])))
        outcomes = row["metadata"].get("executable_outcomes", {})
        for index, evidence_id in enumerate(row["evidence_ids"]):
            outcome = outcomes.get(evidence_id)
            if not isinstance(outcome, dict):
                raise ValueError(f"{row['sample_id']}: missing outcome for {evidence_id}")
            counts[split]["candidates"] += 1
            if bool(outcome["false_edit"]):
                counts[split]["false_edits"] += 1
            if bool(outcome["missed_edit"]):
                counts[split]["missed_edits"] += 1
            if float(outcome["quality_gain"]) > 0.0:
                counts[split]["positive_quality"] += 1
            risk = float(row["false_edit_risks"][index])
            bucket = f"{min(int(risk * 10), 9) / 10:.1f}-{min(int(risk * 10), 9) / 10 + 0.1:.1f}"
            risk_buckets[split][f"{bucket}:count"] += 1
            risk_buckets[split][f"{bucket}:false"] += int(bool(outcome["false_edit"]))
    overlap = groups.get("train", set()) & groups.get("val", set())
    return {
        "schema_version": "updater-conditioned-candidate-risk-audit-v1",
        "state_count": len(rows),
        "split_counts": {key: dict(value) for key, value in sorted(counts.items())},
        "group_counts": {key: len(value) for key, value in sorted(groups.items())},
        "cross_split_group_overlap": len(overlap),
        "risk_histogram": {key: dict(value) for key, value in sorted(risk_buckets.items())},
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("states", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    summary = summarize(_load(args.states))
    summary["states"] = str(args.states.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
