#!/usr/bin/env python3
"""Audit how counterfactual evidence value changes across updater versions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def read_rows(
    path: Path, *, oracle_step: int | None
) -> tuple[dict[str, dict[str, Any]], str]:
    """Read either cached VLM branches or selector-oracle state records.

    Selector-oracle files are expanded to one counterfactual-value item per
    available evidence candidate. This keeps staleness at the decision point,
    rather than hiding it behind an episode-level maximum.
    """
    rows: dict[str, dict[str, Any]] = {}
    schema: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        if "policy_relative_advantage" in row:
            row_schema = "policy-relative-trace"
            candidates = [(str(row["example_id"]), float(row["policy_relative_advantage"]))]
        elif {"sample_id", "evidence_ids", "oracle_utilities", "evidence_costs", "false_edit_risks"} <= row.keys():
            row_schema = "selector-oracle-state"
            observed_step = int(row.get("metadata", {}).get("oracle_step", 0))
            if oracle_step is not None and observed_step != oracle_step:
                continue
            lengths = {
                len(row["evidence_ids"]),
                len(row["oracle_utilities"]),
                len(row["evidence_costs"]),
                len(row["false_edit_risks"]),
            }
            if len(lengths) != 1:
                raise ValueError(f"inconsistent candidate lengths in {path}: {row['sample_id']}")
            candidates = [
                (
                    f"{row['sample_id']}::{evidence_id}",
                    float(utility) - float(row.get("stop_utility", 0.0)),
                )
                for evidence_id, utility in zip(row["evidence_ids"], row["oracle_utilities"], strict=True)
            ]
        else:
            raise ValueError(f"unsupported record schema in {path}")
        if schema is None:
            schema = row_schema
        elif schema != row_schema:
            raise ValueError(f"mixed record schemas in {path}")
        for example_id, advantage in candidates:
            if example_id in rows:
                raise ValueError(f"duplicate candidate ID in {path}: {example_id}")
            if not np.isfinite(advantage):
                raise ValueError(f"non-finite counterfactual value in {path}: {example_id}")
            rows[example_id] = {"policy_relative_advantage": advantage}
    if not rows:
        raise ValueError(f"empty trace file: {path}")
    return rows, str(schema)


def markdown(summary: dict[str, Any]) -> str:
    transition = summary["label_transitions"]
    return "\n".join(
        [
            "# Policy-Relative Staleness Audit",
            "",
            "| Quantity | Value |",
            "| --- | ---: |",
            f"| Matched candidate edits | {summary['matched_examples']} |",
            f"| Old positive-value rate | {summary['old_positive_rate']:.6f} |",
            f"| Refreshed positive-value rate | {summary['new_positive_rate']:.6f} |",
            f"| Positive-set Jaccard | {summary['positive_jaccard']:.6f} |",
            f"| Utility-value correlation | {summary['advantage_correlation']:.6f} |",
            f"| Old positive -> refreshed non-positive | {transition['positive_to_nonpositive']} |",
            f"| Old non-positive -> refreshed positive | {transition['nonpositive_to_positive']} |",
            f"| Unchanged sign | {transition['unchanged']} |",
            "",
            "A non-trivial sign transition rate establishes that a selector trained on the",
            "old updater's counterfactual values is stale for the refreshed updater. This",
            "audit is descriptive; final claims require the matched executable policy",
            "matrix and paired task/AOI bootstrap.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-traces", type=Path, required=True)
    parser.add_argument("--new-traces", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--oracle-step",
        type=int,
        default=None,
        help="For selector-oracle states, audit only a shared sequential step.",
    )
    args = parser.parse_args()
    if args.oracle_step is not None and args.oracle_step < 0:
        raise ValueError("oracle step must be non-negative")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    old_rows, old_schema = read_rows(args.old_traces, oracle_step=args.oracle_step)
    new_rows, new_schema = read_rows(args.new_traces, oracle_step=args.oracle_step)
    if old_schema != new_schema:
        raise ValueError(f"record schemas differ: old={old_schema}, new={new_schema}")
    if old_rows.keys() != new_rows.keys():
        missing_old = sorted(new_rows.keys() - old_rows.keys())[:5]
        missing_new = sorted(old_rows.keys() - new_rows.keys())[:5]
        raise ValueError(
            "trace candidate sets differ; "
            f"missing_from_old={missing_old}, missing_from_new={missing_new}"
        )

    ids = sorted(old_rows)
    old_values = np.asarray(
        [float(old_rows[example_id]["policy_relative_advantage"]) for example_id in ids]
    )
    new_values = np.asarray(
        [float(new_rows[example_id]["policy_relative_advantage"]) for example_id in ids]
    )
    old_positive = old_values > 0.0
    new_positive = new_values > 0.0
    union = np.logical_or(old_positive, new_positive).sum()
    correlation = 1.0
    if len(ids) > 1 and np.std(old_values) and np.std(new_values):
        correlation = float(np.corrcoef(old_values, new_values)[0, 1])

    summary = {
        "schema_version": "policy-relative-staleness-audit-v1",
        "old_traces": str(args.old_traces.resolve()),
        "new_traces": str(args.new_traces.resolve()),
        "record_schema": old_schema,
        "oracle_step_filter": args.oracle_step,
        "matched_examples": len(ids),
        "old_positive_rate": float(old_positive.mean()),
        "new_positive_rate": float(new_positive.mean()),
        "positive_jaccard": float(np.logical_and(old_positive, new_positive).sum() / union)
        if union
        else 1.0,
        "advantage_correlation": correlation,
        "label_transitions": {
            "positive_to_nonpositive": int(np.logical_and(old_positive, ~new_positive).sum()),
            "nonpositive_to_positive": int(np.logical_and(~old_positive, new_positive).sum()),
            "unchanged": int((old_positive == new_positive).sum()),
        },
        "test_assets_read": False,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "SUMMARY.md").write_text(markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
