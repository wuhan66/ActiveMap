#!/usr/bin/env python3
"""Summarize validation utility across predeclared structured-head thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--thresholds", default="-0.012187,-0.01,-0.005,0,0.005,0.01")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    rows = [
        json.loads(line)
        for line in args.predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = []
    for threshold in (float(value) for value in args.thresholds.split(",")):
        called = [row for row in rows if float(row["predicted_utility"]) >= threshold]
        utilities = [
            float(row["policy_relative_realized_advantage"]) for row in called
        ]
        positives = sum(value > 0.0 for value in utilities)
        result.append(
            {
                "threshold": threshold,
                "calls": len(called),
                "call_rate": len(called) / len(rows),
                "precision": positives / max(len(called), 1),
                "utility_sum": sum(utilities),
                "utility_mean_all_states": sum(utilities) / len(rows),
                "risk_sum": sum(max(-value, 0.0) for value in utilities),
            }
        )
    payload = {
        "schema_version": "active-catalog-vla-threshold-analysis-v1",
        "sample_count": len(rows),
        "grid": result,
        "test_assets_read": False,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
