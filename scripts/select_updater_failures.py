#!/usr/bin/env python3
"""Write a ranked failure-case manifest from frozen updater predictions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from activemap.evaluation.failure_cases import select_failure_cases
from activemap.evaluation.update import load_update_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--per-category", type=int, default=12)
    args = parser.parse_args()
    records = load_update_predictions(args.predictions)
    cases = select_failure_cases(records, per_category=args.per_category)
    payload = {
        "predictions": str(args.predictions),
        "prediction_count": len(records),
        "per_category": args.per_category,
        "selected_count": len(cases),
        "category_counts": dict(Counter(case["failure_category"] for case in cases)),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["category_counts"], indent=2))


if __name__ == "__main__":
    main()
