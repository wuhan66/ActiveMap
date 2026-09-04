#!/usr/bin/env python3
"""Aggregate promoted operation-selector validation results across seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from activemap.evaluation.operation_selector_promotion import (
    aggregate_operation_selector_seeds,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--promotion", action="append", required=True, metavar="SEED=JSON")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    promotions = {}
    for item in args.promotion:
        seed, separator, raw_path = item.partition("=")
        if not separator or not seed or not raw_path:
            parser.error(f"invalid --promotion value: {item}")
        promotions[seed] = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    summary = aggregate_operation_selector_seeds(promotions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
