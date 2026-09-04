#!/usr/bin/env python3
"""Reweight selector utilities for a predeclared evidence-cost protocol."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def reweight_record(record: dict[str, Any], target_weight: float) -> dict[str, Any]:
    metadata = dict(record.get("metadata", {}))
    source_weight = float(metadata["cost_weight"])
    utilities = list(record["oracle_utilities"])
    costs = list(record["evidence_costs"])
    if len(utilities) != len(costs):
        raise ValueError("oracle utility and evidence cost lengths differ")
    record = dict(record)
    record["oracle_utilities"] = [
        float(utility) + (source_weight - target_weight) * float(cost)
        for utility, cost in zip(utilities, costs, strict=True)
    ]
    metadata.update(
        {
            "source_cost_weight": source_weight,
            "cost_weight": target_weight,
            "utility_reweight_formula": "U_new=U_old+(w_old-w_new)*cost",
        }
    )
    record["metadata"] = metadata
    return record


def should_acquire(record: dict[str, Any]) -> bool:
    return max(float(value) for value in record["oracle_utilities"]) > float(
        record.get("stop_utility", 0.0)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--cost-weight", type=float, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    if args.cost_weight < 0:
        raise ValueError("cost weight must be non-negative")
    if args.destination.exists():
        raise FileExistsError(f"refusing to overwrite {args.destination}")
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.destination.with_suffix(args.destination.suffix + ".tmp")
    counts: Counter[str] = Counter()
    source_weights: Counter[str] = Counter()
    try:
        with args.source.open("r", encoding="utf-8") as source, temporary.open(
            "x", encoding="utf-8"
        ) as destination:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                split = str(record["split"])
                counts[f"{split}_states"] += 1
                counts[f"{split}_acquire_before"] += int(should_acquire(record))
                source_weights[str(record["metadata"]["cost_weight"])] += 1
                adjusted = reweight_record(record, args.cost_weight)
                counts[f"{split}_acquire_after"] += int(should_acquire(adjusted))
                destination.write(
                    json.dumps(adjusted, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
        temporary.replace(args.destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    splits = sorted(key.removesuffix("_states") for key in counts if key.endswith("_states"))
    summary = {
        "schema_version": "selector-utility-reweight-v1",
        "source": str(args.source.resolve()),
        "destination": str(args.destination.resolve()),
        "target_cost_weight": args.cost_weight,
        "source_weight_counts": dict(source_weights),
        "test_assets_read": False,
        "splits": {
            split: {
                "states": counts[f"{split}_states"],
                "acquire_before": counts[f"{split}_acquire_before"],
                "acquire_after": counts[f"{split}_acquire_after"],
                "acquire_fraction_before": counts[f"{split}_acquire_before"]
                / counts[f"{split}_states"],
                "acquire_fraction_after": counts[f"{split}_acquire_after"]
                / counts[f"{split}_states"],
            }
            for split in splits
        },
    }
    summary_path = args.summary or args.destination.with_suffix(".reweight_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
