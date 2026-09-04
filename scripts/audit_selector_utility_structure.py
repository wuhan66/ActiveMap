#!/usr/bin/env python3
"""Audit which observable catalog fields explain selector oracle utility."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def _candidate_year(evidence_id: str) -> str:
    marker = "__y"
    return evidence_id.rsplit(marker, 1)[-1] if marker in evidence_id else "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("states", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    candidate_variation: dict[str, list[int]] = defaultdict(list)
    with args.states.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            split = str(row["split"])
            edit_type = str(row["edit_type"])
            aoi = str(row["metadata"]["aoi_id"])
            features = row["evidence_features"]
            varying_dimensions = sum(
                len({round(float(vector[index]), 7) for vector in features}) > 1
                for index in range(len(features[0]))
            )
            candidate_variation[split].append(varying_dimensions)
            for evidence_id, utility in zip(
                row["evidence_ids"], row["oracle_utilities"], strict=True
            ):
                grouped[(split, edit_type, aoi, _candidate_year(evidence_id))].append(
                    float(utility)
                )

    rows = []
    for (split, edit_type, aoi, year), values in sorted(grouped.items()):
        rows.append(
            {
                "split": split,
                "edit_type": edit_type,
                "aoi_id": aoi,
                "year": year,
                "count": len(values),
                "mean_utility": mean(values),
                "positive_fraction": sum(value > 0 for value in values) / len(values),
            }
        )
    summary = {
        "candidate_feature_varying_dimensions": {
            split: {
                "minimum": min(values),
                "maximum": max(values),
                "mean": mean(values),
            }
            for split, values in sorted(candidate_variation.items())
        },
        "groups": rows,
        "test_assets_read": False,
    }
    text = json.dumps(summary, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
