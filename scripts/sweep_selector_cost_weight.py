#!/usr/bin/env python3
"""Sweep evidence-cost weights using train states only."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from activemap.training.data import load_selector_samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("states", type=Path)
    parser.add_argument("--weights", default="0,0.02,0.05,0.08,0.10,0.14,0.18")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    weights = [float(value) for value in args.weights.split(",")]

    samples = [
        sample
        for sample in load_selector_samples(args.states, split="train")
        if int(sample.metadata.get("oracle_step", -1)) == 0
    ]
    if not samples:
        raise ValueError("no train initial states found")
    rows = []
    for weight in weights:
        acquire = []
        best_utilities = []
        by_edit: dict[str, list[bool]] = defaultdict(list)
        for sample in samples:
            original_weight = float(sample.metadata["cost_weight"])
            adjusted = np.asarray(sample.oracle_utilities) + (
                original_weight - weight
            ) * np.asarray(sample.evidence_costs)
            should_acquire = float(np.max(adjusted)) > sample.stop_utility
            acquire.append(should_acquire)
            best_utilities.append(float(np.max(adjusted)))
            by_edit[sample.edit_type.value].append(should_acquire)
        rows.append(
            {
                "cost_weight": weight,
                "state_count": len(samples),
                "acquire_fraction": float(np.mean(acquire)),
                "mean_best_utility": float(np.mean(best_utilities)),
                "by_initial_edit_acquire_fraction": {
                    edit: float(np.mean(values)) for edit, values in sorted(by_edit.items())
                },
            }
        )
    result = {
        "selection_split": "train",
        "test_assets_read": False,
        "rows": rows,
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
