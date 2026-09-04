#!/usr/bin/env python3
"""Audit expanded selector states before training."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from activemap.training.data import load_selector_samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("states", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    samples = load_selector_samples(args.states)
    split_counts: Counter[str] = Counter()
    budget_counts: Counter[str] = Counter()
    step_counts: Counter[int] = Counter()
    stop_counts: Counter[str] = Counter()
    candidate_counts: Counter[int] = Counter()
    utility_values: dict[str, list[float]] = defaultdict(list)
    target_by_gt: Counter[str] = Counter()
    draft_gt_cross: Counter[str] = Counter()
    best_gain_by_gt: dict[str, list[float]] = defaultdict(list)
    episodes_by_split: dict[str, set[str]] = defaultdict(set)
    aois_by_split: dict[str, set[str]] = defaultdict(set)
    source_budget_steps: Counter[tuple[str, float, int]] = Counter()
    for sample in samples:
        split_counts[sample.split] += 1
        budget = float(sample.metadata["budget"])
        step = int(sample.metadata["oracle_step"])
        budget_counts[f"{budget:g}"] += 1
        step_counts[step] += 1
        candidate_counts[len(sample.evidence_ids)] += 1
        source = str(sample.metadata["source_episode"])
        gt_edit = str(sample.metadata.get("gt_edit", "UNKNOWN"))
        aoi_id = str(sample.metadata.get("aoi_id", "UNKNOWN"))
        source_budget_steps[(source, budget, step)] += 1
        target = sample.target_index(allow_stop=True)
        action = "STOP" if target == len(sample.evidence_ids) else "ACQUIRE"
        stop_counts[action] += 1
        target_by_gt[f"{sample.split}:{gt_edit}:{action}"] += 1
        draft_gt_cross[f"{sample.split}:{sample.edit_type.value}:{gt_edit}"] += 1
        best_gain_by_gt[f"{sample.split}:{gt_edit}"].append(
            max(sample.oracle_utilities) - sample.stop_utility
        )
        episodes_by_split[sample.split].add(source)
        aois_by_split[sample.split].add(aoi_id)
        utility_values[sample.split].extend(sample.oracle_utilities)

    test_assets_read = "test" in split_counts
    if test_assets_read:
        from activemap.frozen_test import assert_frozen_test_access

        assert_frozen_test_access()

    duplicates = sum(count - 1 for count in source_budget_steps.values() if count > 1)
    summary = {
        "states": len(samples),
        "splits": dict(sorted(split_counts.items())),
        "budgets": dict(sorted(budget_counts.items(), key=lambda item: float(item[0]))),
        "steps": dict(sorted(step_counts.items())),
        "targets": dict(sorted(stop_counts.items())),
        "target_support_by_ground_truth": dict(sorted(target_by_gt.items())),
        "draft_ground_truth_cross": dict(sorted(draft_gt_cross.items())),
        "best_gain_by_ground_truth": {
            key: {
                "state_count": len(values),
                "mean": float(np.mean(values)),
                "maximum": float(np.max(values)),
                "positive_fraction": float(np.mean(np.asarray(values) > 0.0)),
            }
            for key, values in sorted(best_gain_by_gt.items())
        },
        "unique_episodes": {
            split: len(values) for split, values in sorted(episodes_by_split.items())
        },
        "unique_aois": {
            split: len(values) for split, values in sorted(aois_by_split.items())
        },
        "candidate_counts": dict(sorted(candidate_counts.items())),
        "duplicate_source_budget_steps": duplicates,
        "test_assets_read": test_assets_read,
        "utility": {
            split: {
                "count": len(values),
                "mean": float(np.mean(values)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "positive_fraction": float(np.mean(np.asarray(values) > 0.0)),
            }
            for split, values in sorted(utility_values.items())
        },
    }
    text = json.dumps(summary, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
