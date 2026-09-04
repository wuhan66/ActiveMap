#!/usr/bin/env python3
"""Audit structured ActiveMap SFT and preference records before LLM training."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _action_key(payload: dict[str, object]) -> str:
    action = str(payload.get("action", "MISSING"))
    if action == "COMMIT":
        return f"COMMIT:{payload.get('edit', 'MISSING')}"
    return action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sft", type=Path)
    parser.add_argument("preferences", type=Path)
    parser.add_argument("--expected-split", choices=("train", "val"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    action_counts: Counter[str] = Counter()
    trajectory_ids: set[str] = set()
    invalid_sft = 0
    split_mismatches = 0
    with args.sft.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            trajectory_ids.add(str(row["trajectory_id"]))
            try:
                observation = json.loads(row["messages"][1]["content"])
                action = json.loads(row["messages"][2]["content"])
                action_counts[_action_key(action)] += 1
                split_mismatches += observation.get("split") != args.expected_split
            except (KeyError, TypeError, json.JSONDecodeError):
                invalid_sft += 1

    margins = []
    chosen_counts: Counter[str] = Counter()
    rejected_counts: Counter[str] = Counter()
    pair_counts: Counter[str] = Counter()
    pair_margins: defaultdict[str, list[float]] = defaultdict(list)
    invalid_preferences = 0
    with args.preferences.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            try:
                chosen_key = _action_key(json.loads(row["chosen"]))
                rejected_key = _action_key(json.loads(row["rejected"]))
                margin = float(row["chosen_utility"]) - float(row["rejected_utility"])
                margins.append(margin)
                chosen_counts[chosen_key] += 1
                rejected_counts[rejected_key] += 1
                pair = f"{chosen_key} -> {rejected_key}"
                pair_counts[pair] += 1
                pair_margins[pair].append(margin)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                invalid_preferences += 1

    total_actions = sum(action_counts.values())
    acquire_count = action_counts["ACQUIRE"]
    summary = {
        "expected_split": args.expected_split,
        "sft_count": total_actions + invalid_sft,
        "trajectory_count": len(trajectory_ids),
        "action_counts": dict(sorted(action_counts.items())),
        "acquire_fraction": acquire_count / max(total_actions, 1),
        "invalid_sft": invalid_sft,
        "split_mismatches": split_mismatches,
        "preference_count": len(margins) + invalid_preferences,
        "invalid_preferences": invalid_preferences,
        "preference_chosen_counts": dict(sorted(chosen_counts.items())),
        "preference_rejected_counts": dict(sorted(rejected_counts.items())),
        "preference_pair_counts": dict(sorted(pair_counts.items())),
        "preference_pair_margin_mean": {
            key: float(np.mean(values)) for key, values in sorted(pair_margins.items())
        },
        "safety_pair_count": sum(
            count
            for pair, count in pair_counts.items()
            if pair.startswith("REJECT -> COMMIT:")
        ),
        "update_pair_count": sum(
            count
            for pair, count in pair_counts.items()
            if pair.startswith("COMMIT:") and pair.endswith(" -> REJECT")
        ),
        "preference_margin": {
            "minimum": float(np.min(margins)) if margins else None,
            "mean": float(np.mean(margins)) if margins else None,
            "median": float(np.median(margins)) if margins else None,
            "nonpositive_count": sum(value <= 0 for value in margins),
        },
        "test_assets_read": False,
    }
    text = json.dumps(summary, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
