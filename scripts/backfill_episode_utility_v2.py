#!/usr/bin/env python3
"""Backfill proxy episode-utility-v2 fields without changing source JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from activemap.evaluation.episode_utility import score_episode_profiles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    if args.input.resolve() == args.output.resolve():
        raise ValueError("output must differ from input")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.input.open("r", encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8"
    ) as destination:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if "episode_utility_v2_proxy_balanced" not in row:
                false_edit = bool(row["false_edit"])
                missed_edit = bool(row["missed_edit"])
                wrong_edit = bool(
                    not row["terminal_correct"] and not false_edit and not missed_edit
                )
                profiles = score_episode_profiles(
                    final_map_quality=float(row["terminal_correct"]),
                    prior_map_quality=float(
                        str(row["target"]) == "REJECT"
                        or str(row["target"]).endswith(":KEEP")
                    ),
                    spent_cost=float(row["spent_cost"]),
                    budget=float(row["budget"]),
                    false_edit=false_edit,
                    missed_edit=missed_edit,
                    wrong_edit=wrong_edit,
                )
                row["wrong_edit"] = wrong_edit
                row["episode_utility_v2_proxy"] = profiles
                for name, result in profiles.items():
                    row[f"episode_utility_v2_proxy_{name}"] = result["value"]
            destination.write(json.dumps(row, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
