#!/usr/bin/env python3
"""Build safety-targeted Agent preferences from frozen counterfactual utilities."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from activemap.agent.trajectories import SYSTEM_PROMPT


def _action_key(action: dict[str, Any]) -> str:
    kind = str(action["action"])
    if kind == "COMMIT":
        return f"COMMIT:{action['edit']}"
    if kind == "ACQUIRE":
        return f"ACQUIRE:{action['evidence_id']}"
    return kind


def _action_from_key(key: str) -> dict[str, str]:
    if key.startswith("COMMIT:"):
        return {"action": "COMMIT", "edit": key.split(":", 1)[1]}
    if key.startswith("ACQUIRE:"):
        return {"action": "ACQUIRE", "evidence_id": key.split(":", 1)[1]}
    if key == "REJECT":
        return {"action": "REJECT"}
    raise ValueError(f"unsupported action key: {key}")


def _preference_family(chosen_key: str) -> str:
    if chosen_key == "REJECT":
        return "safety"
    if chosen_key.startswith("COMMIT:"):
        return "update"
    if chosen_key.startswith("ACQUIRE:"):
        return "acquire"
    raise ValueError(f"unsupported chosen action: {chosen_key}")


def _rejected_candidates(chosen_key: str, utilities: dict[str, float]) -> list[str]:
    family = _preference_family(chosen_key)
    if family == "safety":
        return [key for key in utilities if key.startswith("COMMIT:")]
    if family == "update":
        return ["REJECT"] if "REJECT" in utilities else []
    return [
        key
        for key in utilities
        if key == "REJECT" or key.startswith("COMMIT:")
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectories", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-split", choices=("train", "val"), required=True)
    args = parser.parse_args()

    pair_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    margins: list[float] = []
    skipped = 0
    split_mismatches = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (
        args.trajectories.open(encoding="utf-8") as source,
        args.output.open("w", encoding="utf-8") as destination,
    ):
        for line in source:
            trajectory = json.loads(line)
            split_mismatches += trajectory.get("split") != args.expected_split
            for transition in trajectory["transitions"]:
                chosen_action = transition["action"]
                chosen_key = _action_key(chosen_action)
                utilities = {
                    str(key): float(value)
                    for key, value in transition["oracle_utilities"].items()
                }
                candidates = _rejected_candidates(chosen_key, utilities)
                if not candidates or chosen_key not in utilities:
                    skipped += 1
                    continue
                rejected_key = max(candidates, key=utilities.__getitem__)
                margin = utilities[chosen_key] - utilities[rejected_key]
                if margin <= 0:
                    detail = f"{chosen_key} > {rejected_key}: {margin}"
                    raise ValueError(
                        f"non-positive preference margin for {detail}"
                    )
                observation = json.dumps(
                    transition["observation"], separators=(",", ":")
                )
                chosen = json.dumps(chosen_action, separators=(",", ":"))
                rejected = json.dumps(
                    _action_from_key(rejected_key), separators=(",", ":")
                )
                destination.write(
                    json.dumps(
                        {
                            "system": SYSTEM_PROMPT,
                            "prompt": f"{SYSTEM_PROMPT}\n{observation}\nAction:",
                            "chosen": chosen,
                            "rejected": rejected,
                            "chosen_utility": utilities[chosen_key],
                            "rejected_utility": utilities[rejected_key],
                            "preference_family": _preference_family(chosen_key),
                            "trajectory_id": trajectory["trajectory_id"],
                            "step": transition["observation"]["step"],
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                family = _preference_family(chosen_key)
                family_counts[family] += 1
                chosen_kind = chosen_key.split(":", 1)[0]
                rejected_kind = rejected_key.split(":", 1)[0]
                pair_counts[f"{chosen_kind} -> {rejected_kind}"] += 1
                margins.append(margin)

    summary = {
        "trajectories": str(args.trajectories.resolve()),
        "expected_split": args.expected_split,
        "preference_count": len(margins),
        "family_counts": dict(sorted(family_counts.items())),
        "pair_counts": dict(sorted(pair_counts.items())),
        "skipped": skipped,
        "split_mismatches": split_mismatches,
        "margin_min": min(margins) if margins else None,
        "margin_mean": sum(margins) / len(margins) if margins else None,
        "nonpositive_margin_count": sum(value <= 0 for value in margins),
        "test_assets_read": False,
    }
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
