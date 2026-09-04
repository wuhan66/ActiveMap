#!/usr/bin/env python3
"""Build test-free contextual GRPO states from frozen Agent trajectories."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from activemap.agent.records import AgentObservation
from activemap.agent.rl import target_terminal_action
from activemap.agent.trajectories import SYSTEM_PROMPT


def build_rl_states(trajectories: Path, output: Path, *, expected_split: str) -> dict[str, Any]:
    if expected_split == "test":
        raise ValueError("RL state construction must not read the test split")
    output.parent.mkdir(parents=True, exist_ok=True)
    action_targets: Counter[str] = Counter()
    oracle_best_actions: Counter[str] = Counter()
    trajectory_count = 0
    state_count = 0
    split_mismatches = 0
    with (
        trajectories.open(encoding="utf-8") as source,
        output.open("w", encoding="utf-8") as destination,
    ):
        for line in source:
            if not line.strip():
                continue
            trajectory = json.loads(line)
            trajectory_count += 1
            split = str(trajectory.get("split", ""))
            split_mismatches += split != expected_split
            for transition in trajectory.get("transitions", []):
                observation = AgentObservation.model_validate(transition["observation"])
                if observation.split != expected_split:
                    split_mismatches += 1
                utilities = {
                    str(key): float(value) for key, value in transition["oracle_utilities"].items()
                }
                target, stop_utility = target_terminal_action(utilities)
                oracle_best = max(utilities, key=utilities.__getitem__)
                action_targets[target.split(":", 1)[0]] += 1
                oracle_best_actions[oracle_best.split(":", 1)[0]] += 1
                observation_json = observation.model_dump_json(exclude_none=True)
                row = {
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": observation_json},
                    ],
                    "observation_json": observation_json,
                    "oracle_utilities": utilities,
                    "target_action_key": target,
                    "stop_utility": stop_utility,
                    "oracle_best_action_key": oracle_best,
                    "oracle_best_advantage": utilities[oracle_best] - stop_utility,
                    "trajectory_id": str(trajectory["trajectory_id"]),
                    "step": observation.step,
                    "split": expected_split,
                }
                destination.write(json.dumps(row, separators=(",", ":")) + "\n")
                state_count += 1
    if split_mismatches:
        raise ValueError(f"found {split_mismatches} trajectory/observation split mismatches")
    if not state_count:
        raise ValueError(f"no RL states found in {trajectories}")
    summary = {
        "schema_version": "activemap-contextual-grpo-v1",
        "source_trajectories": str(trajectories.resolve()),
        "expected_split": expected_split,
        "trajectory_count": trajectory_count,
        "state_count": state_count,
        "target_action_counts": dict(sorted(action_targets.items())),
        "oracle_best_action_counts": dict(sorted(oracle_best_actions.items())),
        "split_mismatches": split_mismatches,
        "online_closed_loop": False,
        "test_assets_read": False,
    }
    output.with_suffix(output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectories", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-split", choices=("train", "val"), required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build_rl_states(args.trajectories, args.output, expected_split=args.expected_split),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
