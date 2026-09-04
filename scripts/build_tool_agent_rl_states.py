#!/usr/bin/env python3
"""Build grounded sparse-tool GRPO states from frozen SFT preferences."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from activemap.agent.records import AgentAction, AgentObservation
from activemap.agent.rl import reward_action_key, target_terminal_action


def _preference_observation(prompt: str) -> AgentObservation:
    marker = "\nAction:"
    if not prompt.endswith(marker):
        raise ValueError("tool preference prompt lacks terminal Action marker")
    prefix = prompt[: -len(marker)]
    _, observation_json = prefix.rsplit("\n", 1)
    return AgentObservation.model_validate_json(observation_json)


def _add_utility(
    utilities: dict[str, float], action_json: str, value: float, *, context: str
) -> None:
    action = AgentAction.model_validate_json(action_json)
    key = reward_action_key(action)
    previous = utilities.get(key)
    if previous is not None and abs(previous - value) > 1e-8:
        raise ValueError(f"inconsistent utility for {key} in {context}: {previous} vs {value}")
    utilities[key] = float(value)


def build_tool_rl_states(
    sft_path: Path,
    preferences_path: Path,
    output: Path,
    *,
    expected_split: str,
) -> dict[str, Any]:
    if expected_split == "test":
        raise ValueError("tool RL state construction must not read test")
    preference_utilities: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    preference_count = 0
    with preferences_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != expected_split:
                raise ValueError(f"preference split mismatch: {row.get('split')!r}")
            observation = _preference_observation(str(row["prompt"]))
            key = (str(row["trajectory_id"]), observation.step)
            utilities = preference_utilities[key]
            _add_utility(
                utilities,
                str(row["chosen"]),
                float(row["chosen_utility"]),
                context=str(key),
            )
            _add_utility(
                utilities,
                str(row["rejected"]),
                float(row["rejected_utility"]),
                context=str(key),
            )
            preference_count += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    targets: Counter[str] = Counter()
    oracle_best_actions: Counter[str] = Counter()
    state_count = 0
    with (
        sft_path.open(encoding="utf-8") as source,
        output.open("w", encoding="utf-8") as destination,
    ):
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) != 3:
                raise ValueError("tool SFT row must contain three messages")
            observation = AgentObservation.model_validate_json(messages[1]["content"])
            if observation.split != expected_split:
                raise ValueError(f"tool SFT observation split={observation.split!r}")
            key = (str(row["trajectory_id"]), int(row["step"]))
            utilities = preference_utilities.get(key)
            if not utilities:
                raise ValueError(f"tool SFT state has no frozen preferences: {key}")
            target_action = AgentAction.model_validate_json(messages[2]["content"])
            target_key = reward_action_key(target_action)
            if target_key not in utilities:
                raise ValueError(f"SFT target {target_key} lacks utility in {key}")
            oracle_best = max(utilities, key=utilities.__getitem__)
            if utilities[target_key] + 1e-8 < utilities[oracle_best]:
                raise ValueError(f"SFT target is not utility-optimal in {key}")
            terminal_target, stop_utility = target_terminal_action(utilities)
            targets[target_action.action.value] += 1
            oracle_best_actions[oracle_best.split(":", 1)[0]] += 1
            observation_json = observation.model_dump_json(exclude_none=True)
            output_row = {
                "prompt": messages[:2],
                "observation_json": observation_json,
                "oracle_utilities": utilities,
                "target_action_key": terminal_target,
                "stop_utility": stop_utility,
                "oracle_best_action_key": oracle_best,
                "oracle_best_advantage": utilities[oracle_best] - stop_utility,
                "trajectory_id": key[0],
                "step": key[1],
                "split": expected_split,
                "rl_source": "grounded_sparse_tool_preference",
            }
            destination.write(json.dumps(output_row, separators=(",", ":")) + "\n")
            state_count += 1
    if state_count != len(preference_utilities):
        raise ValueError(
            f"SFT/preference state mismatch: {state_count} vs {len(preference_utilities)}"
        )
    summary = {
        "schema_version": "activemap-grounded-tool-grpo-v1",
        "sft": str(sft_path.resolve()),
        "preferences": str(preferences_path.resolve()),
        "expected_split": expected_split,
        "state_count": state_count,
        "preference_count": preference_count,
        "target_action_counts": dict(sorted(targets.items())),
        "oracle_best_action_counts": dict(sorted(oracle_best_actions.items())),
        "online_closed_loop": False,
        "test_assets_read": False,
    }
    output.with_suffix(output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sft", type=Path)
    parser.add_argument("preferences", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-split", choices=("train", "val"), required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build_tool_rl_states(
                args.sft,
                args.preferences,
                args.output,
                expected_split=args.expected_split,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
