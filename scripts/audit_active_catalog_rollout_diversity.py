#!/usr/bin/env python3
"""Audit whether stochastic recurrent rollouts contain matched-state action diversity."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def _action(event: dict[str, Any]) -> str | None:
    payload = event.get("selector_action")
    if not isinstance(payload, dict) or payload.get("stage") != "SELECT":
        return None
    if payload.get("selection") == "STOP":
        return "STOP"
    if payload.get("selection") == "ACQUIRE" and payload.get("evidence_id"):
        return f"ACQUIRE:{payload['evidence_id']}"
    return "INVALID_SELECTOR"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", action="append", type=Path, required=True)
    args = parser.parse_args()

    action_counts: Counter[str] = Counter()
    rollout_counts: dict[str, Counter[str]] = {}
    exact: dict[tuple[str, float, str], set[str]] = defaultdict(set)
    by_position: dict[tuple[str, float, int], set[tuple[str, str]]] = defaultdict(set)
    unsupported = invalid = 0

    for trace in args.traces:
        current: Counter[str] = Counter()
        for line in trace.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            selector_index = 0
            for event in row.get("events", []):
                if not event.get("valid_action"):
                    invalid += 1
                    continue
                action = _action(event)
                state = event.get("observable_state")
                if action is None or not isinstance(state, dict):
                    unsupported += 1
                    continue
                episode = str(row["source_episode"])
                budget = float(row["budget"])
                state_hash = _digest(state)
                exact[(episode, budget, state_hash)].add(action)
                by_position[(episode, budget, selector_index)].add((state_hash, action))
                action_counts[action] += 1
                current[action] += 1
                selector_index += 1
        rollout_counts[str(trace)] = current

    exact_multi = sum(len(actions) >= 2 for actions in exact.values())
    positional_action_multi = sum(
        len({action for _, action in values}) >= 2 for values in by_position.values()
    )
    positional_state_drift = sum(
        len({state for state, _ in values}) >= 2 for values in by_position.values()
    )
    payload = {
        "rollout_files": len(args.traces),
        "selector_actions": sum(action_counts.values()),
        "action_counts": dict(action_counts.most_common()),
        "exact_states": len(exact),
        "exact_states_with_multiple_actions": exact_multi,
        "positional_states": len(by_position),
        "positional_states_with_multiple_actions": positional_action_multi,
        "positional_states_with_hash_drift": positional_state_drift,
        "unsupported_nonselector_events": unsupported,
        "invalid_events": invalid,
        "per_rollout_action_counts": {
            path: dict(counts.most_common()) for path, counts in rollout_counts.items()
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
