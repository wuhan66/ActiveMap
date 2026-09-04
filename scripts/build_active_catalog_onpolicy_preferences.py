#!/usr/bin/env python3
"""Build matched-state preferences from executed stochastic recurrent rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from activemap.agent.active_catalog import ACTIVE_CATALOG_SYSTEM_PROMPT
from scripts.build_active_catalog_vlm_preferences import action_message
from scripts.evaluate_active_catalog_closed_loop import image_index


def _state_id(state: dict[str, Any]) -> str:
    payload = json.dumps(state, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def _action_key(payload: dict[str, Any]) -> str:
    if payload.get("stage") != "SELECT":
        raise ValueError("on-policy action is not SELECT")
    if payload.get("selection") == "STOP":
        return "STOP"
    if payload.get("selection") == "ACQUIRE" and payload.get("evidence_id"):
        return f"ACQUIRE:{payload['evidence_id']}"
    raise ValueError("invalid on-policy selector action")


def _cost(state: dict[str, Any], key: str) -> float:
    if key == "STOP":
        return 0.0
    evidence_id = key.split(":", 1)[1]
    for candidate in state.get("candidate_evidence", []):
        if candidate.get("evidence_id") == evidence_id:
            return float(candidate["cost"])
    raise ValueError(f"executed action is absent from observable state: {key}")


def _trajectory_return(row: dict[str, Any]) -> float:
    return (
        float(row["quality_cost_utility"])
        + 0.5 * float(bool(row["terminal_correct"]))
        - 1.0 * float(bool(row["false_edit"]))
        - 0.75 * float(bool(row["missed_edit"]))
    )


def build_preferences(
    traces: list[Path],
    sft: Path,
    evaluation_index: Path,
    output: Path,
    *,
    expected_split: str,
    minimum_margin: float = 1e-4,
    preference_mode: str = "weighted",
    false_edit_tolerance: float = 0.02,
) -> dict[str, Any]:
    if expected_split not in {"train", "val"} or len(traces) < 2:
        raise ValueError("on-policy preferences require train/val and at least two rollouts")
    if preference_mode not in {"weighted", "safety_first"}:
        raise ValueError("unknown preference mode")
    if not 0.0 <= false_edit_tolerance <= 1.0:
        raise ValueError("false-edit tolerance must be in [0, 1]")
    images = image_index(sft, evaluation_index, split=expected_split)
    adapters = set()
    observations: dict[tuple[str, float, str], dict[str, Any]] = {}
    returns: dict[tuple[str, float, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    false_edits: dict[tuple[str, float, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    missed_edits: dict[tuple[str, float, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    invalid_events = 0
    unsupported_events = 0
    trajectory_count = 0
    for trace in traces:
        summary_path = trace.parent / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("split") != expected_split or summary.get("test_assets_read") is not False:
            raise ValueError(f"invalid on-policy summary: {summary_path}")
        if summary.get("protocol", {}).get("stochastic_policy_sampling") is not True:
            raise ValueError("on-policy preference source must use stochastic sampling")
        adapters.add(str(summary["sources"]["adapter"]))
        for line in trace.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            if row.get("split") != expected_split or row.get("test_assets_read") is not False:
                raise ValueError("trace violates split/test isolation")
            episode = str(row["source_episode"])
            if episode not in images:
                raise ValueError(f"missing on-policy image for {episode}")
            trajectory_count += 1
            value = _trajectory_return(row)
            for event in row.get("events", []):
                if not event.get("valid_action"):
                    invalid_events += 1
                    continue
                state = event.get("observable_state")
                selector_action = event.get("selector_action")
                if not isinstance(state, dict) or not isinstance(
                    selector_action, dict
                ):
                    unsupported_events += 1
                    continue
                identity = (episode, float(row["budget"]), _state_id(state))
                action = _action_key(selector_action)
                _cost(state, action)
                observations[identity] = state
                returns[identity][action].append(value)
                false_edits[identity][action].append(float(bool(row["false_edit"])))
                missed_edits[identity][action].append(float(bool(row["missed_edit"])))
    if len(adapters) != 1:
        raise ValueError("on-policy rollout sources must share one policy snapshot")

    output.parent.mkdir(parents=True, exist_ok=True)
    skipped_single_action = skipped_ties = 0
    families: Counter[str] = Counter()
    margins = []
    written = 0
    with output.open("x", encoding="utf-8") as destination:
        for identity in sorted(returns):
            by_action = returns[identity]
            if len(by_action) < 2:
                skipped_single_action += 1
                continue
            means = {key: statistics.fmean(values) for key, values in by_action.items()}
            false_rates = {
                key: statistics.fmean(false_edits[identity][key]) for key in by_action
            }
            missed_rates = {
                key: statistics.fmean(missed_edits[identity][key]) for key in by_action
            }
            if preference_mode == "safety_first":
                safest = min(false_rates.values())
                feasible = [
                    key
                    for key in means
                    if false_rates[key] <= safest + false_edit_tolerance
                ]
                chosen = sorted(feasible, key=lambda key: (-means[key], key))[0]
                rejected = sorted(
                    (key for key in means if key != chosen),
                    key=lambda key: (-false_rates[key], means[key], key),
                )[0]
            else:
                ranked = sorted(means, key=lambda key: (-means[key], key))
                chosen, rejected = ranked[0], ranked[-1]
            margin = means[chosen] - means[rejected]
            false_edit_gap = false_rates[rejected] - false_rates[chosen]
            if margin < minimum_margin and false_edit_gap <= false_edit_tolerance:
                skipped_ties += 1
                continue
            episode, budget, state_id = identity
            state = observations[identity]
            family = (
                "stop_over_acquire" if chosen == "STOP"
                else "acquire_over_stop" if rejected == "STOP"
                else "evidence_ranking"
            )
            row = {
                "schema_version": "active-catalog-vlm-preference-v1",
                "example_id": f"online-{state_id}",
                "task_id": episode,
                "split": expected_split,
                "prompt": [
                    {"role": "system", "content": [{"type": "text", "text": ACTIVE_CATALOG_SYSTEM_PROMPT}]},
                    {"role": "user", "content": [
                        {"type": "image", "image": str(images[episode].resolve())},
                        {"type": "text", "text": json.dumps(state, separators=(",", ":"))},
                    ]},
                ],
                "chosen": action_message(chosen),
                "rejected": action_message(rejected),
                "chosen_action_key": chosen,
                "rejected_action_key": rejected,
                "chosen_utility": means[chosen],
                "rejected_utility": means[rejected],
                "utility_margin": margin,
                "chosen_false_edit_rate": false_rates[chosen],
                "rejected_false_edit_rate": false_rates[rejected],
                "false_edit_rate_gap": false_edit_gap,
                "chosen_missed_edit_rate": missed_rates[chosen],
                "rejected_missed_edit_rate": missed_rates[rejected],
                "chosen_cost": _cost(state, chosen),
                "rejected_cost": _cost(state, rejected),
                "preference_family": family,
                "negative_rank": 1,
                "on_policy_executed_recurrent": True,
                "rollout_policy_snapshot": next(iter(adapters)),
                "budget": budget,
                "model_visible_utility": False,
                "test_assets_read": False,
            }
            destination.write(json.dumps(row, separators=(",", ":")) + "\n")
            written += 1
            margins.append(margin)
            families[family] += 1
    if not written:
        raise ValueError("stochastic rollouts produced no matched-state preferences")
    summary = {
        "schema_version": "active-catalog-onpolicy-preference-summary-v1",
        "split": expected_split,
        "rollout_files": len(traces),
        "trajectories": trajectory_count,
        "matched_states": len(returns),
        "preferences": written,
        "skipped_single_action": skipped_single_action,
        "skipped_ties": skipped_ties,
        "invalid_events_excluded": invalid_events,
        "unsupported_nonselector_events_excluded": unsupported_events,
        "family_counts": dict(sorted(families.items())),
        "margin_mean": statistics.fmean(margins),
        "preference_mode": preference_mode,
        "false_edit_tolerance": false_edit_tolerance,
        "policy_snapshot": next(iter(adapters)),
        "actions_executed_in_recurrent_environment": True,
        "optimization_type": "iterative_on_policy_preference_optimization",
        "test_assets_read": False,
    }
    output.with_suffix(output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sft", type=Path)
    parser.add_argument("evaluation_index", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--traces", action="append", type=Path, required=True)
    parser.add_argument("--expected-split", choices=("train", "val"), required=True)
    parser.add_argument("--minimum-margin", type=float, default=1e-4)
    parser.add_argument(
        "--preference-mode",
        choices=("weighted", "safety_first"),
        default="weighted",
    )
    parser.add_argument("--false-edit-tolerance", type=float, default=0.02)
    args = parser.parse_args()
    print(json.dumps(build_preferences(
        args.traces, args.sft, args.evaluation_index, args.output,
        expected_split=args.expected_split, minimum_margin=args.minimum_margin,
        preference_mode=args.preference_mode,
        false_edit_tolerance=args.false_edit_tolerance,
    ), indent=2))


if __name__ == "__main__":
    main()
