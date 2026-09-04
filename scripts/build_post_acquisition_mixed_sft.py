#!/usr/bin/env python3
"""Build a balanced, train-only cold start for post-acquisition tool control.

Historical reachable-tool supervision contains useful tool-positive paths but
too few terminal no-tool states.  This builder adds REJECT/COMMIT labels from
the same public post-acquisition state and labels the result as behaviour data,
never as on-policy GRPO data.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from activemap.agent.identifiers import public_evidence_id, public_task_id
from activemap.agent.records import AgentObservation
from activemap.agent.tool_sft import TOOL_SYSTEM_PROMPT, policy_action_json, terminal_action
from activemap.agent.tools import CounterfactualBeliefUpdater, belief_from_features
from activemap.geo_tools.records import GeoToolName
from activemap.models import EditOperation


PROTOCOL = "post-acquisition-mixed-behavior-cold-start-v1"
FORBIDDEN_PROMPT_KEYS = {
    "gt_edit",
    "oracle_utilities",
    "target_operation",
    "policy_relative_advantage",
}
ACTION_NAMES = {"REJECT", "COMMIT", "USE_TOOL"}


def _nested_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        result = set(value)
        for item in value.values():
            result.update(_nested_keys(item))
        return result
    if isinstance(value, list):
        result: set[str] = set()
        for item in value:
            result.update(_nested_keys(item))
        return result
    return set()


def _public_handle(value: str) -> str:
    return value if value.startswith("evidence-") else public_evidence_id(value)


def _action_name(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError("SFT row must contain exactly three messages")
    payload = json.loads(str(messages[2]["content"]))
    action = str(payload.get("action", ""))
    if action not in ACTION_NAMES:
        raise ValueError(f"post-acquisition data forbids action={action!r}")
    return action


def _row_split(row: dict[str, Any]) -> str:
    messages = row["messages"]
    observation = json.loads(str(messages[1]["content"]))
    split = str(observation.get("split", ""))
    if split not in {"train", "val"}:
        raise ValueError(f"post-acquisition data forbids split={split!r}")
    leaked = FORBIDDEN_PROMPT_KEYS & _nested_keys(observation)
    if leaked:
        raise ValueError("prompt leaks supervision fields: " + ", ".join(sorted(leaked)))
    return split


def _terminal_observation(sample: Any) -> AgentObservation:
    metadata = sample.metadata
    selected_raw = [str(value) for value in metadata.get("selected_evidence_ids", [])]
    if not selected_raw:
        raise ValueError(f"post-acquisition state has no selected evidence: {sample.sample_id}")
    budget = float(metadata.get("budget", 1.5))
    if budget <= 0:
        raise ValueError(f"state has non-positive budget: {sample.sample_id}")
    remaining = float(np.clip(float(sample.state_features[0]), 0.0, 1.0) * budget)
    predictions = metadata.get("evidence_predictions")
    belief = (
        CounterfactualBeliefUpdater(sample).fuse(selected_raw)
        if isinstance(predictions, dict)
        else belief_from_features(sample)
    )
    task_source = str(metadata.get("source_episode", sample.sample_id))
    return AgentObservation(
        task_id=public_task_id(task_source),
        split=sample.split,
        step=int(metadata.get("oracle_step", 0)),
        initial_budget=budget,
        remaining_budget=remaining,
        spent_cost=max(budget - remaining, 0.0),
        selected_evidence_ids=[_public_handle(value) for value in selected_raw],
        belief=belief,
        candidates=[],
        terminal_score=float(sample.stop_utility),
        available_tools=(
            [GeoToolName.IMAGE_QUALITY, GeoToolName.TEMPORAL_CHANGE]
            if remaining > 0.0
            else []
        ),
        tool_history=[],
    )


def terminal_row(sample: Any) -> dict[str, Any]:
    target = sample.metadata.get("gt_edit", sample.edit_type.value)
    action = terminal_action(EditOperation(str(target)))
    observation = _terminal_observation(sample)
    return {
        "messages": [
            {"role": "system", "content": TOOL_SYSTEM_PROMPT},
            {"role": "user", "content": observation.model_dump_json(exclude_none=True)},
            {"role": "assistant", "content": policy_action_json(action, observation)},
        ],
        "trajectory_id": f"terminal-state-{sample.sample_id}",
        "step": observation.step,
        "protocol": PROTOCOL,
        "composition_source": "counterfactual_terminal_state",
        "behavior_policy": "counterfactual_terminal_oracle",
        "action_encoding": "selected_evidence_index_v1",
    }


def counterfactual_keep_rollout_sample(sample: Any) -> Any:
    """Materialize a train/validation-only post-acquisition KEEP state.

    A natural no-change task stops before the post-acquisition controller is
    reached.  This fixture follows the same affordable public acquisition used
    by :func:`synthetic_keep_row`, but returns a ``SelectorSample`` that the
    executable environment can reset from.  It is behavior-policy support for
    exploration diagnostics, never a paper test example or an on-policy GRPO
    trajectory by itself.
    """

    metadata = sample.metadata
    target = EditOperation(str(metadata.get("gt_edit", sample.edit_type.value)))
    if target != EditOperation.KEEP:
        raise ValueError("counterfactual re-entry requires a no-change source state")
    if int(metadata.get("oracle_step", -1)) != 0:
        raise ValueError("counterfactual re-entry requires an oracle step-zero state")
    predictions = metadata.get("evidence_predictions")
    if not isinstance(predictions, dict):
        raise ValueError("counterfactual re-entry requires evidence predictions")
    selected_raw = [str(value) for value in metadata.get("selected_evidence_ids", [])]
    budget = float(metadata.get("budget", 1.5))
    if budget <= 0.0:
        raise ValueError("counterfactual re-entry requires a positive budget")
    remaining_before = float(np.clip(float(sample.state_features[0]), 0.0, 1.0) * budget)
    options = [
        index
        for index, evidence_id in enumerate(sample.evidence_ids)
        if evidence_id not in selected_raw
        and evidence_id in predictions
        and float(sample.evidence_costs[index]) <= remaining_before
    ]
    if not options:
        raise ValueError(f"no affordable evidence for counterfactual KEEP: {sample.sample_id}")
    chosen_index = max(
        options,
        key=lambda index: (
            float(sample.oracle_utilities[index]),
            -float(sample.evidence_costs[index]),
        ),
    )
    chosen_id = str(sample.evidence_ids[chosen_index])
    selected_after = [*selected_raw, chosen_id]
    belief = CounterfactualBeliefUpdater(sample).fuse(selected_after)
    chosen_cost = float(sample.evidence_costs[chosen_index])
    remaining_after = remaining_before - chosen_cost
    if remaining_after < -1e-8:
        raise AssertionError("counterfactual acquisition exceeded the available budget")
    remaining_after = max(remaining_after, 0.0)

    penalty = (
        chosen_cost * float(metadata.get("cost_weight", 0.0))
        + float(sample.false_edit_risks[chosen_index]) * sample.false_edit_penalty_weight
    )
    prior_gain = float(sample.state_features[7])
    if str(metadata.get("utility_mode", "proxy")) == "executable":
        from activemap.evaluation.episode_utility import UTILITY_PROFILES

        outcomes = metadata.get("executable_outcomes")
        if not isinstance(outcomes, dict) or chosen_id not in outcomes:
            raise ValueError("executable counterfactual state lacks the chosen outcome")
        profile = UTILITY_PROFILES[str(metadata.get("utility_profile", "balanced"))]
        post_gain = max(
            prior_gain, float(outcomes[chosen_id]["terminal_score_before_cost"])
        )
        penalty = chosen_cost * profile.cost / budget
    else:
        post_gain = max(
            prior_gain,
            prior_gain + max(float(sample.oracle_utilities[chosen_index]) + penalty, 0.0),
        )

    state = list(sample.state_features)
    state[0] = remaining_after / budget
    state[1] = 1.0 - state[0]
    state[2:6] = belief.edit_probabilities
    total_evidence = len(set(sample.evidence_ids) | set(selected_after) | set(predictions))
    state[6] = len(selected_after) / max(total_evidence, 1)
    state[7] = post_gain
    hypothesis = list(sample.hypothesis_features)
    hypothesis[:4] = belief.edit_probabilities
    hypothesis[12] = belief.uncertainty
    hypothesis[13] = belief.confidence
    updated_metadata = {
        **metadata,
        "selected_evidence_ids": selected_after,
        "oracle_step": 1,
        "counterfactual_reentry": True,
        "counterfactual_source_sample_id": sample.sample_id,
        "counterfactual_acquired_evidence_id": chosen_id,
        "preacquired_spent_cost": budget - remaining_after,
        "preacquired_evidence_penalty": penalty,
    }
    return sample.model_copy(
        update={
            "sample_id": f"{sample.sample_id}__counterfactual_keep_reentry",
            "hypothesis_features": hypothesis,
            "state_features": state,
            "metadata": updated_metadata,
        }
    )


def synthetic_keep_row(sample: Any) -> dict[str, Any]:
    """Create a legal post-acquisition KEEP state from a no-change state.

    Natural oracle step-one states contain only tasks whose oracle chose to
    acquire.  A no-change task normally stops at step zero, so it cannot teach
    a post-acquisition controller to stop safely.  This behaviour-policy row
    takes one already public, affordable candidate through the normal
    counterfactual belief transition and then labels the executable REJECT
    action.  It is deliberately excluded from on-policy GRPO collection.
    """

    metadata = sample.metadata
    target = EditOperation(str(metadata.get("gt_edit", sample.edit_type.value)))
    if target != EditOperation.KEEP:
        raise ValueError("synthetic KEEP support requires a no-change source state")
    predictions = metadata.get("evidence_predictions")
    if not isinstance(predictions, dict):
        raise ValueError("synthetic KEEP support requires evidence predictions")
    selected_raw = [str(value) for value in metadata.get("selected_evidence_ids", [])]
    budget = float(metadata.get("budget", 1.5))
    remaining = float(np.clip(float(sample.state_features[0]), 0.0, 1.0) * budget)
    options = [
        index
        for index, evidence_id in enumerate(sample.evidence_ids)
        if evidence_id not in selected_raw
        and evidence_id in predictions
        and float(sample.evidence_costs[index]) <= remaining
    ]
    if not options:
        raise ValueError(f"no affordable evidence for synthetic KEEP: {sample.sample_id}")
    chosen_index = max(
        options,
        key=lambda index: (float(sample.oracle_utilities[index]), -float(sample.evidence_costs[index])),
    )
    chosen = str(sample.evidence_ids[chosen_index])
    selected_after = [*selected_raw, chosen]
    if any(value not in predictions for value in selected_after):
        raise ValueError(f"synthetic KEEP state lacks prediction support: {sample.sample_id}")
    remaining_after = max(remaining - float(sample.evidence_costs[chosen_index]), 0.0)
    belief = CounterfactualBeliefUpdater(sample).fuse(selected_after)
    task_source = str(metadata.get("source_episode", sample.sample_id))
    observation = AgentObservation(
        task_id=public_task_id(task_source),
        split=sample.split,
        step=int(metadata.get("oracle_step", 0)) + 1,
        initial_budget=budget,
        remaining_budget=remaining_after,
        spent_cost=max(budget - remaining_after, 0.0),
        selected_evidence_ids=[_public_handle(value) for value in selected_after],
        belief=belief,
        candidates=[],
        terminal_score=float(sample.stop_utility),
        available_tools=(
            [GeoToolName.IMAGE_QUALITY, GeoToolName.TEMPORAL_CHANGE]
            if remaining_after > 0.0
            else []
        ),
        tool_history=[],
    )
    action = terminal_action(EditOperation.KEEP)
    return {
        "messages": [
            {"role": "system", "content": TOOL_SYSTEM_PROMPT},
            {"role": "user", "content": observation.model_dump_json(exclude_none=True)},
            {"role": "assistant", "content": policy_action_json(action, observation)},
        ],
        "trajectory_id": f"synthetic-keep-{sample.sample_id}",
        "step": observation.step,
        "protocol": PROTOCOL,
        "composition_source": "synthetic_public_keep_transition",
        "behavior_policy": "counterfactual_keep_exploration",
        "action_encoding": "selected_evidence_index_v1",
    }


def _read_tool_rows(path: Path, split: str) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if _row_split(row) != split:
                continue
            if _action_name(row) == "USE_TOOL":
                payload = json.loads(row["messages"][2]["content"])
                call = payload.get("tool_call")
                inputs = call.get("inputs") if isinstance(call, dict) else None
                if not isinstance(inputs, dict) or "evidence_index" not in inputs:
                    raise ValueError(
                        f"{path}:{line_number} USE_TOOL must use an evidence_index pointer"
                    )
            rows.append(
                {
                    **row,
                    "protocol": PROTOCOL,
                    "composition_source": "verified_reachable_tool",
                    "behavior_policy": "verified_tool_trajectory",
                    "action_encoding": "selected_evidence_index_v1",
                }
            )
    if not rows:
        raise ValueError(f"no {split} tool rows in {path}")
    return rows


def _load_terminal_rows(
    states: Path,
    split: str,
    oracle_step: int,
    synthetic_keep_source_step: int | None,
) -> list[dict[str, Any]]:
    from activemap.training.data import load_selector_samples

    rows = []
    for sample in load_selector_samples(states, split=split):
        if int(sample.metadata.get("oracle_step", -1)) == oracle_step:
            rows.append(terminal_row(sample))
        if (
            synthetic_keep_source_step is not None
            and int(sample.metadata.get("oracle_step", -1)) == synthetic_keep_source_step
            and EditOperation(str(sample.metadata.get("gt_edit", sample.edit_type.value)))
            == EditOperation.KEEP
        ):
            try:
                rows.append(synthetic_keep_row(sample))
            except ValueError as exc:
                if "no affordable evidence" not in str(exc):
                    raise
    if not rows:
        raise ValueError(f"no {split} selector states at oracle_step={oracle_step}")
    return rows


def parse_action_weights(value: str | None) -> dict[str, float] | None:
    if value is None:
        return None
    weights: dict[str, float] = {}
    for part in value.split(","):
        name, separator, raw = part.strip().partition("=")
        if separator != "=" or name not in ACTION_NAMES:
            raise ValueError("weights must use REJECT=...,COMMIT=...,USE_TOOL=...")
        weight = float(raw)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("action weights must be finite and positive")
        weights[name] = weight
    if set(weights) != ACTION_NAMES:
        raise ValueError("weights must specify REJECT, COMMIT, and USE_TOOL")
    total = sum(weights.values())
    return {name: weight / total for name, weight in weights.items()}


def rebalance_train_rows(
    rows: list[dict[str, Any]], *, action_weights: dict[str, float] | None, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Oversample only, retaining all original behavior-policy support."""

    original_counts = Counter(_action_name(row) for row in rows)
    if action_weights is None:
        return rows, {
            "enabled": False,
            "input_action_counts": dict(sorted(original_counts.items())),
            "output_action_counts": dict(sorted(original_counts.items())),
            "sampling_repetitions": 0,
        }
    missing = sorted(ACTION_NAMES - set(original_counts))
    if missing:
        raise ValueError("cannot balance absent action classes: " + ", ".join(missing))
    target_total = max(
        math.ceil(original_counts[action] / action_weights[action])
        for action in ACTION_NAMES
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_action_name(row)].append(row)
    rng = random.Random(seed)
    balanced = list(rows)
    repetitions = 0
    for action in sorted(ACTION_NAMES):
        desired = math.ceil(target_total * action_weights[action])
        for repeat_index in range(max(0, desired - len(grouped[action]))):
            source = dict(rng.choice(grouped[action]))
            source["behavior_sampling_repeat"] = repeat_index + 1
            balanced.append(source)
            repetitions += 1
    rng.shuffle(balanced)
    final_counts = Counter(_action_name(row) for row in balanced)
    return balanced, {
        "enabled": True,
        "weights": action_weights,
        "input_action_counts": dict(sorted(original_counts.items())),
        "output_action_counts": dict(sorted(final_counts.items())),
        "sampling_repetitions": repetitions,
    }


def build_split(
    terminal_rows: list[dict[str, Any]], tool_rows: list[dict[str, Any]], *, split: str,
    action_weights: dict[str, float] | None, seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [*terminal_rows, *tool_rows]
    if any(_row_split(row) != split for row in rows):
        raise ValueError(f"row split mismatch while building {split}")
    if split == "train":
        rows, balance = rebalance_train_rows(rows, action_weights=action_weights, seed=seed)
    else:
        counts = Counter(_action_name(row) for row in rows)
        balance = {
            "enabled": False,
            "input_action_counts": dict(sorted(counts.items())),
            "output_action_counts": dict(sorted(counts.items())),
            "sampling_repetitions": 0,
        }
    return rows, {
        "terminal_state_rows": len(terminal_rows),
        "reachable_tool_rows": len(tool_rows),
        "output_rows": len(rows),
        "action_counts": dict(sorted(Counter(_action_name(row) for row in rows).items())),
        "balance": balance,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("selector_states", type=Path)
    parser.add_argument("tool_train", type=Path)
    parser.add_argument("tool_val", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--oracle-step", type=int, default=1)
    parser.add_argument("--synthetic-keep-source-step", type=int, default=0)
    parser.add_argument("--train-action-weights")
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()
    if args.oracle_step < 0 or args.synthetic_keep_source_step < 0:
        raise ValueError("oracle steps must be non-negative")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing existing output directory: {args.output_dir}")
    weights = parse_action_weights(args.train_action_weights)
    train_rows, train_summary = build_split(
        _load_terminal_rows(
            args.selector_states, "train", args.oracle_step, args.synthetic_keep_source_step
        ),
        _read_tool_rows(args.tool_train, "train"),
        split="train", action_weights=weights, seed=args.seed,
    )
    val_rows, val_summary = build_split(
        _load_terminal_rows(
            args.selector_states, "val", args.oracle_step, args.synthetic_keep_source_step
        ),
        _read_tool_rows(args.tool_val, "val"),
        split="val", action_weights=None, seed=args.seed,
    )
    train_tasks = {json.loads(row["messages"][1]["content"])["task_id"] for row in train_rows}
    val_tasks = {json.loads(row["messages"][1]["content"])["task_id"] for row in val_rows}
    if train_tasks & val_tasks:
        raise ValueError(f"train/validation task leakage: {len(train_tasks & val_tasks)} tasks")
    args.output_dir.mkdir(parents=True)
    _write_jsonl(args.output_dir / "train.jsonl", train_rows)
    _write_jsonl(args.output_dir / "val.jsonl", val_rows)
    summary = {
        "schema_version": PROTOCOL,
        "selector_states": str(args.selector_states.resolve()),
        "tool_train": str(args.tool_train.resolve()),
        "tool_val": str(args.tool_val.resolve()),
        "oracle_step": args.oracle_step,
        "synthetic_keep_source_step": args.synthetic_keep_source_step,
        "seed": args.seed,
        "train": train_summary,
        "val": val_summary,
        "train_task_count": len(train_tasks),
        "val_task_count": len(val_tasks),
        "task_overlap": 0,
        "behavior_policy_data": True,
        "on_policy_grpo_data": False,
        "test_assets_read": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
