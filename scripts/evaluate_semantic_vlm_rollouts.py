#!/usr/bin/env python3
"""Run cached two-stage visual tool-to-belief rollouts on validation states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("adapter")
    parser.add_argument("rollout_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def index_pairs(rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    order = []
    for row in rows:
        example_id = str(row["example_id"])
        if example_id not in grouped:
            grouped[example_id] = {}
            order.append(example_id)
        stage = str(row.get("stage"))
        if stage in grouped[example_id]:
            raise ValueError(f"duplicate {stage} state for {example_id}")
        grouped[example_id][stage] = row
    pairs = []
    for example_id in order:
        stages = grouped[example_id]
        if set(stages) != {"PRE_TOOL", "POST_TOOL"}:
            raise ValueError(f"incomplete PRE/POST pair for {example_id}")
        pairs.append((stages["PRE_TOOL"], stages["POST_TOOL"]))
    return pairs


def _generate(model: Any, processor: Any, row: dict[str, Any], args: argparse.Namespace) -> str:
    import torch

    from activemap.agent.vlm_sft import materialize_vlm_messages

    messages = materialize_vlm_messages(row["messages"][:2])
    encoded = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(args.device)
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    continuation = generated[:, encoded["input_ids"].shape[1] :]
    return processor.tokenizer.batch_decode(continuation, skip_special_tokens=True)[0]


def _parse_action(text: str) -> tuple[Any | None, str | None]:
    from activemap.agent.records import AgentAction
    from activemap.agent.vlm_evaluation import extract_json_object

    try:
        return AgentAction.model_validate(extract_json_object(text)), None
    except Exception as exception:
        return None, str(exception)


def _pre_executable(action: Any, observation: dict[str, Any]) -> bool:
    if action is None:
        return False
    if action.action.value == "USE_TOOL":
        return bool(
            action.tool_call is not None
            and action.tool_call.tool.value in observation.get("available_tools", [])
            and str(action.tool_call.inputs.get("evidence_id"))
            == str(observation.get("evidence_id"))
        )
    return action.action.value in {"COMMIT", "REJECT"}


def _terminal_operation(action: Any) -> Any | None:
    from activemap.agent.vlm_evaluation import operation_from_action

    if action is None or action.action.value not in {"COMMIT", "REJECT"}:
        return None
    return operation_from_action(action)


def _rates(targets: list[str], predictions: list[str]) -> tuple[float, float]:
    keep = "KEEP"
    false_edit = sum(t == keep and p != keep for t, p in zip(targets, predictions, strict=True))
    missed = sum(t != keep and p == keep for t, p in zip(targets, predictions, strict=True))
    return (
        false_edit / max(sum(target == keep for target in targets), 1),
        missed / max(sum(target != keep for target in targets), 1),
    )


def main() -> None:
    args = parse_args()
    import torch
    from peft import PeftModel
    from tqdm.auto import tqdm
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from activemap.agent.records import AgentAction
    from activemap.agent.tool_sft import terminal_reward
    from activemap.agent.vlm_evaluation import (
        majority_operation,
        message_text,
        multiclass_metrics,
        operation_from_action,
        realized_tool_utility,
    )
    from activemap.agent.vlm_sft import load_vlm_sft_rows
    from activemap.models import EditOperation

    rows = load_vlm_sft_rows(args.rollout_jsonl)
    pairs = index_pairs(rows)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, args.adapter).to(args.device)
    model.eval()

    targets = []
    policy_predictions = []
    baseline_predictions = []
    forced_predictions = []
    baseline_utilities = []
    forced_utilities = []
    policy_utilities = []
    oracle_utilities = []
    tool_costs = []
    traces = []
    pre_schema_valid = 0
    pre_executable = 0
    post_schema_valid = 0
    post_terminal_valid = 0
    target_calls = 0
    predicted_calls = 0
    true_calls = 0
    false_calls = 0
    for pre, post in tqdm(pairs, desc="Visual Agent rollouts"):
        target_action = AgentAction.model_validate_json(message_text(post["messages"][2]))
        gt = operation_from_action(target_action)
        assert gt is not None
        pre_observation = json.loads(message_text(pre["messages"][1]))
        consensus = pre["consensus"]
        baseline = EditOperation(consensus["baseline_operation"])
        forced = majority_operation(consensus["semantic_operations"])
        baseline_utility = terminal_reward(gt, baseline)
        forced_utility = realized_tool_utility(
            gt,
            forced,
            float(pre_observation["semantic_tool_cost"]),
        )
        oracle_utility = max(baseline_utility, forced_utility)
        target_call = bool(pre["oracle_use_tool"])
        target_calls += int(target_call)

        raw_pre = _generate(model, processor, pre, args)
        pre_action, pre_error = _parse_action(raw_pre)
        pre_schema_valid += int(pre_action is not None)
        executable = _pre_executable(pre_action, pre_observation)
        pre_executable += int(executable)
        called = bool(executable and pre_action.action.value == "USE_TOOL")
        predicted_calls += int(called)
        true_calls += int(called and target_call)
        false_calls += int(called and not target_call)
        raw_post = None
        post_error = None
        post_action = None
        if called:
            raw_post = _generate(model, processor, post, args)
            post_action, post_error = _parse_action(raw_post)
            post_schema_valid += int(post_action is not None)
            operation = _terminal_operation(post_action)
            post_terminal_valid += int(operation is not None)
            cost = float(pre_observation["semantic_tool_cost"])
            if operation is None:
                operation = baseline
        elif executable:
            operation = _terminal_operation(pre_action)
            assert operation is not None
            cost = 0.0
        else:
            operation = baseline
            cost = 0.0
        policy_utility = terminal_reward(gt, operation) - cost
        targets.append(gt.value)
        policy_predictions.append(operation.value)
        baseline_predictions.append(baseline.value)
        forced_predictions.append(forced.value)
        baseline_utilities.append(baseline_utility)
        forced_utilities.append(forced_utility)
        policy_utilities.append(policy_utility)
        oracle_utilities.append(oracle_utility)
        tool_costs.append(cost)
        traces.append(
            {
                "example_id": pre["example_id"],
                "task_id": pre["task_id"],
                "split": pre["split"],
                "evidence_id": pre_observation["evidence_id"],
                "target_operation": gt.value,
                "baseline_operation": baseline.value,
                "forced_operation": forced.value,
                "policy_operation": operation.value,
                "target_use_tool": target_call,
                "predicted_use_tool": called,
                "pre_schema_valid": pre_action is not None,
                "pre_executable": executable,
                "post_schema_valid": post_action is not None if called else None,
                "post_terminal_valid": _terminal_operation(post_action) is not None
                if called
                else None,
                "baseline_utility": baseline_utility,
                "forced_utility": forced_utility,
                "policy_utility": policy_utility,
                "oracle_utility": oracle_utility,
                "tool_cost": cost,
                "raw_pre": raw_pre,
                "raw_post": raw_post,
                "pre_error": pre_error,
                "post_error": post_error,
            }
        )

    labels = [operation.value for operation in EditOperation]
    policy_metrics = multiclass_metrics(targets, policy_predictions, labels)
    baseline_metrics = multiclass_metrics(targets, baseline_predictions, labels)
    forced_metrics = multiclass_metrics(targets, forced_predictions, labels)
    false_edit, missed_edit = _rates(targets, policy_predictions)
    baseline_false_edit, baseline_missed = _rates(targets, baseline_predictions)
    forced_false_edit, forced_missed = _rates(targets, forced_predictions)
    precision = true_calls / max(predicted_calls, 1)
    recall = true_calls / max(target_calls, 1)
    summary = {
        "schema_version": "semantic-vlm-cached-rollout-evaluation-v1",
        "model": args.model,
        "adapter": args.adapter,
        "model_training_seed": args.seed,
        "validation_jsonl": str(args.rollout_jsonl.resolve()),
        "sample_count": len(pairs),
        "schema_valid_rate": pre_schema_valid / len(pairs),
        "executable_valid_rate": pre_executable / len(pairs),
        "post_schema_valid_rate": post_schema_valid / max(predicted_calls, 1),
        "post_terminal_valid_rate": post_terminal_valid / max(predicted_calls, 1),
        "operation_metrics": policy_metrics,
        "baseline_operation_metrics": baseline_metrics,
        "forced_operation_metrics": forced_metrics,
        "false_edit_rate": false_edit,
        "baseline_false_edit_rate": baseline_false_edit,
        "forced_false_edit_rate": forced_false_edit,
        "missed_edit_rate": missed_edit,
        "baseline_missed_edit_rate": baseline_missed,
        "forced_missed_edit_rate": forced_missed,
        "mean_baseline_utility": sum(baseline_utilities) / len(pairs),
        "mean_forced_utility": sum(forced_utilities) / len(pairs),
        "mean_policy_utility": sum(policy_utilities) / len(pairs),
        "mean_oracle_utility": sum(oracle_utilities) / len(pairs),
        "mean_policy_gain": sum(
            policy - baseline
            for policy, baseline in zip(policy_utilities, baseline_utilities, strict=True)
        )
        / len(pairs),
        "tool_metrics": {
            "target_calls": target_calls,
            "predicted_calls": predicted_calls,
            "call_rate": predicted_calls / len(pairs),
            "precision": precision,
            "recall": recall,
            "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
            "false_call_rate": false_calls / max(len(pairs) - target_calls, 1),
            "mean_cost": sum(tool_costs) / len(pairs),
        },
        "test_assets_read": False,
        "limit": args.limit,
        "fallback_protocol": "invalid-pre-or-post-preserves-baseline-operation",
        "max_tool_calls_per_episode": 1,
        "visual_evidence_selected_for_writeback": True,
        "utility_protocol": "realized-terminal-reward-minus-executed-tool-cost",
        "consensus_gain_use": "training-target-construction-only",
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "traces.jsonl").open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace, separators=(",", ":")) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
