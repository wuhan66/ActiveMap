#!/usr/bin/env python3
"""Evaluate a visual semantic-tool adapter on frozen validation SFT states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("adapter")
    parser.add_argument("validation_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--target-tool-only", action="store_true")
    return parser.parse_args()


def _target_operations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from activemap.agent.records import AgentAction
    from activemap.agent.vlm_evaluation import message_text, operation_from_action

    result = {}
    for row in rows:
        action = AgentAction.model_validate_json(message_text(row["messages"][2]))
        operation = operation_from_action(action)
        if operation is not None:
            result[str(row["example_id"])] = operation
    return result


def _observation(row: dict[str, Any]) -> dict[str, Any]:
    from activemap.agent.vlm_evaluation import message_text

    return json.loads(message_text(row["messages"][1]))


def _is_executable(action: Any, observation: dict[str, Any]) -> bool:
    if action.action.value == "USE_TOOL":
        if action.tool_call is None:
            return False
        return (
            action.tool_call.tool.value in observation.get("available_tools", [])
            and str(action.tool_call.inputs.get("evidence_id"))
            == str(observation.get("evidence_id"))
        )
    return action.action.value in {"COMMIT", "REJECT"}


def main() -> None:
    args = parse_args()
    import torch
    from peft import PeftModel
    from tqdm.auto import tqdm
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from activemap.agent.records import AgentAction
    from activemap.agent.tool_sft import terminal_reward
    from activemap.agent.vlm_evaluation import (
        action_class,
        extract_json_object,
        majority_operation,
        message_text,
        multiclass_metrics,
        operation_from_action,
        realized_tool_utility,
    )
    from activemap.agent.vlm_sft import load_vlm_sft_rows, materialize_vlm_messages
    from activemap.models import EditOperation

    rows = load_vlm_sft_rows(args.validation_jsonl)
    targets_by_example = _target_operations(rows)
    pre_rows = [row for row in rows if row.get("stage") == "PRE_TOOL"]
    if args.target_tool_only:
        pre_rows = [row for row in pre_rows if row.get("oracle_use_tool") is True]
    if args.limit is not None:
        pre_rows = pre_rows[: args.limit]
    if not pre_rows:
        raise ValueError("validation data contains no PRE_TOOL states")

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, args.adapter).to(args.device)
    model.eval()

    records = []
    operation_targets: list[str] = []
    operation_predictions: list[str] = []
    baseline_operation_predictions: list[str] = []
    forced_operation_predictions: list[str] = []
    action_targets: list[str] = []
    action_predictions: list[str] = []
    schema_valid_count = 0
    executable_count = 0
    predicted_tool_count = 0
    target_tool_count = 0
    true_tool_count = 0
    false_tool_count = 0
    baseline_utilities = []
    forced_utilities = []
    policy_utilities = []
    oracle_utilities = []
    policy_costs = []
    for row in tqdm(pre_rows, desc="Visual Agent validation"):
        example_id = str(row["example_id"])
        gt = targets_by_example.get(example_id)
        if gt is None:
            raise ValueError(f"missing terminal target for {example_id}")
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
        raw = processor.tokenizer.batch_decode(continuation, skip_special_tokens=True)[0]
        observation = _observation(row)
        consensus = row["consensus"]
        baseline_operation = EditOperation(consensus["baseline_operation"])
        semantic_operation = majority_operation(consensus["semantic_operations"])
        target_action = AgentAction.model_validate_json(message_text(row["messages"][2]))
        target_is_tool = target_action.action.value == "USE_TOOL"
        target_tool_count += int(target_is_tool)
        schema_valid = False
        executable = False
        predicted_action = None
        error = None
        try:
            predicted_action = AgentAction.model_validate(extract_json_object(raw))
            schema_valid = True
            executable = _is_executable(predicted_action, observation)
        except Exception as exception:
            error = str(exception)
        schema_valid_count += int(schema_valid)
        executable_count += int(executable)
        predicted_is_tool = bool(
            executable
            and predicted_action is not None
            and predicted_action.action.value == "USE_TOOL"
        )
        predicted_tool_count += int(predicted_is_tool)
        true_tool_count += int(target_is_tool and predicted_is_tool)
        false_tool_count += int(not target_is_tool and predicted_is_tool)
        if predicted_is_tool:
            predicted_operation = semantic_operation
            cost = float(observation["semantic_tool_cost"])
            policy_utility = realized_tool_utility(gt, predicted_operation, cost)
        elif executable and predicted_action is not None:
            predicted_operation = operation_from_action(predicted_action)
            assert predicted_operation is not None
            cost = 0.0
            policy_utility = terminal_reward(gt, predicted_operation)
        else:
            predicted_operation = baseline_operation
            cost = 0.0
            policy_utility = terminal_reward(gt, baseline_operation)
        baseline_utility = terminal_reward(gt, baseline_operation)
        forced_utility = realized_tool_utility(
            gt,
            semantic_operation,
            float(observation["semantic_tool_cost"]),
        )
        oracle_utility = max(baseline_utility, forced_utility)
        baseline_utilities.append(baseline_utility)
        forced_utilities.append(forced_utility)
        policy_utilities.append(policy_utility)
        oracle_utilities.append(oracle_utility)
        policy_costs.append(cost)
        operation_targets.append(gt.value)
        operation_predictions.append(predicted_operation.value)
        baseline_operation_predictions.append(baseline_operation.value)
        forced_operation_predictions.append(semantic_operation.value)
        action_targets.append(action_class(target_action))
        action_predictions.append(
            action_class(predicted_action) if executable and predicted_action else "INVALID"
        )
        records.append(
            {
                "example_id": example_id,
                "task_id": row["task_id"],
                "split": row["split"],
                "target_action": action_targets[-1],
                "predicted_action": action_predictions[-1],
                "target_operation": gt.value,
                "predicted_operation": predicted_operation.value,
                "policy_operation": predicted_operation.value,
                "baseline_operation": baseline_operation.value,
                "forced_operation": semantic_operation.value,
                "target_use_tool": target_is_tool,
                "predicted_use_tool": predicted_is_tool,
                "schema_valid": schema_valid,
                "executable": executable,
                "baseline_utility": baseline_utility,
                "forced_utility": forced_utility,
                "policy_utility": policy_utility,
                "oracle_utility": oracle_utility,
                "consensus_mean_utility_gain": float(consensus["mean_utility_gain"]),
                "tool_cost": cost,
                "raw_output": raw,
                "error": error,
            }
        )

    operation_metrics = multiclass_metrics(
        operation_targets, operation_predictions, [operation.value for operation in EditOperation]
    )
    baseline_operation_metrics = multiclass_metrics(
        operation_targets,
        baseline_operation_predictions,
        [operation.value for operation in EditOperation],
    )
    forced_operation_metrics = multiclass_metrics(
        operation_targets,
        forced_operation_predictions,
        [operation.value for operation in EditOperation],
    )
    action_labels = sorted(set(action_targets))
    action_metrics = multiclass_metrics(action_targets, action_predictions, action_labels)
    precision = true_tool_count / max(predicted_tool_count, 1)
    recall = true_tool_count / max(target_tool_count, 1)
    summary = {
        "schema_version": "semantic-vlm-action-evaluation-v1",
        "model": args.model,
        "adapter": args.adapter,
        "model_training_seed": args.seed,
        "validation_jsonl": str(args.validation_jsonl.resolve()),
        "sample_count": len(pre_rows),
        "schema_valid_rate": schema_valid_count / len(pre_rows),
        "executable_valid_rate": executable_count / len(pre_rows),
        "action_metrics": action_metrics,
        "operation_metrics": operation_metrics,
        "baseline_operation_metrics": baseline_operation_metrics,
        "forced_operation_metrics": forced_operation_metrics,
        "baseline_false_edit_rate": sum(
            target == EditOperation.KEEP.value and prediction != EditOperation.KEEP.value
            for target, prediction in zip(
                operation_targets, baseline_operation_predictions, strict=True
            )
        )
        / max(sum(target == EditOperation.KEEP.value for target in operation_targets), 1),
        "baseline_missed_edit_rate": sum(
            target != EditOperation.KEEP.value and prediction == EditOperation.KEEP.value
            for target, prediction in zip(
                operation_targets, baseline_operation_predictions, strict=True
            )
        )
        / max(sum(target != EditOperation.KEEP.value for target in operation_targets), 1),
        "forced_false_edit_rate": sum(
            target == EditOperation.KEEP.value and prediction != EditOperation.KEEP.value
            for target, prediction in zip(
                operation_targets, forced_operation_predictions, strict=True
            )
        )
        / max(sum(target == EditOperation.KEEP.value for target in operation_targets), 1),
        "forced_missed_edit_rate": sum(
            target != EditOperation.KEEP.value and prediction == EditOperation.KEEP.value
            for target, prediction in zip(
                operation_targets, forced_operation_predictions, strict=True
            )
        )
        / max(sum(target != EditOperation.KEEP.value for target in operation_targets), 1),
        "false_edit_rate": sum(
            target == EditOperation.KEEP.value and prediction != EditOperation.KEEP.value
            for target, prediction in zip(operation_targets, operation_predictions, strict=True)
        )
        / max(sum(target == EditOperation.KEEP.value for target in operation_targets), 1),
        "missed_edit_rate": sum(
            target != EditOperation.KEEP.value and prediction == EditOperation.KEEP.value
            for target, prediction in zip(operation_targets, operation_predictions, strict=True)
        )
        / max(sum(target != EditOperation.KEEP.value for target in operation_targets), 1),
        "tool_metrics": {
            "target_calls": target_tool_count,
            "predicted_calls": predicted_tool_count,
            "call_rate": predicted_tool_count / len(pre_rows),
            "precision": precision,
            "recall": recall,
            "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
            "false_call_rate": false_tool_count
            / max(len(pre_rows) - target_tool_count, 1),
            "mean_cost": sum(policy_costs) / len(policy_costs),
        },
        "mean_baseline_utility": sum(baseline_utilities) / len(baseline_utilities),
        "mean_forced_utility": sum(forced_utilities) / len(forced_utilities),
        "mean_policy_utility": sum(policy_utilities) / len(policy_utilities),
        "mean_oracle_utility": sum(oracle_utilities) / len(oracle_utilities),
        "mean_policy_gain": sum(
            policy - baseline
            for policy, baseline in zip(policy_utilities, baseline_utilities, strict=True)
        )
        / len(pre_rows),
        "test_assets_read": False,
        "limit": args.limit,
        "diagnostic_target_tool_only": args.target_tool_only,
        "utility_protocol": "realized-terminal-reward-minus-executed-tool-cost",
        "consensus_gain_use": "training-target-construction-only",
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
