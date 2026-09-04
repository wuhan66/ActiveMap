#!/usr/bin/env python3
"""Evaluate structured controller actions without exposing chain-of-thought text."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _extract_json(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object found")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("action output is not an object")
    return value


def _action_key(action: Any) -> str:
    if action.action.value == "USE_TOOL":
        evidence_id = (
            action.tool_call.inputs.get("evidence_id")
            if action.tool_call is not None
            else None
        )
        tool = action.tool_call.tool.value if action.tool_call is not None else "UNKNOWN"
        return f"USE_TOOL:{tool}:{evidence_id or 'UNKNOWN'}"
    return action.key


def _action_class(action: Any) -> str:
    if action.action.value == "COMMIT":
        return f"COMMIT:{action.edit.value}"
    return action.action.value


def _is_executable(action: Any, observation: dict[str, Any]) -> bool:
    if action.action.value == "ACQUIRE":
        candidate_ids = {
            str(candidate["evidence_id"]) for candidate in observation.get("candidates", [])
        }
        return action.evidence_id in candidate_ids
    if action.action.value == "USE_TOOL":
        available = {str(value) for value in observation.get("available_tools", [])}
        selected = {str(value) for value in observation.get("selected_evidence_ids", [])}
        if action.tool_call is None or action.tool_call.tool.value not in available:
            return False
        evidence_id = action.tool_call.inputs.get("evidence_id")
        return evidence_id is not None and str(evidence_id) in selected
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("sft_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--adapter")
    parser.add_argument("--trajectories-jsonl", type=Path)
    args = parser.parse_args()

    import torch
    from tqdm.auto import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from activemap.agent.records import AgentAction

    rows = []
    with args.sft_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows.append(row)
            if args.limit is not None and len(rows) >= args.limit:
                break
    utility_by_state: dict[tuple[str, int], dict[str, float]] = {}
    if args.trajectories_jsonl is not None:
        with args.trajectories_jsonl.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                trajectory = json.loads(line)
                trajectory_id = str(trajectory["trajectory_id"])
                for transition in trajectory["transitions"]:
                    step = int(transition["observation"]["step"])
                    utility_by_state[(trajectory_id, step)] = {
                        str(key): float(value)
                        for key, value in transition.get("oracle_utilities", {}).items()
                    }
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    if args.adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model = model.to(args.device)
    model.eval()

    predictions = []
    confusion: Counter[tuple[str, str]] = Counter()
    valid_count = 0
    executable_count = 0
    exact_count = 0
    policy_utilities = []
    oracle_utilities = []
    invalid_utility_count = 0
    target_tool_count = 0
    predicted_tool_count = 0
    executable_tool_count = 0
    correct_tool_count = 0
    false_tool_call_count = 0
    no_tool_target_count = 0
    tool_positive_state_count = 0
    tool_positive_exact_count = 0
    for offset in tqdm(range(0, len(rows), args.batch_size), desc="Agent evaluation"):
        batch = rows[offset : offset + args.batch_size]
        prompts = []
        for row in batch:
            messages = row["messages"][:2]
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            prompts.append(prompt)
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        ).to(args.device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        continuation = generated[:, encoded["input_ids"].shape[1] :]
        texts = tokenizer.batch_decode(continuation, skip_special_tokens=True)
        for row, text in zip(batch, texts, strict=True):
            target = AgentAction.model_validate_json(row["messages"][2]["content"])
            target_key = _action_key(target)
            target_class = _action_class(target)
            observation = json.loads(row["messages"][1]["content"])
            schema_valid = False
            executable = False
            predicted_key = "INVALID"
            predicted_class = "INVALID"
            predicted = None
            error = None
            try:
                predicted = AgentAction.model_validate(_extract_json(text))
                predicted_key = _action_key(predicted)
                predicted_class = _action_class(predicted)
                schema_valid = True
                executable = _is_executable(predicted, observation)
                if not executable:
                    predicted_class = "INVALID"
            except Exception as exc:
                error = str(exc)
            valid_count += int(schema_valid)
            executable_count += int(executable)
            exact_count += int(executable and predicted_key == target_key)
            target_is_tool = target.action.value == "USE_TOOL"
            predicted_is_tool = (
                predicted is not None and predicted.action.value == "USE_TOOL"
            )
            target_tool_count += int(target_is_tool)
            predicted_tool_count += int(predicted_is_tool)
            executable_tool_count += int(predicted_is_tool and executable)
            correct_tool_count += int(
                target_is_tool and executable and predicted_key == target_key
            )
            no_tool_target_count += int(not target_is_tool)
            false_tool_call_count += int(
                not target_is_tool and predicted_is_tool and executable
            )
            tool_positive = int(row.get("oracle_tool_stage", 0)) > 0
            tool_positive_state_count += int(tool_positive)
            tool_positive_exact_count += int(
                tool_positive and executable and predicted_key == target_key
            )
            state_utilities = utility_by_state.get(
                (str(row["trajectory_id"]), int(row["step"]))
            )
            predicted_utility = None
            oracle_utility = None
            if state_utilities:
                oracle_utility = max(state_utilities.values())
                if executable and predicted_key in state_utilities:
                    predicted_utility = state_utilities[predicted_key]
                else:
                    predicted_utility = min(state_utilities.values())
                    invalid_utility_count += 1
                policy_utilities.append(predicted_utility)
                oracle_utilities.append(oracle_utility)
            confusion[(target_class, predicted_class)] += 1
            predictions.append(
                {
                    "trajectory_id": row["trajectory_id"],
                    "step": row["step"],
                    "target": target_key,
                    "target_class": target_class,
                    "prediction": predicted_key,
                    "prediction_class": predicted_class,
                    "schema_valid": schema_valid,
                    "executable": executable,
                    "error": error,
                    "raw_output": text,
                    "predicted_utility": predicted_utility,
                    "oracle_utility": oracle_utility,
                }
            )

    labels = sorted({target for target, _ in confusion})
    per_action = {}
    f1_values = []
    for label in labels:
        true_positive = confusion[(label, label)]
        false_positive = sum(
            count for (target, prediction), count in confusion.items()
            if prediction == label and target != label
        )
        false_negative = sum(
            count for (target, prediction), count in confusion.items()
            if target == label and prediction != label
        )
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        f1_values.append(f1)
        per_action[label] = {
            "support": sum(count for (target, _), count in confusion.items() if target == label),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "dataset": str(args.sft_jsonl.resolve()),
        "sample_count": len(rows),
        "schema_valid_rate": valid_count / max(len(rows), 1),
        "executable_valid_rate": executable_count / max(len(rows), 1),
        "exact_action_accuracy": exact_count / max(len(rows), 1),
        "macro_f1": sum(f1_values) / max(len(f1_values), 1),
        "per_action": per_action,
        "tool_metrics": {
            "target_call_count": target_tool_count,
            "predicted_call_count": predicted_tool_count,
            "executable_call_count": executable_tool_count,
            "grounded_call_accuracy": correct_tool_count / max(target_tool_count, 1),
            "false_call_rate": false_tool_call_count / max(no_tool_target_count, 1),
            "tool_positive_state_count": tool_positive_state_count,
            "tool_positive_exact_accuracy": (
                tool_positive_exact_count / max(tool_positive_state_count, 1)
            ),
            "exact_key_includes_tool_and_evidence_id": True,
        },
        "utility_state_count": len(policy_utilities),
        "mean_policy_utility": (
            sum(policy_utilities) / len(policy_utilities) if policy_utilities else None
        ),
        "mean_oracle_utility": (
            sum(oracle_utilities) / len(oracle_utilities) if oracle_utilities else None
        ),
        "mean_regret": (
            sum(
                oracle - policy
                for oracle, policy in zip(
                    oracle_utilities, policy_utilities, strict=True
                )
            )
            / len(policy_utilities)
            if policy_utilities
            else None
        ),
        "invalid_utility_count": invalid_utility_count,
        "invalid_utility_protocol": (
            "minimum legal state utility" if utility_by_state else None
        ),
        "limit": args.limit,
        "test_assets_read": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, separators=(",", ":")) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
