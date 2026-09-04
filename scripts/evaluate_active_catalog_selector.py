#!/usr/bin/env python3
"""Evaluate executable active-catalog SELECT actions under utility and cost."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from activemap.agent.sequential_controller import (
    ControllerStage,
    SelectionDecision,
    SequentialControllerAction,
)
from activemap.agent.vlm_sft import load_vlm_sft_rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(message: dict[str, Any]) -> str:
    return next(
        part["text"] for part in message["content"] if part.get("type") == "text"
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object found")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("generated action is not a JSON object")
    return value


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def active_catalog_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("active-catalog evaluation is empty")
    target = [row["target_selection"] == "ACQUIRE" for row in rows]
    predicted = [row["predicted_selection"] == "ACQUIRE" for row in rows]
    tp = sum(left and right for left, right in zip(target, predicted, strict=True))
    fp = sum(not left and right for left, right in zip(target, predicted, strict=True))
    fn = sum(left and not right for left, right in zip(target, predicted, strict=True))
    tn = len(rows) - tp - fp - fn
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    stop_precision = tn / max(tn + fn, 1)
    stop_recall = tn / max(tn + fp, 1)
    acquire_f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    stop_f1 = 2 * stop_precision * stop_recall / max(stop_precision + stop_recall, 1e-12)
    exact = sum(
        row["target_selection"] == "ACQUIRE"
        and row["predicted_selection"] == "ACQUIRE"
        and row.get("predicted_evidence_id") == row.get("target_evidence_id")
        for row in rows
    )
    target_calls = sum(target)
    predicted_calls = sum(predicted)
    utilities = [float(row["policy_utility"]) for row in rows]
    oracle = [float(row["oracle_utility"]) for row in rows]
    regrets = [float(row["regret"]) for row in rows]
    costs = [float(row["policy_cost"]) for row in rows]
    harmful = sum(
        call and utility < float(row["stop_utility"]) - 1e-9
        for row, call, utility in zip(rows, predicted, utilities, strict=True)
    )
    always_stop_macro_f1 = (2.0 * (1.0 - sum(target) / len(rows))) / (
        2.0 - sum(target) / len(rows)
    ) / 2.0
    random_exact = sum(
        1.0 / float(row["candidate_count"])
        for row in rows
        if row["target_selection"] == "ACQUIRE"
    ) / max(target_calls, 1)
    return {
        "states": float(len(rows)),
        "selection_accuracy": (tp + tn) / len(rows),
        "selection_macro_f1": (acquire_f1 + stop_f1) / 2.0,
        "selection_macro_f1_delta_vs_always_stop": (
            (acquire_f1 + stop_f1) / 2.0 - always_stop_macro_f1
        ),
        "always_stop_macro_f1": always_stop_macro_f1,
        "acquire_precision": precision,
        "acquire_recall": recall,
        "acquire_f1": acquire_f1,
        "stop_f1": stop_f1,
        "target_call_rate": target_calls / len(rows),
        "predicted_call_rate": predicted_calls / len(rows),
        "false_call_rate": fp / max(len(rows) - target_calls, 1),
        "missed_call_rate": fn / max(target_calls, 1),
        "exact_evidence_recall": exact / max(target_calls, 1),
        "random_exact_evidence_recall": random_exact,
        "harmful_call_rate_all_states": harmful / len(rows),
        "harmful_call_fraction_of_calls": harmful / max(predicted_calls, 1),
        "realized_utility_sum": sum(utilities),
        "realized_utility_mean": sum(utilities) / len(rows),
        "oracle_utility_mean": sum(oracle) / len(rows),
        "mean_regret": sum(regrets) / len(rows),
        "mean_cost_per_state": sum(costs) / len(rows),
        "mean_cost_per_call": sum(costs) / max(predicted_calls, 1),
        "utility_per_unit_cost": sum(utilities) / max(sum(costs), 1e-12),
    }


def _project_policy(
    rows: list[dict[str, Any]], prefix: str
) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "predicted_selection": row[f"{prefix}_selection"],
            "predicted_evidence_id": row.get(f"{prefix}_evidence_id"),
            "policy_utility": row[f"{prefix}_utility"],
            "policy_cost": row[f"{prefix}_cost"],
            "regret": row["oracle_utility"] - row[f"{prefix}_utility"],
        }
        for row in rows
    ]


def grouped_bootstrap(
    rows: list[dict[str, Any]], *, group_key: str, repetitions: int, seed: int
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(row)
    groups = sorted(grouped)
    if len(groups) < 2 or repetitions <= 0:
        raise ValueError("grouped bootstrap requires multiple groups and repetitions")
    keys = (
        "selection_macro_f1_delta_vs_always_stop",
        "realized_utility_mean",
        "mean_regret",
        "false_call_rate",
        "harmful_call_rate_all_states",
        "exact_evidence_recall",
    )
    observed = active_catalog_metrics(rows)
    distributions = {key: [] for key in keys}
    rng = random.Random(seed)
    for _ in range(repetitions):
        sampled = rng.choices(groups, k=len(groups))
        metrics = active_catalog_metrics([row for group in sampled for row in grouped[group]])
        for key in keys:
            distributions[key].append(metrics[key])
    intervals = {
        key: {
            "observed": observed[key],
            "ci95_low": _quantile(values, 0.025),
            "ci95_high": _quantile(values, 0.975),
        }
        for key, values in distributions.items()
    }
    for key in ("selection_macro_f1_delta_vs_always_stop", "realized_utility_mean"):
        intervals[key]["bootstrap_probability_gt_zero"] = sum(
            value > 0 for value in distributions[key]
        ) / repetitions
    return {
        "group_key": group_key,
        "group_count": len(groups),
        "repetitions": repetitions,
        "seed": seed,
        "intervals": intervals,
    }


def _stratified(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = sorted({str(row[key]) for row in rows})
    return {
        value: active_catalog_metrics([row for row in rows if str(row[key]) == value])
        for value in values
    }


def promotion_gate(
    valid_rate: float,
    metrics: dict[str, float],
    aoi_bootstrap: dict[str, Any],
) -> dict[str, bool]:
    requirements = {
        "valid_action_rate_at_least_0_99": valid_rate >= 0.99,
        "nonzero_calls": metrics["predicted_call_rate"] > 0.0,
        "macro_f1_above_always_stop": metrics[
            "selection_macro_f1_delta_vs_always_stop"
        ]
        > 0.0,
        "utility_positive": metrics["realized_utility_mean"] > 0.0,
        "utility_aoi_bootstrap_ci_above_zero": aoi_bootstrap["intervals"][
            "realized_utility_mean"
        ]["ci95_low"]
        > 0.0,
        "false_call_rate_at_most_0_10": metrics["false_call_rate"] <= 0.10,
        "exact_recall_above_random": metrics["exact_evidence_recall"]
        > metrics["random_exact_evidence_recall"],
    }
    return {
        **requirements,
        "valid_action_rate_1_audit": valid_rate == 1.0,
        "passed": all(requirements.values()),
    }


def _evaluation_index(path: Path) -> dict[str, Any]:
    result = {}
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    for row in rows:
        if row.get("split") != "val" or row.get("model_visible") is not False:
            raise ValueError("evaluation index must be validation-only and model-hidden")
        if row.get("test_assets_read") is not False:
            raise ValueError("evaluation index reports test access")
        example_id = str(row["example_id"])
        if example_id in result:
            raise ValueError(f"duplicate evaluation example id: {example_id}")
        result[example_id] = row
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("val_jsonl", type=Path)
    parser.add_argument("evaluation_index", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--expected-records", type=int)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    import torch
    from peft import PeftModel
    from tqdm.auto import tqdm
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from activemap.agent.vlm_sft import materialize_vlm_messages

    rows = load_vlm_sft_rows(args.val_jsonl)
    if args.expected_records is not None and len(rows) != args.expected_records:
        raise ValueError(f"expected {args.expected_records} records, found {len(rows)}")
    if {row["split"] for row in rows} != {"val"}:
        raise ValueError("active-catalog evaluation only permits validation records")
    evaluation_index = _evaluation_index(args.evaluation_index)
    row_ids = {str(row["example_id"]) for row in rows}
    if row_ids != set(evaluation_index):
        raise ValueError("validation SFT and evaluation index do not have identical examples")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    base = AutoModelForImageTextToText.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, args.adapter).to(args.device).eval()
    traces = []
    for row in tqdm(rows, desc="Active-catalog validation"):
        example_id = str(row["example_id"])
        evaluation = evaluation_index.get(example_id)
        if evaluation is None:
            raise ValueError(f"missing validation evaluation index: {example_id}")
        state = json.loads(_text(row["messages"][1]))
        candidates = state["candidate_evidence"]
        candidate_ids = [str(item["evidence_id"]) for item in candidates]
        utility_by_id = {
            str(item["evidence_id"]): float(item["utility"])
            for item in evaluation["candidates"]
        }
        cost_by_id = {
            str(item["evidence_id"]): float(item["cost"])
            for item in evaluation["candidates"]
        }
        if set(candidate_ids) != set(utility_by_id):
            raise ValueError(f"prompt/evaluation candidates disagree: {example_id}")
        target = SequentialControllerAction.model_validate_json(_text(row["messages"][2]))
        if (
            target.selection.value != evaluation["target_selection"]
            or target.evidence_id != evaluation.get("target_evidence_id")
        ):
            raise ValueError(f"SFT target and evaluation index disagree: {example_id}")
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
        error = None
        try:
            predicted = SequentialControllerAction.model_validate(_extract_json_object(raw))
            if predicted.stage != ControllerStage.SELECT:
                raise ValueError("generated action is not SELECT")
            if (
                predicted.selection == SelectionDecision.ACQUIRE
                and predicted.evidence_id not in candidate_ids
            ):
                raise ValueError("generated evidence_id is unavailable")
        except Exception as exception:
            predicted = SequentialControllerAction(
                stage=ControllerStage.SELECT, selection=SelectionDecision.STOP
            )
            error = str(exception)

        stop_utility = float(evaluation["stop_utility"])
        oracle_utility = max(
            stop_utility, *(utility_by_id[value] for value in candidate_ids)
        )
        predicted_id = predicted.evidence_id
        policy_utility = (
            utility_by_id[str(predicted_id)]
            if predicted.selection == SelectionDecision.ACQUIRE
            else stop_utility
        )
        policy_cost = (
            cost_by_id[str(predicted_id)]
            if predicted.selection == SelectionDecision.ACQUIRE
            else 0.0
        )
        cheapest = min(candidate_ids, key=lambda value: (cost_by_id[value], value))
        clear_per_cost = max(
            candidates,
            key=lambda item: (
                float(item["clear_fraction"]) / max(float(item["cost"]), 1e-12),
                str(item["evidence_id"]),
            ),
        )["evidence_id"]
        random_id = random.Random(f"active-catalog-random-v1|{example_id}").choice(
            candidate_ids
        )
        target_id = target.evidence_id
        trace = {
            "example_id": example_id,
            "task_id": row["task_id"],
            "aoi_id": str(evaluation["aoi_id"]),
            "source_episode": str(evaluation["source_episode"]),
            "split": "val",
            "budget": float(evaluation["budget"]),
            "oracle_step": int(evaluation["oracle_step"]),
            "gt_edit": str(evaluation["gt_edit"]),
            "draft_edit": str(evaluation["draft_edit"]),
            "candidate_count": len(candidate_ids),
            "target_selection": target.selection.value,
            "target_evidence_id": target_id,
            "predicted_selection": predicted.selection.value,
            "predicted_evidence_id": predicted_id,
            "valid_action": error is None,
            "executable_action": error is None,
            "stop_utility": stop_utility,
            "oracle_utility": oracle_utility,
            "policy_utility": policy_utility,
            "policy_cost": policy_cost,
            "regret": oracle_utility - policy_utility,
            "raw_output": raw,
            "parse_error": error,
        }
        for name, evidence_id in {
            "cheapest": cheapest,
            "clear_per_cost": str(clear_per_cost),
            "random": random_id,
        }.items():
            trace[f"{name}_selection"] = "ACQUIRE"
            trace[f"{name}_evidence_id"] = evidence_id
            trace[f"{name}_utility"] = utility_by_id[evidence_id]
            trace[f"{name}_cost"] = cost_by_id[evidence_id]
        trace.update(
            {
                "always_stop_selection": "STOP",
                "always_stop_evidence_id": None,
                "always_stop_utility": stop_utility,
                "always_stop_cost": 0.0,
                "oracle_selection": target.selection.value,
                "oracle_evidence_id": target_id,
                "oracle_cost": cost_by_id[str(target_id)] if target_id else 0.0,
            }
        )
        traces.append(trace)

    metrics = active_catalog_metrics(traces)
    baselines = {
        name: active_catalog_metrics(_project_policy(traces, name))
        for name in ("always_stop", "cheapest", "clear_per_cost", "random", "oracle")
    }
    valid_rate = sum(row["valid_action"] for row in traces) / len(traces)
    aoi_bootstrap = grouped_bootstrap(
        traces,
        group_key="aoi_id",
        repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    episode_bootstrap = grouped_bootstrap(
        traces,
        group_key="source_episode",
        repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    promotion = promotion_gate(valid_rate, metrics, aoi_bootstrap)
    args.output_dir.mkdir(parents=True)
    traces_path = args.output_dir / "traces.jsonl"
    with traces_path.open("w", encoding="utf-8") as handle:
        for row in traces:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {
        "schema_version": "active-catalog-selector-evaluation-v1",
        "evaluation_role": "component_teacher_forced_belief_active_evidence_selection",
        "sample_count": len(traces),
        "aoi_count": len({row["aoi_id"] for row in traces}),
        "episode_count": len({row["source_episode"] for row in traces}),
        "valid_action_rate": valid_rate,
        "metrics": metrics,
        "baselines": baselines,
        "by_budget": _stratified(traces, "budget"),
        "by_ground_truth_edit": _stratified(traces, "gt_edit"),
        "by_draft_edit": _stratified(traces, "draft_edit"),
        "aoi_bootstrap": aoi_bootstrap,
        "episode_bootstrap": episode_bootstrap,
        "promotion_gate": promotion,
        "sources": {
            "model": str(args.model.resolve()),
            "adapter": str(args.adapter.resolve()),
            "validation": {
                "path": str(args.val_jsonl.resolve()),
                "sha256": _sha256(args.val_jsonl),
            },
            "evaluation_index": {
                "path": str(args.evaluation_index.resolve()),
                "sha256": _sha256(args.evaluation_index),
            },
            "trace_sha256": _sha256(traces_path),
        },
        "test_assets_read": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
