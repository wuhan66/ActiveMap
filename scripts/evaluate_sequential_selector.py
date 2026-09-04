#!/usr/bin/env python3
"""Evaluate a staged SELECT controller on frozen direct-draft validation states."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from activemap.agent.sequential_controller import (
    ControllerStage,
    SelectionDecision,
    SequentialControllerAction,
)
from activemap.agent.vlm_evaluation import extract_json_object


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selector_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("selector evaluation is empty")
    target = [row["target_selection"] == "ACQUIRE" for row in rows]
    predicted = [row["predicted_selection"] == "ACQUIRE" for row in rows]
    true_positive = sum(left and right for left, right in zip(target, predicted, strict=True))
    false_positive = sum(not left and right for left, right in zip(target, predicted, strict=True))
    false_negative = sum(left and not right for left, right in zip(target, predicted, strict=True))
    true_negative = len(rows) - true_positive - false_positive - false_negative
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    stop_precision = true_negative / max(true_negative + false_negative, 1)
    stop_recall = true_negative / max(true_negative + false_positive, 1)
    acquire_f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    stop_f1 = 2.0 * stop_precision * stop_recall / max(stop_precision + stop_recall, 1e-12)
    utilities = [
        float(row["policy_relative_advantage"]) if prediction else 0.0
        for row, prediction in zip(rows, predicted, strict=True)
    ]
    risks = [max(-utility, 0.0) for utility in utilities]
    return {
        "accuracy": (true_positive + true_negative) / len(rows),
        "macro_f1": (acquire_f1 + stop_f1) / 2.0,
        "acquire_precision": precision,
        "acquire_recall": recall,
        "acquire_f1": acquire_f1,
        "stop_f1": stop_f1,
        "target_call_rate": sum(target) / len(rows),
        "predicted_call_rate": sum(predicted) / len(rows),
        "predicted_calls": sum(predicted),
        "target_calls": sum(target),
        "true_calls": true_positive,
        "false_calls": false_positive,
        "false_call_rate": false_positive / max(sum(not value for value in target), 1),
        "missed_call_rate": false_negative / max(sum(target), 1),
        "realized_utility_sum": sum(utilities),
        "realized_utility_mean": sum(utilities) / len(rows),
        "realized_risk_sum": sum(risks),
        "realized_risk_mean": sum(risks) / len(rows),
    }


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def task_bootstrap(
    rows: list[dict[str, Any]], *, repetitions: int, seed: int
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_id"])].append(row)
    tasks = sorted(grouped)
    if len(tasks) < 2 or repetitions <= 0:
        raise ValueError("selector bootstrap requires multiple tasks and positive repetitions")
    observed = selector_metrics(rows)
    keys = ("macro_f1", "realized_utility_mean", "false_call_rate", "missed_call_rate")
    distributions = {key: [] for key in keys}
    rng = random.Random(seed)
    for _ in range(repetitions):
        sampled = rng.choices(tasks, k=len(tasks))
        metrics = selector_metrics([row for task_id in sampled for row in grouped[task_id]])
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
    for key in ("macro_f1", "realized_utility_mean"):
        intervals[key]["bootstrap_probability_gt_zero"] = sum(
            value > 0.0 for value in distributions[key]
        ) / repetitions
    return {
        "grouping_unit": "task_id",
        "task_count": len(tasks),
        "repetitions": repetitions,
        "seed": seed,
        "intervals": intervals,
    }


def _assistant_action(row: dict[str, Any]) -> SequentialControllerAction:
    content = row["messages"][2]["content"]
    text = next(part["text"] for part in content if part.get("type") == "text")
    return SequentialControllerAction.model_validate_json(text)


def _visible_select_evidence_id(row: dict[str, Any]) -> str | None:
    """Return the unique evidence ID explicitly exposed in a SELECT prompt.

    This deliberately reads only the observable user payload, never the
    assistant target or the validation advantage.  The MUNO21 SELECT protocol
    presents one evidence candidate per state, so an otherwise complete
    ``ACQUIRE`` action may be made executable by filling in that public ID.
    """

    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    user = messages[1]
    if user.get("role") != "user" or not isinstance(user.get("content"), list):
        return None
    ids: set[str] = set()
    for part in user["content"]:
        if part.get("type") != "text" or not isinstance(part.get("text"), str):
            continue
        try:
            payload = json.loads(part["text"])
        except json.JSONDecodeError:
            continue
        evidence_id = payload.get("evidence_id") if isinstance(payload, dict) else None
        if isinstance(evidence_id, str) and evidence_id:
            ids.add(evidence_id)
    return next(iter(ids)) if len(ids) == 1 else None


@dataclass(frozen=True)
class CanonicalizedSelectAction:
    """Raw and executable status for a generated SELECT action."""

    action: SequentialControllerAction | None
    raw_json_valid: bool
    raw_schema_valid: bool
    evidence_id_hydrated: bool
    raw_json_error: str | None
    raw_schema_error: str | None
    executable_action_error: str | None


def canonicalize_select_action(raw: str, row: dict[str, Any]) -> CanonicalizedSelectAction:
    """Validate a SELECT action and safely fill one public missing identifier.

    Hydration is permitted only for the exact, otherwise unambiguous payload
    ``{stage: SELECT, selection: ACQUIRE}`` missing *only* ``evidence_id``.
    The Pydantic schema remains the authority after hydration, so unknown
    fields, malformed values, non-SELECT stages, and ambiguous prompts fail.
    """

    try:
        payload = extract_json_object(raw)
    except Exception as exception:
        message = str(exception)
        return CanonicalizedSelectAction(
            action=None,
            raw_json_valid=False,
            raw_schema_valid=False,
            evidence_id_hydrated=False,
            raw_json_error=message,
            raw_schema_error=None,
            executable_action_error=message,
        )

    try:
        action = SequentialControllerAction.model_validate(payload)
        if action.stage != ControllerStage.SELECT:
            raise ValueError("generated action is not SELECT")
        return CanonicalizedSelectAction(
            action=action,
            raw_json_valid=True,
            raw_schema_valid=True,
            evidence_id_hydrated=False,
            raw_json_error=None,
            raw_schema_error=None,
            executable_action_error=None,
        )
    except Exception as exception:
        raw_schema_error = str(exception)

    can_hydrate = (
        payload.get("stage") == ControllerStage.SELECT.value
        and payload.get("selection") == SelectionDecision.ACQUIRE.value
        and "evidence_id" not in payload
    )
    evidence_id = _visible_select_evidence_id(row) if can_hydrate else None
    if evidence_id is not None:
        hydrated_payload = {**payload, "evidence_id": evidence_id}
        try:
            action = SequentialControllerAction.model_validate(hydrated_payload)
            if action.stage != ControllerStage.SELECT:
                raise ValueError("generated action is not SELECT")
            return CanonicalizedSelectAction(
                action=action,
                raw_json_valid=True,
                raw_schema_valid=False,
                evidence_id_hydrated=True,
                raw_json_error=None,
                raw_schema_error=raw_schema_error,
                executable_action_error=None,
            )
        except Exception as exception:
            executable_error = str(exception)
    else:
        executable_error = raw_schema_error
    return CanonicalizedSelectAction(
        action=None,
        raw_json_valid=True,
        raw_schema_valid=False,
        evidence_id_hydrated=False,
        raw_json_error=None,
        raw_schema_error=raw_schema_error,
        executable_action_error=executable_error,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("val_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--expected-records", type=int)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    import torch
    from peft import PeftModel
    from tqdm.auto import tqdm
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from activemap.agent.vlm_sft import load_vlm_sft_rows, materialize_vlm_messages

    rows = load_vlm_sft_rows(args.val_jsonl)
    if args.expected_records is not None and len(rows) != args.expected_records:
        raise ValueError(f"expected {args.expected_records} selector rows, found {len(rows)}")
    if {row["stage"] for row in rows} != {ControllerStage.SELECT.value}:
        raise ValueError("selector evaluation contains non-SELECT records")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    base = AutoModelForImageTextToText.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, args.adapter).to(args.device).eval()
    traces = []
    for row in tqdm(rows, desc="Sequential selector validation"):
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
        canonical = canonicalize_select_action(raw, row)
        predicted_action = canonical.action
        predicted = (
            predicted_action.selection
            if predicted_action is not None
            else SelectionDecision.STOP
        )
        target_action = _assistant_action(row)
        traces.append(
            {
                "trajectory_id": row["trajectory_id"],
                "task_id": row["task_id"],
                "split": row["split"],
                "target_selection": target_action.selection.value,
                "predicted_selection": predicted.value,
                "policy_relative_advantage": float(row["policy_relative_advantage"]),
                # Retained for old aggregators; executable validity is the
                # explicit v2 protocol field used for promotion.
                "valid_action": canonical.action is not None,
                "raw_json_valid": canonical.raw_json_valid,
                "raw_schema_valid": canonical.raw_schema_valid,
                "executable_action_valid": canonical.action is not None,
                "evidence_id_hydrated": canonical.evidence_id_hydrated,
                "resolved_evidence_id": (
                    predicted_action.evidence_id if predicted_action is not None else None
                ),
                "raw_output": raw,
                "parse_error": canonical.executable_action_error,
                "raw_json_error": canonical.raw_json_error,
                "raw_schema_error": canonical.raw_schema_error,
            }
        )
    metrics = selector_metrics(traces)
    bootstrap = task_bootstrap(
        traces, repetitions=args.bootstrap_repetitions, seed=args.seed
    )
    raw_json_valid_rate = sum(row["raw_json_valid"] for row in traces) / len(traces)
    raw_schema_valid_rate = sum(row["raw_schema_valid"] for row in traces) / len(traces)
    executable_action_valid_rate = sum(
        row["executable_action_valid"] for row in traces
    ) / len(traces)
    hydrated_action_rate = sum(row["evidence_id_hydrated"] for row in traces) / len(traces)
    always_stop_macro_f1 = (2.0 * (1.0 - metrics["target_call_rate"])) / (
        2.0 - metrics["target_call_rate"]
    ) / 2.0
    promotion = {
        "executable_action_valid_rate_1": executable_action_valid_rate == 1.0,
        "nonzero_calls": metrics["predicted_calls"] > 0,
        "call_rate_at_most_0_50": metrics["predicted_call_rate"] <= 0.50,
        "acquire_recall_at_least_0_10": metrics["acquire_recall"] >= 0.10,
        "precision_at_least_prevalence": metrics["acquire_precision"]
        >= metrics["target_call_rate"],
        "utility_positive": metrics["realized_utility_sum"] > 0.0,
        "macro_f1_above_always_stop": metrics["macro_f1"] > always_stop_macro_f1,
        "utility_bootstrap_ci_above_zero": bootstrap["intervals"][
            "realized_utility_mean"
        ]["ci95_low"]
        > 0.0,
    }
    args.output_dir.mkdir(parents=True)
    traces_path = args.output_dir / "traces.jsonl"
    with traces_path.open("w", encoding="utf-8") as handle:
        for row in traces:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {
        "schema_version": "sequential-selector-evaluation-v2",
        "evaluation_role": "component-teacher-forced-current-policy-direct-draft",
        "select_action_canonicalization": {
            "enabled": True,
            "rule": (
                "hydrate only a missing evidence_id for otherwise SELECT/ACQUIRE "
                "JSON using the unique evidence_id exposed in the user state"
            ),
            "uses_assistant_target": False,
            "uses_validation_advantage": False,
        },
        "sample_count": len(traces),
        "task_count": bootstrap["task_count"],
        "raw_json_valid_rate": raw_json_valid_rate,
        "raw_schema_valid_rate": raw_schema_valid_rate,
        "executable_action_valid_rate": executable_action_valid_rate,
        "hydrated_action_rate": hydrated_action_rate,
        "valid_action_rate": executable_action_valid_rate,
        "metrics": metrics,
        "always_stop_macro_f1": always_stop_macro_f1,
        "task_bootstrap": bootstrap,
        "promotion_gate": {**promotion, "passed": all(promotion.values())},
        "sources": {
            "model": str(args.model.resolve()),
            "adapter": str(args.adapter.resolve()),
            "validation": {
                "path": str(args.val_jsonl.resolve()),
                "sha256": _sha256(args.val_jsonl),
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
