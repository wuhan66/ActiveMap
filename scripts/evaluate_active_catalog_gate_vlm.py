#!/usr/bin/env python3
"""Evaluate a visual ACQUIRE/STOP gate composed with a frozen candidate ranker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from activemap.agent.active_catalog_candidate_ranker import CandidateUtilityRankerPredictor
from activemap.agent.vlm_sft import load_vlm_sft_rows

try:
    from scripts.evaluate_active_catalog_selector import (
        _extract_json_object,
        active_catalog_metrics,
        grouped_bootstrap,
    )
    from scripts.train_active_catalog_candidate_ranker import load_examples
except ModuleNotFoundError:
    from evaluate_active_catalog_selector import (
        _extract_json_object,
        active_catalog_metrics,
        grouped_bootstrap,
    )
    from train_active_catalog_candidate_ranker import load_examples


def parse_gate_output(raw: str) -> str:
    selection = str(_extract_json_object(raw).get("selection"))
    if selection not in {"ACQUIRE", "STOP"}:
        raise ValueError("gate output is not ACQUIRE or STOP")
    return selection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("ranker_checkpoint", type=Path)
    parser.add_argument("val_jsonl", type=Path)
    parser.add_argument("val_evaluation_index", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--max-new-tokens", type=int, default=16)
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
    examples = load_examples(args.val_jsonl, args.val_evaluation_index, "val")
    examples_by_id = {str(row["example_id"]): row for row in examples}
    if {str(row["example_id"]) for row in rows} != set(examples_by_id):
        raise ValueError("gate validation and evaluation index disagree")
    ranker = CandidateUtilityRankerPredictor(str(args.ranker_checkpoint), "cpu")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    base = AutoModelForImageTextToText.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, args.adapter).to(args.device).eval()
    traces = []
    for row in tqdm(rows, desc="Active-Catalog visual gate validation"):
        example = examples_by_id[str(row["example_id"])]
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
            selection = parse_gate_output(raw)
        except Exception as exception:
            selection = "STOP"
            error = str(exception)
        scores = ranker.score_state(example["state"])
        evidence_id = (
            max(scores, key=lambda value: (scores[value], value))
            if selection == "ACQUIRE"
            else None
        )
        utility_by_id = dict(
            zip(example["candidate_ids"], map(float, example["utilities"]), strict=True)
        )
        cost_by_id = dict(
            zip(example["candidate_ids"], map(float, example["costs"]), strict=True)
        )
        best_id = max(
            example["candidate_ids"],
            key=lambda value: (utility_by_id[value], value),
        )
        stop = float(example["stop_utility"])
        target_call = utility_by_id[best_id] > stop
        policy_utility = utility_by_id[str(evidence_id)] if evidence_id else stop
        traces.append(
            {
                "example_id": str(row["example_id"]),
                "aoi_id": example["aoi_id"],
                "source_episode": example["source_episode"],
                "candidate_count": len(example["candidate_ids"]),
                "target_selection": "ACQUIRE" if target_call else "STOP",
                "target_evidence_id": best_id if target_call else None,
                "predicted_selection": selection,
                "predicted_evidence_id": evidence_id,
                "valid_action": error is None,
                "stop_utility": stop,
                "oracle_utility": max(stop, utility_by_id[best_id]),
                "policy_utility": policy_utility,
                "policy_cost": cost_by_id[str(evidence_id)] if evidence_id else 0.0,
                "regret": max(stop, utility_by_id[best_id]) - policy_utility,
                "raw_output": raw,
                "parse_error": error,
            }
        )
    metrics = active_catalog_metrics(traces)
    bootstrap = grouped_bootstrap(
        traces,
        group_key="aoi_id",
        repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    valid_rate = sum(row["valid_action"] for row in traces) / len(traces)
    checks = {
        "valid_action_rate_1": valid_rate == 1.0,
        "nonzero_calls": metrics["predicted_call_rate"] > 0.0,
        "macro_f1_above_always_stop": metrics[
            "selection_macro_f1_delta_vs_always_stop"
        ]
        > 0.0,
        "utility_positive": metrics["realized_utility_mean"] > 0.0,
        "utility_aoi_bootstrap_ci_above_zero": bootstrap["intervals"][
            "realized_utility_mean"
        ]["ci95_low"]
        > 0.0,
        "false_call_rate_at_most_0_10": metrics["false_call_rate"] <= 0.10,
        "exact_recall_above_random": metrics["exact_evidence_recall"]
        > metrics["random_exact_evidence_recall"],
    }
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "traces.jsonl").open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace, separators=(",", ":")) + "\n")
    summary = {
        "schema_version": "active-catalog-visual-gate-ranker-evaluation-v1",
        "valid_action_rate": valid_rate,
        "metrics": metrics,
        "aoi_bootstrap": bootstrap,
        "promotion_gate": {**checks, "passed": all(checks.values())},
        "test_assets_read": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
