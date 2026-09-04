#!/usr/bin/env python3
"""Audit catalog-action likelihoods without decoding or reading test assets."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def binary_auc(labels: list[bool], scores: list[float]) -> float:
    positives = [score for label, score in zip(labels, scores, strict=True) if label]
    negatives = [score for label, score in zip(labels, scores, strict=True) if not label]
    if not positives or not negatives:
        raise ValueError("AUC requires both target classes")
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("rl_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--action-limit", type=int, default=4)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    import torch
    from peft import PeftModel
    from tqdm.auto import tqdm
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from activemap.agent.vlm_rl import (
        VisualRLStateCollator,
        VisualRLStateDataset,
        load_rl_rows,
        select_hard_actions,
        sequence_log_probabilities,
    )

    rows = load_rl_rows(args.rl_jsonl, args.max_samples)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    dataset = VisualRLStateDataset(
        rows, processor, max_length=args.max_length, action_limit=args.action_limit
    )
    collator = VisualRLStateCollator(processor.tokenizer.pad_token_id)
    base = AutoModelForImageTextToText.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, args.adapter).to(args.device).eval()

    traces = []
    for index, row in enumerate(tqdm(rows, desc="Catalog action-score audit")):
        actions = select_hard_actions(row, args.action_limit)
        batch = collator([dataset[index]])
        labels = batch.pop("labels").to(args.device)
        for key in tuple(batch):
            if key.startswith("rl__"):
                batch.pop(key)
            else:
                batch[key] = batch[key].to(args.device)
        with torch.inference_mode():
            logits = model(**batch).logits
            summed = sequence_log_probabilities(logits, labels)
            means = sequence_log_probabilities(
                logits, labels, normalize_by_length=True
            )
        keys = [str(action["key"]) for action in actions]
        stop_index = keys.index("STOP")
        acquire_indices = [
            action_index
            for action_index, key in enumerate(keys)
            if key.startswith("ACQUIRE:")
        ]
        best_acquire = max(acquire_indices, key=lambda item: float(means[item]))
        target_index = keys.index(str(row["target_action_key"]))
        sum_prediction = int(summed.argmax())
        mean_prediction = int(means.argmax())
        traces.append(
            {
                "task_id": str(row["task_id"]),
                "target_action": keys[target_index],
                "target_is_acquire": keys[target_index].startswith("ACQUIRE:"),
                "sum_prediction": keys[sum_prediction],
                "mean_prediction": keys[mean_prediction],
                "sum_target_rank": int(
                    1 + (summed > summed[target_index]).sum().item()
                ),
                "mean_target_rank": int(
                    1 + (means > means[target_index]).sum().item()
                ),
                "mean_acquire_stop_margin": float(
                    means[best_acquire] - means[stop_index]
                ),
                "sum_acquire_stop_margin": float(
                    summed[best_acquire] - summed[stop_index]
                ),
            }
        )

    labels = [bool(row["target_is_acquire"]) for row in traces]
    mean_margins = [float(row["mean_acquire_stop_margin"]) for row in traces]
    sum_margins = [float(row["sum_acquire_stop_margin"]) for row in traces]
    summary = {
        "schema_version": "active-catalog-vlm-action-score-audit-v1",
        "sample_count": len(traces),
        "target_acquire_count": sum(labels),
        "metrics": {
            "sum_exact_action_accuracy": statistics.fmean(
                row["sum_prediction"] == row["target_action"] for row in traces
            ),
            "mean_exact_action_accuracy": statistics.fmean(
                row["mean_prediction"] == row["target_action"] for row in traces
            ),
            "sum_acquire_rate": statistics.fmean(
                row["sum_prediction"].startswith("ACQUIRE:") for row in traces
            ),
            "mean_acquire_rate": statistics.fmean(
                row["mean_prediction"].startswith("ACQUIRE:") for row in traces
            ),
            "sum_target_mean_rank": statistics.fmean(
                int(row["sum_target_rank"]) for row in traces
            ),
            "mean_target_mean_rank": statistics.fmean(
                int(row["mean_target_rank"]) for row in traces
            ),
            "sum_margin_auc": binary_auc(labels, sum_margins),
            "mean_margin_auc": binary_auc(labels, mean_margins),
            "positive_mean_margin": statistics.fmean(
                margin for label, margin in zip(labels, mean_margins, strict=True) if label
            ),
            "negative_mean_margin": statistics.fmean(
                margin for label, margin in zip(labels, mean_margins, strict=True) if not label
            ),
        },
        "sources": {
            "model": str(args.model.resolve()),
            "adapter": str(args.adapter.resolve()),
            "rl_states": str(args.rl_jsonl.resolve()),
        },
        "test_assets_read": False,
    }
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "traces.jsonl").open("w", encoding="utf-8") as handle:
        for row in traces:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
