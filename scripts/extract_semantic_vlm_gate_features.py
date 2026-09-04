#!/usr/bin/env python3
"""Extract frozen PRE_TOOL visual states from one trained semantic VLM adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("adapter")
    parser.add_argument("sft_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--stage",
        choices=("PRE_TOOL", "SELECT"),
        default="PRE_TOOL",
        help="Controller stage used to form the binary utility-gate examples.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--pooling", choices=("last", "last_mean"), default="last_mean")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    import numpy as np
    import torch
    from peft import PeftModel
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from activemap.agent.visual_gate import pool_prompt_state, unique_pre_tool_rows
    from activemap.agent.vlm_sft import (
        VisualActionSFTCollator,
        VisualPromptDataset,
        load_vlm_sft_rows,
    )

    source_rows = load_vlm_sft_rows(args.sft_jsonl)
    if args.stage == "PRE_TOOL":
        rows = unique_pre_tool_rows(source_rows)
        label_source = "oracle_use_tool"
        utility_source = "consensus_mean_utility_gain"
    else:
        rows = [row for row in source_rows if row.get("stage") == "SELECT"]
        if not rows:
            raise ValueError("dataset contains no SELECT rows")
        # Sequential SELECT records predate the generic visual-gate schema and
        # use trajectory_id as their unique state ID. Normalize that public ID
        # at the adapter boundary without changing labels or utilities.
        rows = [
            {
                **row,
                "example_id": str(row.get("example_id") or row.get("trajectory_id") or ""),
            }
            for row in rows
        ]
        required = {"example_id", "task_id", "split", "selected_tool", "policy_relative_advantage"}
        if any(required - row.keys() for row in rows):
            raise ValueError("SELECT row lacks binary utility-gate metadata")
        label_source = "selected_tool"
        utility_source = "policy_relative_advantage"
    if args.limit is not None:
        rows = rows[: args.limit]
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = VisualPromptDataset(rows, processor, max_length=args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=VisualActionSFTCollator(tokenizer.pad_token_id),
        num_workers=0,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, args.adapter).to(args.device).eval()

    blocks = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Visual gate features"):
            batch = {key: value.to(args.device) for key, value in batch.items()}
            output = model(
                **batch,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            hidden_states = getattr(output, "hidden_states", None)
            if not hidden_states:
                raise RuntimeError("model did not return language hidden states")
            pooled = pool_prompt_state(
                hidden_states[-1], batch["attention_mask"], mode=args.pooling
            )
            blocks.append(pooled.float().cpu().numpy())
    features = np.concatenate(blocks, axis=0)
    if len(features) != len(rows):
        raise RuntimeError("feature count does not match PRE_TOOL rows")

    args.output_dir.mkdir(parents=True)
    np.save(args.output_dir / "features.npy", features.astype(np.float16))
    metadata: list[dict[str, Any]] = []
    for row in rows:
        if args.stage == "PRE_TOOL":
            consensus = row.get("consensus")
            if not isinstance(consensus, dict) or "mean_utility_gain" not in consensus:
                raise ValueError("PRE_TOOL row lacks consensus utility metadata")
            label = bool(row["oracle_use_tool"])
            utility = float(consensus["mean_utility_gain"])
            beneficial_votes = int(consensus["beneficial_votes"])
        else:
            label = bool(row["selected_tool"])
            utility = float(row["policy_relative_advantage"])
            beneficial_votes = int(label)
        metadata.append(
            {
                "example_id": str(row["example_id"]),
                "task_id": str(row["task_id"]),
                "split": str(row["split"]),
                "controller_stage": args.stage,
                "oracle_use_tool": label,
                "consensus_mean_utility_gain": utility,
                "consensus_beneficial_votes": beneficial_votes,
                "utility_source": utility_source,
            }
        )
    with (args.output_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for row in metadata:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {
        "schema_version": "semantic-vlm-visual-gate-features-v1",
        "model": str(Path(args.model).resolve()),
        "adapter": str(Path(args.adapter).resolve()),
        "source": {"path": str(args.sft_jsonl.resolve()), "sha256": _sha256(args.sft_jsonl)},
        "controller_stage": args.stage,
        "feature_protocol": f"frozen-adapter-{args.stage}-prompt-state-with-recorded-pooling",
        "label_source": label_source,
        "sample_count": len(rows),
        "task_count": len({str(row["task_id"]) for row in rows}),
        "positive_count": sum(bool(row["oracle_use_tool"]) for row in metadata),
        "positive_rate": sum(bool(row["oracle_use_tool"]) for row in metadata) / len(metadata),
        "feature_dim": int(features.shape[1]),
        "feature_dtype": "float16",
        "pooling": args.pooling,
        "utility_metadata": utility_source,
        "assistant_tokens_seen": False,
        "test_assets_read": False,
        "limit": args.limit,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
