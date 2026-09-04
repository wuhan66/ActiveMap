#!/usr/bin/env python3
"""Extract frozen visual-state features for the structured catalog action head."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_identity(value: str, *, adapter: bool) -> dict[str, Any]:
    path = Path(value)
    identity: dict[str, Any] = {
        "identifier": value,
        "local_path": path.is_dir(),
    }
    if not path.is_dir():
        return identity
    identity["path"] = str(path.resolve())
    config = path / "adapter_config.json" if adapter else path / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"missing model provenance config: {config}")
    identity["config_sha256"] = _sha256(config)
    if adapter:
        weights = [
            candidate
            for candidate in (
                path / "adapter_model.safetensors",
                path / "adapter_model.bin",
            )
            if candidate.is_file()
        ]
        if len(weights) != 1:
            raise FileNotFoundError(
                f"expected exactly one adapter weight file in {path}"
            )
        identity["weights_sha256"] = _sha256(weights[0])
    return identity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("adapter")
    parser.add_argument("rl_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--pooling", choices=("last", "last_mean"), default="last_mean")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def utility_advantage(row: dict[str, Any]) -> float:
    stop = next(float(action["utility"]) for action in row["actions"] if action["key"] == "STOP")
    acquire = [
        float(action["utility"])
        for action in row["actions"]
        if str(action["key"]).startswith("ACQUIRE:")
    ]
    if not acquire:
        raise ValueError("catalog state has no ACQUIRE action")
    return max(acquire) - stop


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid feature shard")

    import numpy as np
    import torch
    from peft import PeftModel
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from activemap.agent.visual_gate import pool_prompt_state
    from activemap.agent.vlm_rl import load_rl_rows
    from activemap.agent.vlm_sft import (
        VisualActionSFTCollator,
        VisualPromptDataset,
    )

    rows = load_rl_rows(args.rl_jsonl, args.limit)
    rows = rows[args.shard_index :: args.num_shards]
    if not rows:
        raise ValueError("feature shard is empty")
    prompt_rows = [{"messages": row["prompt"]} for row in rows]
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    dataset = VisualPromptDataset(
        prompt_rows, processor, max_length=args.max_length
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=VisualActionSFTCollator(processor.tokenizer.pad_token_id),
        num_workers=0,
    )
    base = AutoModelForImageTextToText.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, args.adapter).to(args.device).eval()

    blocks = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Active-Catalog VLA features ({args.pooling})"):
            batch = {key: value.to(args.device) for key, value in batch.items()}
            output = model(
                **batch,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            pooled = pool_prompt_state(
                output.hidden_states[-1],
                batch["attention_mask"],
                mode=args.pooling,
            )
            blocks.append(pooled.float().cpu().numpy())
    features = np.concatenate(blocks, axis=0).astype(np.float16)
    if len(features) != len(rows):
        raise RuntimeError("feature and state counts differ")

    records = []
    for index, row in enumerate(rows):
        advantage = utility_advantage(row)
        records.append(
            {
                "example_id": str(
                    row.get("state_id")
                    or row.get("example_id")
                    or f"{row['task_id']}:{index}"
                ),
                "task_id": str(row["task_id"]),
                "split": str(row["split"]),
                "oracle_use_tool": advantage > 0.0,
                "consensus_mean_utility_gain": advantage,
            }
        )
    args.output_dir.mkdir(parents=True)
    np.save(args.output_dir / "features.npy", features)
    with (args.output_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {
        "schema_version": "active-catalog-vla-frozen-features-v1",
        "sample_count": len(records),
        "task_count": len({row["task_id"] for row in records}),
        "positive_count": sum(row["oracle_use_tool"] for row in records),
        "positive_rate": sum(row["oracle_use_tool"] for row in records) / len(records),
        "feature_dim": int(features.shape[1]),
        "feature_dtype": "float16",
        "pooling": args.pooling,
        "utility_metadata": "best_acquire_minus_stop_utility",
        "assistant_tokens_seen": False,
        "shard": {"index": args.shard_index, "count": args.num_shards},
        "source": {
            "path": str(args.rl_jsonl.resolve()),
            "sha256": _sha256(args.rl_jsonl),
        },
        "model": model_identity(args.model, adapter=False),
        "adapter": model_identity(args.adapter, adapter=True),
        "test_assets_read": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
