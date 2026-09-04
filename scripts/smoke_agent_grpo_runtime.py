#!/usr/bin/env python3
"""Run one CPU GRPO step with a tiny random model to verify TRL integration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from activemap.agent.rl import (
    executable_schema_reward,
    sparse_acquisition_reward,
    task_utility_reward,
    terminal_safety_reward,
)


def _read_rows(path: Path, count: int = 2) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("split") == "test":
                    raise ValueError("runtime smoke must not read test rows")
                rows.append(row)
                if len(rows) == count:
                    break
    if len(rows) < count:
        raise ValueError(f"runtime smoke needs at least {count} rows")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rl_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from trl import GRPOConfig, GRPOTrainer

    rows = _read_rows(args.rl_jsonl)
    backend = Tokenizer(
        WordLevel(
            {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3},
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }}: "
        "{{ message['content'] }}\\n{% endfor %}assistant: "
    )
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=2048,
            n_ctx=2048,
            n_embd=32,
            n_layer=1,
            n_head=2,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    )
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=["c_attn"],
        ),
    )
    config = GRPOConfig(
        output_dir=str(args.output_dir),
        max_steps=1,
        use_cpu=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        num_generations=2,
        generation_batch_size=2,
        max_completion_length=8,
        beta=0.02,
        learning_rate=1e-6,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        gradient_checkpointing=False,
        reward_weights=[1.0, 1.0, 1.0, 0.25],
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            task_utility_reward,
            executable_schema_reward,
            terminal_safety_reward,
            sparse_acquisition_reward,
        ],
        args=config,
        train_dataset=Dataset.from_list(rows),
        processing_class=tokenizer,
    )
    result = trainer.train()
    summary = {
        "runtime_smoke_passed": True,
        "steps": int(result.global_step),
        "rows": len(rows),
        "device": "cpu",
        "test_assets_read": False,
        "metrics": {
            key: float(value)
            for key, value in result.metrics.items()
            if isinstance(value, int | float)
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "runtime_smoke.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
