"""LoRA SFT for the structured ActiveMap maintenance controller."""

from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("train_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--eval-jsonl", type=Path)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-threshold", type=float, default=0.0)
    parser.add_argument(
        "--select-last-checkpoint",
        action="store_true",
        help=(
            "keep the final optimizer state instead of selecting by eval loss; "
            "use when validation intentionally preserves a different action prior"
        ),
    )
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _read_jsonl(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    if not rows:
        raise ValueError(f"no records in {path}")
    return rows


def main() -> None:
    args = parse_args()
    if args.early_stopping_patience < 0:
        raise ValueError("early-stopping-patience must be non-negative")
    if args.early_stopping_patience and args.eval_jsonl is None:
        raise ValueError("early stopping requires --eval-jsonl")
    if args.early_stopping_patience and args.select_last_checkpoint:
        raise ValueError("early stopping is incompatible with --select-last-checkpoint")
    import torch
    from transformers import AutoTokenizer

    from activemap.agent.sft_data import ActionSFTCollator, ActionSFTDataset

    args.output_dir.mkdir(parents=True, exist_ok=True)
    control_dir = args.output_dir / "control"
    control_dir.mkdir(exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_rows = _read_jsonl(args.train_jsonl, args.max_train_samples)
    eval_rows = (
        _read_jsonl(args.eval_jsonl, args.max_eval_samples)
        if args.eval_jsonl is not None
        else None
    )
    train_dataset = ActionSFTDataset(train_rows, tokenizer, max_length=args.max_length)
    eval_dataset = (
        ActionSFTDataset(eval_rows, tokenizer, max_length=args.max_length)
        if eval_rows is not None
        else None
    )
    data_summary = {
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset) if eval_dataset is not None else 0,
        "max_length": args.max_length,
        "train_truncated": train_dataset.truncated_count,
        "eval_truncated": eval_dataset.truncated_count if eval_dataset is not None else 0,
        "train_max_observed_length": train_dataset.max_observed_length,
        "eval_max_observed_length": (
            eval_dataset.max_observed_length if eval_dataset is not None else 0
        ),
        "assistant_only_loss": True,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_threshold": args.early_stopping_threshold,
        "select_last_checkpoint": args.select_last_checkpoint,
    }
    (args.output_dir / "data_summary.json").write_text(
        json.dumps(data_summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(data_summary, indent=2), flush=True)
    if args.dry_run:
        return

    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        EarlyStoppingCallback,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )

    set_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lora = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        use_rslora=True,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    eval_strategy = "steps" if eval_dataset is not None else "no"
    kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir / "checkpoints"),
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "save_strategy": "steps",
        "save_total_limit": args.save_total_limit,
        "bf16": True,
        "gradient_checkpointing": True,
        "seed": args.seed,
        "report_to": ["tensorboard"],
        "logging_dir": str(args.output_dir / "tensorboard"),
        "remove_unused_columns": False,
        "load_best_model_at_end": eval_dataset is not None and not args.select_last_checkpoint,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
    }
    strategy_name = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(TrainingArguments).parameters
        else "evaluation_strategy"
    )
    kwargs[strategy_name] = eval_strategy
    training_args = TrainingArguments(**kwargs)

    class ControlCallback(TrainerCallback):
        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            while (control_dir / "PAUSE").exists() and not (control_dir / "STOP").exists():
                time.sleep(10)
            if (control_dir / "STOP").exists():
                control.should_training_stop = True
            return control

    callbacks: list[TrainerCallback] = [ControlCallback()]
    if args.early_stopping_patience:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=ActionSFTCollator(tokenizer.pad_token_id),
        callbacks=callbacks,
    )
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir / "final"))
    tokenizer.save_pretrained(args.output_dir / "final")
    trainer.save_state()
    (args.output_dir / "train_metrics.json").write_text(
        json.dumps(train_result.metrics, indent=2) + "\n", encoding="utf-8"
    )
    if eval_dataset is not None:
        metrics = trainer.evaluate()
        (args.output_dir / "eval_metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
