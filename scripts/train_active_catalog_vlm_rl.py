#!/usr/bin/env python3
"""KL-constrained multimodal contextual policy optimization for Active-Catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("sft_adapter", type=Path)
    parser.add_argument("train_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--action-limit", type=int, default=4)
    parser.add_argument("--kl-beta", type=float, default=0.05)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--utility-scale", type=float, default=1.0)
    parser.add_argument("--acquire-fraction", type=float, default=0.0)
    parser.add_argument("--pairwise-weight", type=float, default=0.0)
    parser.add_argument("--pairwise-margin", type=float, default=0.0)
    parser.add_argument("--length-normalize-action-score", action="store_true")
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if (
        args.epochs <= 0
        or args.learning_rate <= 0
        or args.action_limit < 2
        or args.utility_scale <= 0
        or not 0 <= args.acquire_fraction <= 1
        or args.pairwise_weight < 0
        or args.pairwise_margin < 0
    ):
        raise ValueError("invalid contextual RL optimization settings")

    from activemap.agent.vlm_rl import (
        audit_rl_splits,
        load_rl_rows,
        sample_rl_training_rows,
    )

    train_pool = load_rl_rows(args.train_jsonl)
    train_rows = sample_rl_training_rows(
        train_pool,
        limit=args.max_train_samples,
        acquire_fraction=args.acquire_fraction,
        seed=args.seed,
    )
    eval_rows = load_rl_rows(args.eval_jsonl, args.max_eval_samples)
    split_audit = audit_rl_splits(train_rows, eval_rows)
    summary = {
        "schema_version": "active-catalog-vlm-contextual-rl-train-v1",
        "method": "exact_action_set_kl_constrained_policy_optimization",
        "claim_boundary": "offline multimodal contextual RL; not online recurrent RL",
        "sft_adapter": str(args.sft_adapter.resolve()),
        "split_audit": split_audit,
        "optimization": {
            "seed": args.seed,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "action_limit": args.action_limit,
            "kl_beta": args.kl_beta,
            "entropy_weight": args.entropy_weight,
            "temperature": args.temperature,
            "utility_scale": args.utility_scale,
            "acquire_fraction": args.acquire_fraction,
            "pairwise_weight": args.pairwise_weight,
            "pairwise_margin": args.pairwise_margin,
            "length_normalize_action_score": args.length_normalize_action_score,
        },
        "train_target_counts": {
            "acquire": sum(
                str(row["target_action_key"]).startswith("ACQUIRE:") for row in train_rows
            ),
            "stop": sum(row["target_action_key"] == "STOP" for row in train_rows),
        },
        "reward": "frozen quality-cost utility over executable STOP/ACQUIRE actions",
        "test_assets_read": False,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "data_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return

    import torch
    from peft import PeftConfig, PeftModel
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )

    from activemap.agent.vlm_rl import (
        VisualRLStateCollator,
        VisualRLStateDataset,
        contextual_policy_loss,
        sequence_log_probabilities,
    )
    from scripts.train_active_catalog_vlm_dpo import adapter_audit

    set_seed(args.seed)
    peft_config = PeftConfig.from_pretrained(args.sft_adapter)
    base_source = str(peft_config.base_model_name_or_path)
    processor_source = (
        args.sft_adapter
        if any((args.sft_adapter / name).is_file() for name in (
            "processor_config.json", "preprocessor_config.json", "tokenizer_config.json"
        ))
        else base_source
    )
    processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    train_dataset = VisualRLStateDataset(
        train_rows, processor, max_length=args.max_length, action_limit=args.action_limit
    )
    eval_dataset = VisualRLStateDataset(
        eval_rows, processor, max_length=args.max_length, action_limit=args.action_limit
    )
    collator = VisualRLStateCollator(processor.tokenizer.pad_token_id)
    smoke = collator([train_dataset[0]])
    summary["first_batch_shapes"] = {key: list(value.shape) for key, value in smoke.items()}

    base = AutoModelForImageTextToText.from_pretrained(
        base_source, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    base.config.use_cache = False
    model = PeftModel.from_pretrained(
        base, args.sft_adapter, adapter_name="default", is_trainable=True
    )
    model.load_adapter(args.sft_adapter, adapter_name="ref", is_trainable=False)
    model.set_adapter("default")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    summary["reference_policy"] = adapter_audit(model)
    (args.output_dir / "data_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    def set_adapter(active_model: Any, name: str) -> None:
        core = active_model.module if hasattr(active_model, "module") else active_model
        core.set_adapter(name)
        for parameter_name, parameter in core.named_parameters():
            if ".ref." in parameter_name:
                parameter.requires_grad_(False)

    class PolicyTrainer(Trainer):
        def compute_loss(
            self, model: Any, inputs: dict[str, torch.Tensor], return_outputs: bool = False,
            num_items_in_batch: torch.Tensor | None = None,
        ) -> Any:
            utilities = inputs.pop("rl__utilities")
            stop_index = int(inputs.pop("rl__stop_index"))
            target_index = int(inputs.pop("rl__target_index"))
            target_is_acquire = bool(inputs.pop("rl__target_is_acquire"))
            labels = inputs.pop("labels")
            set_adapter(model, "default")
            policy = sequence_log_probabilities(
                model(**inputs).logits,
                labels,
                normalize_by_length=args.length_normalize_action_score,
            )
            set_adapter(model, "ref")
            with torch.no_grad():
                reference = sequence_log_probabilities(
                    model(**inputs).logits,
                    labels,
                    normalize_by_length=args.length_normalize_action_score,
                )
            set_adapter(model, "default")
            loss, diagnostics = contextual_policy_loss(
                policy, reference, utilities * args.utility_scale,
                kl_beta=args.kl_beta, entropy_weight=args.entropy_weight,
                temperature=args.temperature,
                pairwise_weight=args.pairwise_weight if target_is_acquire else 0.0,
                pairwise_margin=args.pairwise_margin,
                stop_index=stop_index,
                target_index=target_index,
            )
            return (loss, diagnostics) if return_outputs else loss

    history = args.output_dir / "history.jsonl"

    class HistoryCallback(TrainerCallback):
        def on_log(self, args: Any, state: Any, control: Any, logs=None, **kwargs: Any) -> Any:
            if logs:
                with history.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"step": state.global_step, "epoch": state.epoch, **logs}) + "\n")
            return control

    training = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=["tensorboard"],
        logging_dir=str(args.output_dir / "tensorboard"),
        seed=args.seed,
    )
    trainer = PolicyTrainer(
        model=model, args=training, train_dataset=train_dataset, eval_dataset=eval_dataset,
        data_collator=collator, callbacks=[HistoryCallback()]
    )
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    model.set_adapter("default")
    final = args.output_dir / "final"
    model.save_pretrained(final, selected_adapters=["default"])
    processor.save_pretrained(final)
    trainer.save_state()
    (args.output_dir / "train_metrics.json").write_text(
        json.dumps(result.metrics, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "eval_metrics.json").write_text(
        json.dumps(trainer.evaluate(), indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
