#!/usr/bin/env python3
"""Memory-efficient multimodal DPO for the Qwen3-VL active-catalog selector."""

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
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument(
        "--family-balance",
        choices=("none", "inverse_frequency"),
        default="none",
    )
    parser.add_argument("--family-balance-power", type=float, default=1.0)
    parser.add_argument("--safe-acquire-boost", type=float, default=0.0)
    parser.add_argument("--unsafe-stop-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def adapter_audit(model: Any) -> dict[str, Any]:
    configs = getattr(model, "peft_config", {})
    if set(configs) < {"default", "ref"}:
        raise RuntimeError("multimodal DPO requires default and ref adapters")
    parameters = dict(model.named_parameters())
    default_names = [name for name in parameters if ".default." in name]
    if not default_names:
        raise RuntimeError("DPO model has no default LoRA parameters")
    mismatched = []
    trainable_ref = []
    for name in default_names:
        ref_name = name.replace(".default.", ".ref.")
        ref = parameters.get(ref_name)
        if ref is None:
            raise RuntimeError(f"missing reference parameter: {ref_name}")
        trainable_ref += [ref_name] if ref.requires_grad else []
        mismatched += [ref_name] if not bool(parameters[name].detach().equal(ref.detach())) else []
    if trainable_ref or mismatched:
        raise RuntimeError(
            "invalid reference adapter: "
            f"trainable={len(trainable_ref)}, mismatched={len(mismatched)}"
        )
    return {
        "source": "copied_promoted_sft_adapter",
        "parameter_tensors_compared": len(default_names),
        "exact_initial_match": True,
        "trainable_reference_parameters": 0,
    }


def audit_onpolicy_preferences(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    adapter: Path,
) -> dict[str, Any]:
    rows = [*train_rows, *eval_rows]
    flagged = [row for row in rows if row.get("on_policy_executed_recurrent") is True]
    if not flagged:
        return {"on_policy": False, "matched_policy_snapshot": None}
    if len(flagged) != len(rows):
        raise ValueError("cannot mix offline and on-policy preferences")
    snapshots = {str(Path(row["rollout_policy_snapshot"]).resolve()) for row in flagged}
    expected = str(adapter.resolve())
    if snapshots != {expected}:
        raise ValueError(
            f"on-policy snapshot mismatch: expected {expected}, got {sorted(snapshots)}"
        )
    if not all(row.get("test_assets_read") is False for row in flagged):
        raise ValueError("on-policy preferences do not prove test isolation")
    return {
        "on_policy": True,
        "matched_policy_snapshot": expected,
        "executed_recurrent_preferences": len(flagged),
    }


def main() -> None:
    args = parse_args()
    if args.beta <= 0 or args.epochs <= 0 or args.learning_rate <= 0:
        raise ValueError("invalid DPO optimization settings")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

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

    from activemap.agent.vlm_preference import (
        VisualPreferenceCollator,
        VisualPreferenceDataset,
        audit_preference_splits,
        combine_preference_weights,
        combined_preference_batch,
        dpo_loss,
        load_vlm_preference_rows,
        preference_family_weights,
        preference_safety_weights,
        preference_weighted_loss,
        sequence_log_probabilities,
    )
    from scripts.train_semantic_vlm_sft import adapter_initialization_manifest

    set_seed(args.seed)
    initialization = adapter_initialization_manifest(args.sft_adapter)
    peft_config = PeftConfig.from_pretrained(args.sft_adapter)
    base_source = str(peft_config.base_model_name_or_path)
    if not base_source:
        raise ValueError("SFT adapter does not declare a base model")
    processor_assets = (
        "processor_config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
    )
    processor_source = (
        args.sft_adapter
        if any((args.sft_adapter / name).is_file() for name in processor_assets)
        else base_source
    )
    processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    train_rows = load_vlm_preference_rows(args.train_jsonl, args.max_train_samples)
    eval_rows = load_vlm_preference_rows(args.eval_jsonl, args.max_eval_samples)
    split_audit = audit_preference_splits(train_rows, eval_rows)
    onpolicy_audit = audit_onpolicy_preferences(
        train_rows, eval_rows, args.sft_adapter
    )
    family_loss_weights, family_weights = preference_family_weights(
        train_rows, mode=args.family_balance, power=args.family_balance_power
    )
    safety_loss_weights, safety_weight_summary = preference_safety_weights(
        train_rows,
        safe_acquire_boost=args.safe_acquire_boost,
        unsafe_stop_scale=args.unsafe_stop_scale,
    )
    train_loss_weights = combine_preference_weights(
        family_loss_weights, safety_loss_weights
    )
    train_dataset = VisualPreferenceDataset(
        train_rows,
        processor,
        max_length=args.max_length,
        loss_weights=train_loss_weights,
    )
    eval_dataset = VisualPreferenceDataset(eval_rows, processor, max_length=args.max_length)
    collator = VisualPreferenceCollator(processor.tokenizer.pad_token_id)
    train_smoke = collator([train_dataset[0]])
    eval_smoke = collator([eval_dataset[0]])
    summary = {
        "schema_version": "active-catalog-vlm-dpo-train-v1",
        "claim_boundary": (
            "iterative on-policy preference optimization from recurrently executed actions; "
            "not policy-gradient RL"
            if onpolicy_audit["on_policy"]
            else "offline multimodal preference optimization; not online recurrent RL"
        ),
        "sft_adapter": initialization,
        "base_model": base_source,
        "train_samples": len(train_rows),
        "eval_samples": len(eval_rows),
        "split_audit": split_audit,
        "onpolicy_audit": onpolicy_audit,
        "max_length": args.max_length,
        "beta": args.beta,
        "family_balance": args.family_balance,
        "family_balance_power": args.family_balance_power,
        "train_family_weights": family_weights,
        "safe_acquire_boost": args.safe_acquire_boost,
        "unsafe_stop_scale": args.unsafe_stop_scale,
        "train_safety_weights": safety_weight_summary,
        "first_batch_shapes": {
            key: list(value.shape) for key, value in train_smoke.items()
        },
        "first_eval_batch_shapes": {
            key: list(value.shape) for key, value in eval_smoke.items()
        },
        "test_assets_read": False,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "data_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return

    base = AutoModelForImageTextToText.from_pretrained(
        base_source,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    base.config.use_cache = False
    model = PeftModel.from_pretrained(
        base, args.sft_adapter, adapter_name="default", is_trainable=True
    )
    model.load_adapter(args.sft_adapter, adapter_name="ref", is_trainable=False)
    model.set_adapter("default")
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    summary["reference_policy"] = adapter_audit(model)
    (args.output_dir / "data_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    def pair_logps(
        active_model: Any, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = combined_preference_batch(batch)
        labels = inputs.pop("labels")
        outputs = active_model(**inputs)
        logps = sequence_log_probabilities(outputs.logits, labels)
        pair_count = logps.shape[0] // 2
        return logps[:pair_count], logps[pair_count:]

    def set_adapter(active_model: Any, name: str) -> None:
        core = active_model.module if hasattr(active_model, "module") else active_model
        core.set_adapter(name)
        for parameter_name, parameter in core.named_parameters():
            if ".ref." in parameter_name:
                parameter.requires_grad_(False)

    class PreferenceTrainer(Trainer):
        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, torch.Tensor],
            return_outputs: bool = False,
            num_items_in_batch: torch.Tensor | None = None,
        ) -> Any:
            set_adapter(model, "default")
            policy_chosen, policy_rejected = pair_logps(model, inputs)
            set_adapter(model, "ref")
            with torch.no_grad():
                reference_chosen, reference_rejected = pair_logps(model, inputs)
            set_adapter(model, "default")
            losses = dpo_loss(
                policy_chosen,
                policy_rejected,
                reference_chosen,
                reference_rejected,
                beta=args.beta,
            )
            weights = inputs.get("pair_loss_weight")
            loss = preference_weighted_loss(losses, weights)
            outputs = {
                "policy_margin": (policy_chosen - policy_rejected).detach().mean(),
                "reference_margin": (reference_chosen - reference_rejected).detach().mean(),
            }
            return (loss, outputs) if return_outputs else loss

    class HistoryCallback(TrainerCallback):
        def on_log(self, args: Any, state: Any, control: Any, logs=None, **kwargs: Any) -> Any:
            if logs:
                history = Path(args.output_dir).parent / "history.jsonl"
                payload = {"step": state.global_step, "epoch": state.epoch, **logs}
                with history.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload) + "\n")
            return control

    training = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
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
        label_names=["pair__labels"],
        report_to=["tensorboard"],
        logging_dir=str(args.output_dir / "tensorboard"),
        seed=args.seed,
    )
    trainer = PreferenceTrainer(
        model=model,
        args=training,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[HistoryCallback()],
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
