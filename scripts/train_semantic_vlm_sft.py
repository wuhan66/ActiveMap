#!/usr/bin/env python3
"""LoRA SFT for the Gemma-3 visual semantic-tool controller."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("train_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stopping-threshold", type=float, default=0.0)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--init-adapter", type=Path)
    parser.add_argument("--acquire-sampling-target", type=float)
    parser.add_argument("--acquire-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--record-weight-key",
        help="Optional positive scalar metadata key used only to weight the training loss.",
    )
    parser.add_argument("--teacher-loss-model-selection", action="store_true")
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--dataloader-prefetch-factor", type=int, default=2)
    parser.add_argument("--dataloader-persistent-workers", action="store_true")
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_initialization_manifest(path: Path) -> dict[str, Any]:
    config_path = path / "adapter_config.json"
    candidates = [path / "adapter_model.safetensors", path / "adapter_model.bin"]
    model_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if not config_path.is_file() or model_path is None:
        raise FileNotFoundError(f"incomplete initialization adapter: {path}")
    return {
        "path": str(path.resolve()),
        "config": {
            "path": str(config_path.resolve()),
            "sha256": sha256_file(config_path),
            "bytes": config_path.stat().st_size,
        },
        "weights": {
            "path": str(model_path.resolve()),
            "sha256": sha256_file(model_path),
            "bytes": model_path.stat().st_size,
        },
    }


def _row_action(row: dict[str, Any]) -> str:
    content = row["messages"][-1]["content"]
    text = next(part["text"] for part in content if part.get("type") == "text")
    payload = json.loads(text)
    return str(payload.get("selection", payload.get("action", "UNKNOWN")))


def action_sampling_weights(
    rows: list[dict[str, Any]], acquire_target: float
) -> tuple[list[float], dict[str, Any]]:
    """Return per-record weights with a fixed expected ACQUIRE exposure."""

    if not 0.0 < acquire_target < 1.0:
        raise ValueError("acquire sampling target must be between zero and one")
    actions = [_row_action(row) for row in rows]
    counts = {action: actions.count(action) for action in sorted(set(actions))}
    acquire = counts.get("ACQUIRE", 0)
    other = len(actions) - acquire
    if acquire == 0 or other == 0:
        raise ValueError("balanced action sampling requires ACQUIRE and non-ACQUIRE rows")
    acquire_weight = acquire_target / acquire
    other_weight = (1.0 - acquire_target) / other
    weights = [acquire_weight if action == "ACQUIRE" else other_weight for action in actions]
    return weights, {
        "mode": "weighted_replacement",
        "original_action_counts": counts,
        "acquire_target_fraction": acquire_target,
        "acquire_record_weight": acquire_weight,
        "non_acquire_record_weight": other_weight,
        "samples_per_epoch": len(rows),
    }


def action_sequence_weights(
    rows: list[dict[str, Any]], acquire_weight: float
) -> list[float]:
    if acquire_weight < 1.0:
        raise ValueError("acquire loss weight must be at least one")
    return [acquire_weight if _row_action(row) == "ACQUIRE" else 1.0 for row in rows]


def record_loss_weights(
    rows: list[dict[str, Any]], key: str | None
) -> tuple[list[float], dict[str, Any]]:
    """Load optional train-only reliability weights without exposing them in prompts."""

    if key is None:
        return [1.0] * len(rows), {"mode": "none"}
    weights = []
    for index, row in enumerate(rows):
        if key not in row:
            raise ValueError(f"record {index} is missing --record-weight-key={key}")
        weight = float(row[key])
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"record {index} has an invalid {key}={weight}")
        weights.append(weight)
    return weights, {
        "mode": "loss_only",
        "key": key,
        "minimum": min(weights),
        "maximum": max(weights),
        "mean": sum(weights) / len(weights),
    }


def main() -> None:
    args = parse_args()
    if args.early_stopping_patience < 0:
        raise ValueError("early-stopping-patience must be non-negative")
    if args.acquire_loss_weight < 1.0:
        raise ValueError("acquire-loss-weight must be at least one")
    if args.acquire_loss_weight != 1.0 and args.batch_size != 1:
        raise ValueError("action-weighted loss currently requires batch-size=1")
    if args.record_weight_key is not None and args.batch_size != 1:
        raise ValueError("record-weighted loss currently requires batch-size=1")
    if args.dataloader_num_workers < 0 or args.dataloader_prefetch_factor <= 0:
        raise ValueError("invalid DataLoader worker or prefetch configuration")
    if args.dataloader_persistent_workers and args.dataloader_num_workers == 0:
        raise ValueError("persistent DataLoader workers require at least one worker")
    import torch
    from transformers import AutoConfig, AutoProcessor, set_seed

    from activemap.agent.vlm_sft import (
        VisualActionSFTCollator,
        VisualActionSFTDataset,
        load_vlm_sft_rows,
    )

    set_seed(args.seed)
    initialization = (
        adapter_initialization_manifest(args.init_adapter)
        if args.init_adapter is not None
        else None
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    control_dir = args.output_dir / "control"
    control_dir.mkdir(exist_ok=True)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_rows = load_vlm_sft_rows(args.train_jsonl, args.max_train_samples)
    eval_rows = load_vlm_sft_rows(args.eval_jsonl, args.max_eval_samples)
    sampling_weights = None
    sampling_summary = {"mode": "shuffle_without_replacement"}
    if args.acquire_sampling_target is not None:
        sampling_weights, sampling_summary = action_sampling_weights(
            train_rows, args.acquire_sampling_target
        )
    action_weights = action_sequence_weights(train_rows, args.acquire_loss_weight)
    reliability_weights, reliability_summary = record_loss_weights(
        train_rows, args.record_weight_key
    )
    sequence_weights = [
        action_weight * reliability_weight
        for action_weight, reliability_weight in zip(
            action_weights, reliability_weights, strict=True
        )
    ]
    use_sequence_weights = (
        args.acquire_loss_weight != 1.0 or args.record_weight_key is not None
    )

    class ActionWeightedVisualDataset(VisualActionSFTDataset):
        def __getitem__(self, index: int) -> dict[str, Any]:
            encoded = super().__getitem__(index)
            if use_sequence_weights:
                encoded["example_weight"] = torch.tensor(sequence_weights[index])
            return encoded

    train_dataset = ActionWeightedVisualDataset(train_rows, processor, max_length=args.max_length)
    eval_dataset = VisualActionSFTDataset(eval_rows, processor, max_length=args.max_length)
    smoke_train = train_dataset[0]
    smoke_eval = eval_dataset[0]
    collator = VisualActionSFTCollator(tokenizer.pad_token_id)
    smoke_batch = collator(
        [train_dataset[index] for index in range(min(2, len(train_dataset)))]
    )
    data_summary = {
        "schema_version": "semantic-vlm-sft-train-v1",
        "model": str(Path(args.model).resolve()),
        "model_type": str(getattr(config, "model_type", "unknown")),
        "architectures": list(getattr(config, "architectures", None) or []),
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "max_length": args.max_length,
        "first_train_tokens": int(smoke_train["input_ids"].numel()),
        "first_train_supervised_tokens": int((smoke_train["labels"] != -100).sum()),
        "first_eval_tokens": int(smoke_eval["input_ids"].numel()),
        "collated_tensor_shapes": {
            key: list(value.shape) for key, value in sorted(smoke_batch.items())
        },
        "assistant_only_loss": True,
        "test_assets_read": False,
        "seed": args.seed,
        "initialization_adapter": initialization,
        "train_jsonl": str(args.train_jsonl.resolve()),
        "train_jsonl_sha256": sha256_file(args.train_jsonl),
        "eval_jsonl": str(args.eval_jsonl.resolve()),
        "eval_jsonl_sha256": sha256_file(args.eval_jsonl),
        "training_protocol": {
            "epochs_ceiling": args.epochs,
            "learning_rate": args.learning_rate,
            "per_device_batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "effective_batch_size": args.batch_size * args.gradient_accumulation,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "logging_steps": args.logging_steps,
            "eval_steps": args.eval_steps,
            "save_steps": args.save_steps,
            "early_stopping_patience": args.early_stopping_patience,
            "action_sampling": sampling_summary,
            "acquire_loss_weight": args.acquire_loss_weight,
            "record_reliability_weighting": reliability_summary,
            "dataloader_num_workers": args.dataloader_num_workers,
            "dataloader_prefetch_factor": args.dataloader_prefetch_factor,
            "dataloader_persistent_workers": args.dataloader_persistent_workers,
            "dataloader_pin_memory": True,
            "attention_implementation": args.attn_implementation,
            "checkpoint_selection": (
                "teacher_forced_eval_loss"
                if args.teacher_loss_model_selection
                else "external_generated_action_gate"
            ),
        },
    }
    (args.output_dir / "data_summary.json").write_text(
        json.dumps(data_summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(data_summary, indent=2), flush=True)
    if args.dry_run:
        return

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import (
        AutoModelForImageTextToText,
        EarlyStoppingCallback,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    base_model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )
    base_model.config.use_cache = False
    lora_target_pattern = (
        r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
    )
    if args.init_adapter is not None:
        model = PeftModel.from_pretrained(
            base_model,
            args.init_adapter,
            is_trainable=True,
        )
        active_name = model.active_adapter
        if isinstance(active_name, list | tuple):
            if len(active_name) != 1:
                raise ValueError("exactly one initialization adapter must be active")
            active_name = active_name[0]
        active_config = model.peft_config[str(active_name)]
        if (
            int(active_config.r) != args.lora_rank
            or int(active_config.lora_alpha) != args.lora_alpha
        ):
            raise ValueError(
                "initialization adapter rank/alpha disagree with the requested training protocol"
            )
        matched_lora_modules = [
            name for name, module in model.named_modules() if hasattr(module, "lora_A")
        ]
        lora_target_pattern = str(active_config.target_modules)
        initialization_mode = "continue_existing_lora"
    else:
        model = base_model
        matched_lora_modules = [
            name
            for name, _ in model.named_modules()
            if re.fullmatch(lora_target_pattern, name)
        ]
        if not matched_lora_modules:
            raise RuntimeError(
                f"language-only LoRA pattern matched no modules for model_type={config.model_type}"
            )
        lora = LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=lora_target_pattern,
            bias="none",
            use_rslora=True,
        )
        model = get_peft_model(model, lora)
        initialization_mode = "new_lora"
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    if not matched_lora_modules:
        raise RuntimeError(
            f"language-only LoRA pattern matched no modules for model_type={config.model_type}"
        )
    visual_trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and (name.startswith("visual.") or ".visual." in name)
    ]
    if visual_trainable:
        raise RuntimeError(f"visual-tower parameters became trainable: {visual_trainable[:3]}")
    model.print_trainable_parameters()
    data_summary.update(
        {
            "lora_target_scope": "language_model_only",
            "lora_target_pattern": lora_target_pattern,
            "lora_matched_module_count": len(matched_lora_modules),
            "visual_tower_lora_enabled": False,
            "initialization_mode": initialization_mode,
        }
    )
    (args.output_dir / "data_summary.json").write_text(
        json.dumps(data_summary, indent=2) + "\n", encoding="utf-8"
    )

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
        "data_seed": args.seed,
        "report_to": ["tensorboard"],
        "logging_dir": str(args.output_dir / "tensorboard"),
        "remove_unused_columns": False,
        "load_best_model_at_end": args.teacher_loss_model_selection,
        "ddp_find_unused_parameters": False,
        "label_names": ["labels"],
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_pin_memory": True,
        "dataloader_persistent_workers": args.dataloader_persistent_workers,
    }
    strategy_name = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(TrainingArguments).parameters
        else "evaluation_strategy"
    )
    kwargs[strategy_name] = "steps"
    if args.dataloader_num_workers > 0:
        if "dataloader_prefetch_factor" not in inspect.signature(
            TrainingArguments
        ).parameters:
            raise RuntimeError(
                "installed Transformers does not support dataloader_prefetch_factor"
            )
        kwargs["dataloader_prefetch_factor"] = args.dataloader_prefetch_factor
    training_args = TrainingArguments(**kwargs)

    class ControlCallback(TrainerCallback):
        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            while (control_dir / "PAUSE").exists() and not (control_dir / "STOP").exists():
                time.sleep(10)
            if (control_dir / "STOP").exists():
                control.should_training_stop = True
            return control

    history_path = args.output_dir / "history.jsonl"

    class JsonlHistoryCallback(TrainerCallback):
        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            row = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "step": int(state.global_step),
                "epoch": state.epoch,
                **(logs or {}),
            }
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    callbacks: list[TrainerCallback] = [ControlCallback(), JsonlHistoryCallback()]
    if args.teacher_loss_model_selection and args.early_stopping_patience:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )
    class ActionBalancedTrainer(Trainer):
        def _get_train_sampler(self, train_dataset: Any = None) -> Any:
            if sampling_weights is None:
                return super()._get_train_sampler(train_dataset)
            generator = torch.Generator()
            generator.manual_seed(args.seed)
            return torch.utils.data.WeightedRandomSampler(
                torch.tensor(sampling_weights, dtype=torch.double),
                num_samples=len(sampling_weights),
                replacement=True,
                generator=generator,
            )

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            example_weight = inputs.pop("example_weight", None)
            loss, outputs = super().compute_loss(
                model,
                inputs,
                return_outputs=True,
                num_items_in_batch=num_items_in_batch,
            )
            if example_weight is not None:
                if int(example_weight.numel()) != 1:
                    raise ValueError("action-weighted loss requires one example per microbatch")
                loss = loss * example_weight.to(device=loss.device, dtype=loss.dtype).reshape(())
            return (loss, outputs) if return_outputs else loss

    trainer = ActionBalancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    final_dir = args.output_dir / "final"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(final_dir)
    trainer.save_state()
    train_metrics = dict(train_result.metrics)
    if torch.cuda.is_available():
        train_metrics.update(
            {
                "gpu_peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "gpu_peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            }
        )
    (args.output_dir / "train_metrics.json").write_text(
        json.dumps(train_metrics, indent=2) + "\n", encoding="utf-8"
    )
    metrics = trainer.evaluate()
    (args.output_dir / "eval_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
