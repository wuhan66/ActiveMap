"""Safety-targeted DPO for the ActiveMap maintenance controller."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any


class ControlledStop(RuntimeError):
    """Raised when a requested stop must prevent publication of a final model."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("preference_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--eval-jsonl", type=Path)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
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


def _audit_rows(
    rows: list[dict[str, Any]], tokenizer: Any, max_length: int
) -> dict[str, Any]:
    families: Counter[str] = Counter()
    nonpositive_margins = 0
    truncated = 0
    max_observed_length = 0
    for row in rows:
        family = str(row.get("preference_family", "unspecified"))
        families[family] += 1
        margin = float(row["chosen_utility"]) - float(row["rejected_utility"])
        nonpositive_margins += margin <= 0
        chosen_length = len(
            tokenizer(
                str(row["prompt"]) + str(row["chosen"]),
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]
        )
        rejected_length = len(
            tokenizer(
                str(row["prompt"]) + str(row["rejected"]),
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]
        )
        observed = max(chosen_length, rejected_length)
        max_observed_length = max(max_observed_length, observed)
        truncated += observed > max_length
    if nonpositive_margins:
        raise ValueError(f"found {nonpositive_margins} non-positive preference margins")
    return {
        "samples": len(rows),
        "families": dict(sorted(families.items())),
        "nonpositive_margins": nonpositive_margins,
        "max_observed_length": max_observed_length,
        "truncated": truncated,
    }


def _tokenizer_source(model: str, peft_config_class: Any) -> str:
    model_path = Path(model)
    if not (model_path / "adapter_config.json").is_file():
        return model
    config = peft_config_class.from_pretrained(model)
    source = str(config.base_model_name_or_path)
    if not source:
        raise ValueError(f"adapter has no base_model_name_or_path: {model}")
    return source


def _rows_digest(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _persistent_dataset(
    rows: list[dict[str, Any]], cache_root: Path, name: str, dataset_class: Any
) -> Any:
    cache_dir = cache_root / name
    manifest_path = cache_root / f"{name}.manifest.json"
    expected = {"row_count": len(rows), "sha256": _rows_digest(rows)}
    if cache_dir.exists() or manifest_path.exists():
        if not cache_dir.is_dir() or not manifest_path.is_file():
            raise ValueError(f"incomplete persistent dataset cache: {name}")
        actual = json.loads(manifest_path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError(f"persistent dataset cache does not match input: {name}")
        return dataset_class.load_from_disk(str(cache_dir))

    cache_root.mkdir(parents=True, exist_ok=True)
    dataset = dataset_class.from_list(rows)
    dataset.save_to_disk(str(cache_dir))
    manifest_path.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    return dataset_class.load_from_disk(str(cache_dir))


def _audit_reference_adapter(model: Any) -> dict[str, Any]:
    configs = getattr(model, "peft_config", {})
    if "default" not in configs or "ref" not in configs:
        raise RuntimeError(
            "DPO requires copied 'default' and 'ref' PEFT adapters so the frozen "
            "SFT policy, rather than the base model, defines reference log probabilities"
        )

    parameters = dict(model.named_parameters())
    default_names = [name for name in parameters if ".default." in name]
    if not default_names:
        raise RuntimeError("DPO model exposes no default-adapter parameters")

    trainable_reference = []
    mismatched = []
    for name in default_names:
        ref_name = name.replace(".default.", ".ref.")
        ref_parameter = parameters.get(ref_name)
        if ref_parameter is None:
            raise RuntimeError(f"reference adapter is missing parameter: {ref_name}")
        if ref_parameter.requires_grad:
            trainable_reference.append(ref_name)
        if not bool(parameters[name].detach().equal(ref_parameter.detach())):
            mismatched.append(ref_name)
    if trainable_reference:
        raise RuntimeError(
            f"reference adapter has {len(trainable_reference)} trainable parameters"
        )
    if mismatched:
        raise RuntimeError(
            f"reference adapter differs from SFT initialization in {len(mismatched)} parameters"
        )
    return {
        "source": "copied_sft_default_adapter",
        "adapter_name": "ref",
        "parameter_tensors_compared": len(default_names),
        "exact_initial_match": True,
        "trainable_reference_parameters": 0,
    }


def _wait_for_control(control_dir: Path, poll_seconds: float = 10.0) -> None:
    while (control_dir / "PAUSE").exists() and not (control_dir / "STOP").exists():
        time.sleep(poll_seconds)
    if (control_dir / "STOP").exists():
        raise ControlledStop("control/STOP requested")


def _append_metric_event(path: Path, event: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, separators=(",", ":"), default=float) + "\n")
        handle.flush()


def main() -> None:
    args = parse_args()
    import torch
    from datasets import Dataset
    from peft import AutoPeftModelForCausalLM, PeftConfig
    from transformers import AutoTokenizer, TrainerCallback, set_seed
    from trl import DPOConfig, DPOTrainer

    if "processing_class" not in inspect.signature(DPOTrainer.__init__).parameters:
        raise RuntimeError("installed TRL does not support processing_class")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    control_dir = args.output_dir / "control"
    control_dir.mkdir(exist_ok=True)
    tokenizer_source = _tokenizer_source(args.model, PeftConfig)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_rows = _read_jsonl(args.preference_jsonl, args.max_train_samples)
    eval_rows = (
        _read_jsonl(args.eval_jsonl, args.max_eval_samples)
        if args.eval_jsonl is not None
        else None
    )
    eval_strategy = "steps" if eval_rows is not None else "no"
    config = DPOConfig(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        max_length=args.max_length,
        beta=args.beta,
        logging_steps=args.logging_steps,
        eval_strategy=eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=eval_rows is not None,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=True,
        precompute_ref_log_probs=True,
        seed=args.seed,
        report_to=["tensorboard"],
        logging_dir=str(args.output_dir / "tensorboard"),
    )
    data_summary = {
        "max_length": args.max_length,
        "train": _audit_rows(train_rows, tokenizer, args.max_length),
        "eval": (
            _audit_rows(eval_rows, tokenizer, args.max_length)
            if eval_rows is not None
            else None
        ),
        "precompute_ref_log_probs": True,
        "tokenizer_source": tokenizer_source,
        "environment": {
            "torch": torch.__version__,
            "trl": __import__("trl").__version__,
            "dpo_processing_class_api": True,
        },
        "persistent_dataset_cache": True,
    }
    (args.output_dir / "data_summary.json").write_text(
        json.dumps(data_summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(data_summary, indent=2), flush=True)
    if args.dry_run:
        return

    set_seed(args.seed)
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.model,
        is_trainable=True,
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    cache_root = args.output_dir / "dataset_cache"
    train_dataset = _persistent_dataset(train_rows, cache_root, "train", Dataset)
    eval_dataset = (
        _persistent_dataset(eval_rows, cache_root, "eval", Dataset)
        if eval_rows is not None
        else None
    )
    class ControlCallback(TrainerCallback):
        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            while (control_dir / "PAUSE").exists() and not (control_dir / "STOP").exists():
                time.sleep(10)
            if (control_dir / "STOP").exists():
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> Any:
            if logs:
                _append_metric_event(
                    Path(args.output_dir).parent / "history.jsonl",
                    {"step": state.global_step, "epoch": state.epoch, **logs},
                )
            return control

    class ControlAwareDPOTrainer(DPOTrainer):
        def compute_ref_log_probs(self, *args: Any, **kwargs: Any) -> Any:
            _wait_for_control(control_dir)
            return super().compute_ref_log_probs(*args, **kwargs)

    trainer = ControlAwareDPOTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[ControlCallback()],
    )
    data_summary["reference_policy"] = _audit_reference_adapter(trainer.model)
    (args.output_dir / "data_summary.json").write_text(
        json.dumps(data_summary, indent=2) + "\n", encoding="utf-8"
    )
    try:
        train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    except ControlledStop as error:
        (args.output_dir / "stopped_status.json").write_text(
            json.dumps({"stopped": True, "reason": str(error)}, indent=2) + "\n",
            encoding="utf-8",
        )
        return
    if (control_dir / "STOP").exists():
        trainer.save_state()
        (args.output_dir / "stopped_status.json").write_text(
            json.dumps(
                {
                    "stopped": True,
                    "reason": "control/STOP requested after optimizer step",
                    "global_step": trainer.state.global_step,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return
    trainer.save_model(str(args.output_dir / "final"))
    tokenizer.save_pretrained(args.output_dir / "final")
    trainer.save_state()
    (args.output_dir / "train_metrics.json").write_text(
        json.dumps(train_result.metrics, indent=2) + "\n", encoding="utf-8"
    )
    if eval_rows is not None:
        metrics = trainer.evaluate()
        (args.output_dir / "eval_metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
