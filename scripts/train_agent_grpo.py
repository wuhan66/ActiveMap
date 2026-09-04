#!/usr/bin/env python3
"""Offline contextual-GRPO ablation over frozen one-step utilities.

This is intentionally not the recurrent executable-trajectory paper method.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

from activemap.agent.records import AgentObservation
from activemap.agent.rl import (
    executable_schema_reward,
    sparse_acquisition_reward,
    target_terminal_action,
    task_utility_reward,
    terminal_safety_reward,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Promoted SFT adapter directory")
    parser.add_argument("train_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--eval-jsonl", type=Path)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=96)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--schema-weight", type=float, default=1.0)
    parser.add_argument("--safety-weight", type=float, default=1.0)
    parser.add_argument("--sparsity-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--allow-no-grounded-tool-data", action="store_true")
    parser.add_argument("--contextual-ablation-only", action="store_true")
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
    if any(row.get("split") == "test" for row in rows):
        raise ValueError("GRPO training/evaluation must not read test rows")
    return rows


def _rows_digest(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _audit(
    rows: list[dict[str, Any]], expected_split: str
) -> tuple[dict[str, Any], set[str]]:
    required = {
        "prompt",
        "observation_json",
        "oracle_utilities",
        "target_action_key",
        "stop_utility",
        "oracle_best_action_key",
        "oracle_best_advantage",
        "trajectory_id",
        "step",
        "split",
    }
    targets: Counter[str] = Counter()
    oracle_best_counts: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    trajectories: set[str] = set()
    tasks: set[str] = set()
    states: set[tuple[str, int]] = set()
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"row {index} is missing fields: {sorted(missing)}")
        if row["split"] != expected_split:
            raise ValueError(f"row {index} has split={row['split']!r}")
        observation = AgentObservation.model_validate_json(row["observation_json"])
        if observation.split != expected_split:
            raise ValueError(f"row {index} observation has split={observation.split!r}")
        if int(row["step"]) != observation.step:
            raise ValueError(f"row {index} step does not match its observation")
        prompt = row["prompt"]
        if (
            not isinstance(prompt, list)
            or not prompt
            or not isinstance(prompt[-1], dict)
            or prompt[-1].get("role") != "user"
            or prompt[-1].get("content") != row["observation_json"]
        ):
            raise ValueError(f"row {index} prompt does not end with its observation")
        utilities = {str(key): float(value) for key, value in row["oracle_utilities"].items()}
        if not utilities or not all(math.isfinite(value) for value in utilities.values()):
            raise ValueError(f"row {index} has empty or non-finite oracle utilities")
        if str(row["target_action_key"]) not in utilities:
            raise ValueError(f"row {index} target_action_key lacks frozen utility")
        oracle_best = max(utilities, key=utilities.__getitem__)
        if row["oracle_best_action_key"] != oracle_best:
            raise ValueError(f"row {index} oracle_best_action_key is inconsistent")
        _, stop_utility = target_terminal_action(utilities)
        if not math.isclose(float(row["stop_utility"]), stop_utility, abs_tol=1e-8):
            raise ValueError(f"row {index} stop_utility is inconsistent")
        expected_advantage = utilities[oracle_best] - stop_utility
        if not math.isclose(
            float(row["oracle_best_advantage"]), expected_advantage, abs_tol=1e-8
        ):
            raise ValueError(f"row {index} oracle_best_advantage is inconsistent")
        state = (str(row["trajectory_id"]), int(row["step"]))
        if state in states:
            raise ValueError(f"row {index} duplicates RL state {state}")
        states.add(state)
        targets[str(row["target_action_key"]).split(":", 1)[0]] += 1
        oracle_best_counts[str(row["oracle_best_action_key"]).split(":", 1)[0]] += 1
        sources[str(row.get("rl_source", "counterfactual_trajectory"))] += 1
        trajectories.add(str(row["trajectory_id"]))
        tasks.add(observation.task_id)
    return (
        {
            "states": len(rows),
            "trajectories": len(trajectories),
            "tasks": len(tasks),
            "target_action_counts": dict(sorted(targets.items())),
            "oracle_best_action_counts": dict(sorted(oracle_best_counts.items())),
            "source_counts": dict(sorted(sources.items())),
            "sha256": _rows_digest(rows),
        },
        tasks,
    )


def audit_rl_splits(
    train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]] | None
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    train_summary, train_tasks = _audit(train_rows, "train")
    if eval_rows is None:
        return train_summary, None
    eval_summary, eval_tasks = _audit(eval_rows, "val")
    overlap = sorted(train_tasks & eval_tasks)
    if overlap:
        raise ValueError(
            f"GRPO train/validation task leakage ({len(overlap)} tasks), first={overlap[0]}"
        )
    return train_summary, eval_summary


def main() -> None:
    args = parse_args()
    if not args.contextual_ablation_only:
        raise ValueError(
            "train_agent_grpo.py is an offline contextual ablation; pass "
            "--contextual-ablation-only explicitly or use the recurrent GRPO pipeline"
        )
    if args.num_generations < 2:
        raise ValueError("num-generations must be at least 2")
    if args.batch_size % args.num_generations:
        raise ValueError("batch-size must be divisible by num-generations")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    control_dir = args.output_dir / "control"
    control_dir.mkdir(exist_ok=True)
    train_rows = _read_jsonl(args.train_jsonl, args.max_train_samples)
    eval_rows = (
        _read_jsonl(args.eval_jsonl, args.max_eval_samples) if args.eval_jsonl is not None else None
    )
    train_audit, eval_audit = audit_rl_splits(train_rows, eval_rows)
    summary = {
        "method": "offline_contextual_grpo_ablation",
        "claim_boundary": "offline state-action optimization; not online tool rollout",
        "train": train_audit,
        "eval": eval_audit,
        "optimization": {
            "seed": args.seed,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "num_generations": args.num_generations,
            "max_completion_length": args.max_completion_length,
            "kl_beta": args.beta,
        },
        "reward_weights": {
            "task_utility": 1.0,
            "executable_schema": args.schema_weight,
            "terminal_safety": args.safety_weight,
            "sparse_acquisition": args.sparsity_weight,
        },
        "test_assets_read": False,
    }
    train_sources = summary["train"]["source_counts"]
    eval_sources = summary["eval"]["source_counts"] if summary["eval"] else {}
    grounded_source = "grounded_sparse_tool_preference"
    if not args.allow_no_grounded_tool_data and (
        train_sources.get(grounded_source, 0) <= 0
        or (eval_rows is not None and eval_sources.get(grounded_source, 0) <= 0)
    ):
        raise ValueError(
            "full Agent GRPO requires natural grounded tool states in train and validation; "
            "use --allow-no-grounded-tool-data only for a diagnostic smoke"
        )
    (args.output_dir / "data_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    if args.dry_run:
        return

    import torch
    from datasets import Dataset
    from peft import AutoPeftModelForCausalLM, PeftConfig
    from transformers import AutoTokenizer, TrainerCallback, set_seed
    from trl import GRPOConfig, GRPOTrainer

    set_seed(args.seed)
    peft_config = PeftConfig.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(
        peft_config.base_model_name_or_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.model,
        is_trainable=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    config = GRPOConfig(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        beta=args.beta,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if eval_rows is not None else "no",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        seed=args.seed,
        report_to=["tensorboard"],
        logging_dir=str(args.output_dir / "tensorboard"),
        reward_weights=[
            1.0,
            args.schema_weight,
            args.safety_weight,
            args.sparsity_weight,
        ],
        log_completions=True,
        num_completions_to_print=4,
    )
    history_path = args.output_dir / "history.jsonl"

    class ControlCallback(TrainerCallback):
        def on_step_end(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
            while (control_dir / "PAUSE").exists() and not (control_dir / "STOP").exists():
                time.sleep(10)
            if (control_dir / "STOP").exists():
                control.should_training_stop = True
            return control

        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: dict[str, Any] | None = None,
            **_: Any,
        ) -> None:
            event = {
                "step": int(state.global_step),
                "epoch": float(state.epoch) if state.epoch is not None else None,
                **{
                    key: float(value) if isinstance(value, int | float) else value
                    for key, value in (logs or {}).items()
                },
            }
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, separators=(",", ":")) + "\n")

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            task_utility_reward,
            executable_schema_reward,
            terminal_safety_reward,
            sparse_acquisition_reward,
        ],
        args=config,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(eval_rows) if eval_rows is not None else None,
        processing_class=tokenizer,
        callbacks=[ControlCallback()],
    )
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if (control_dir / "STOP").exists():
        (args.output_dir / "STOPPED").write_text("control/STOP requested\n")
        return
    final_dir = args.output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    metrics = {key: float(value) for key, value in result.metrics.items()}
    (args.output_dir / "train_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
