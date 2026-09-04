#!/usr/bin/env python3
"""Run visual-controller SFT seeds on at most two pre-checked GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.train_semantic_vlm_sft import adapter_initialization_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("train_jsonl", type=Path)
    parser.add_argument("val_jsonl", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--gpu", type=int, action="append", required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--monitor-interval", type=float, default=5.0)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--expected-train-records", type=int)
    parser.add_argument("--expected-val-records", type=int)
    parser.add_argument("--expected-train-use-tool", type=int)
    parser.add_argument("--expected-val-use-tool", type=int)
    parser.add_argument("--expected-train-action", action="append", default=[])
    parser.add_argument("--expected-val-action", action="append", default=[])
    parser.add_argument("--init-adapter", type=Path)
    parser.add_argument("--acquire-sampling-target", type=float)
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


def gpu_processes(gpu: int) -> list[str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def assignments(seeds: list[int], gpus: list[int]) -> list[tuple[int, int]]:
    return [(seed, gpus[index % len(gpus)]) for index, seed in enumerate(seeds)]


def gpu_snapshot(gpu: int) -> dict[str, Any]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    memory_used, utilization = [int(value.strip()) for value in result.stdout.split(",")]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "memory_used_mib": memory_used,
        "utilization_percent": utilization,
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def read_last_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    return json.loads(lines[-1]) if lines else None


def audit_sft_jsonl(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    action_counts: Counter[str] = Counter()
    stage_counts: Counter[str] = Counter()
    task_ids: set[str] = set()
    records = 0
    invalid_action_records = 0
    with path.open("rb") as handle:
        for raw_line in handle:
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            records += 1
            try:
                row = json.loads(raw_line)
                stage_counts[str(row.get("stage", "UNKNOWN"))] += 1
                if row.get("task_id") is not None:
                    task_ids.add(str(row["task_id"]))
                assistant = next(
                    message
                    for message in reversed(row["messages"])
                    if message["role"] == "assistant"
                )
                content = assistant["content"]
                if isinstance(content, list):
                    text = next(
                        block["text"]
                        for block in reversed(content)
                        if block.get("type") == "text"
                    )
                else:
                    text = str(content)
                payload = json.loads(text)
                if "action" in payload:
                    action = str(payload["action"])
                elif payload.get("stage") == "SELECT":
                    action = str(payload["selection"])
                elif payload.get("stage") in {"TOOL", "TERMINAL"}:
                    action = str(payload["executable_action"]["action"])
                else:
                    action = str(payload["stage"])
                action_counts[action] += 1
            except (KeyError, StopIteration, TypeError, ValueError, json.JSONDecodeError):
                invalid_action_records += 1
    return {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "records": records,
        "action_counts": dict(sorted(action_counts.items())),
        "stage_counts": dict(sorted(stage_counts.items())),
        "task_count": len(task_ids),
        "invalid_action_records": invalid_action_records,
    }


def validate_data_contract(args: argparse.Namespace, audits: dict[str, Any]) -> None:
    checks = {
        "train.records": (args.expected_train_records, audits["train"]["records"]),
        "val.records": (args.expected_val_records, audits["val"]["records"]),
        "train.USE_TOOL": (
            args.expected_train_use_tool,
            audits["train"]["action_counts"].get("USE_TOOL", 0),
        ),
        "val.USE_TOOL": (
            args.expected_val_use_tool,
            audits["val"]["action_counts"].get("USE_TOOL", 0),
        ),
        "train.invalid_action_records": (0, audits["train"]["invalid_action_records"]),
        "val.invalid_action_records": (0, audits["val"]["invalid_action_records"]),
    }
    mismatches = {
        name: {"expected": expected, "actual": actual}
        for name, (expected, actual) in checks.items()
        if expected is not None and expected != actual
    }
    for split in ("train", "val"):
        expected_values = getattr(args, f"expected_{split}_action", [])
        for value in expected_values:
            if "=" not in value:
                raise ValueError("expected actions must use ACTION=COUNT")
            action, count = value.split("=", 1)
            actual = audits[split]["action_counts"].get(action, 0)
            if int(count) != actual:
                mismatches[f"{split}.{action}"] = {
                    "expected": int(count),
                    "actual": actual,
                }
    if mismatches:
        raise ValueError(f"SFT data contract mismatch: {mismatches}")


def command(args: argparse.Namespace, seed: int, output_dir: Path) -> list[str]:
    result = [
        args.python,
        "scripts/train_semantic_vlm_sft.py",
        str(args.model),
        str(args.train_jsonl),
        str(output_dir),
        "--eval-jsonl",
        str(args.val_jsonl),
        "--epochs",
        str(args.epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--batch-size",
        str(args.batch_size),
        "--gradient-accumulation",
        str(args.gradient_accumulation),
        "--max-length",
        str(args.max_length),
        "--lora-rank",
        str(args.lora_rank),
        "--lora-alpha",
        str(args.lora_alpha),
        "--seed",
        str(seed),
        "--logging-steps",
        str(args.logging_steps),
        "--eval-steps",
        str(args.eval_steps),
        "--save-steps",
        str(args.save_steps),
        "--save-total-limit",
        str(args.save_total_limit),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--dataloader-num-workers",
        str(args.dataloader_num_workers),
        "--dataloader-prefetch-factor",
        str(args.dataloader_prefetch_factor),
        "--attn-implementation",
        args.attn_implementation,
    ]
    if args.dataloader_persistent_workers:
        result.append("--dataloader-persistent-workers")
    if args.max_train_samples is not None:
        result.extend(["--max-train-samples", str(args.max_train_samples)])
    if args.max_eval_samples is not None:
        result.extend(["--max-eval-samples", str(args.max_eval_samples)])
    init_adapter = getattr(args, "init_adapter", None)
    if init_adapter is not None:
        result.extend(["--init-adapter", str(init_adapter)])
    if args.acquire_sampling_target is not None:
        result.extend(["--acquire-sampling-target", str(args.acquire_sampling_target)])
    return result


def run_worker(args: argparse.Namespace, gpu: int, seeds: list[int]) -> list[dict[str, Any]]:
    results = []
    for seed in seeds:
        active = gpu_processes(gpu)
        if active:
            raise RuntimeError(f"refusing newly occupied GPU {gpu}: {active}")
        output_dir = args.output_root / f"seed{seed}"
        if output_dir.exists():
            raise FileExistsError(f"refusing to overwrite {output_dir}")
        output_dir.mkdir(parents=True)
        log_path = output_dir / "train.log"
        resource_path = output_dir / "resource_history.jsonl"
        state_path = output_dir / "run_state.json"
        history_path = output_dir / "history.jsonl"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env.setdefault("PYTHONPATH", "src:.")
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        started = datetime.now(timezone.utc).isoformat()
        trainer_command = command(args, seed, output_dir)
        peak_memory_mib = 0
        peak_utilization = 0
        resource_samples = 0
        with (
            log_path.open("w", encoding="utf-8", buffering=1) as log,
            resource_path.open("w", encoding="utf-8", buffering=1) as resource_log,
        ):
            process = subprocess.Popen(
                trainer_command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            write_json(
                state_path,
                {
                    "status": "running",
                    "seed": seed,
                    "gpu": gpu,
                    "pid": process.pid,
                    "started_utc": started,
                },
            )
            while process.poll() is None:
                try:
                    sample = gpu_snapshot(gpu)
                except (subprocess.SubprocessError, ValueError) as error:
                    sample = {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "error": str(error),
                    }
                if "memory_used_mib" in sample:
                    peak_memory_mib = max(peak_memory_mib, sample["memory_used_mib"])
                    peak_utilization = max(
                        peak_utilization, sample["utilization_percent"]
                    )
                resource_samples += 1
                resource_log.write(json.dumps(sample, sort_keys=True) + "\n")
                write_json(
                    state_path,
                    {
                        "status": "running",
                        "seed": seed,
                        "gpu": gpu,
                        "pid": process.pid,
                        "started_utc": started,
                        "last_resource_sample": sample,
                        "last_training_event": read_last_jsonl(history_path),
                        "peak_memory_used_mib": peak_memory_mib,
                        "peak_utilization_percent": peak_utilization,
                    },
                )
                time.sleep(args.monitor_interval)
            returncode = process.wait()
        result = {
            "seed": seed,
            "gpu": gpu,
            "pid": process.pid,
            "returncode": returncode,
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(output_dir.resolve()),
            "log": str(log_path.resolve()),
            "resource_history": str(resource_path.resolve()),
            "resource_samples": resource_samples,
            "peak_memory_used_mib": peak_memory_mib,
            "peak_utilization_percent": peak_utilization,
            "trainer_command": trainer_command,
        }
        results.append(result)
        write_json(output_dir / "process_result.json", result)
        write_json(
            state_path,
            {
                "status": "completed" if returncode == 0 else "failed",
                **result,
            },
        )
        if returncode:
            break
    return results


def main() -> None:
    args = parse_args()
    gpus = list(dict.fromkeys(args.gpu))
    seeds = list(dict.fromkeys(args.seed))
    if not gpus or len(gpus) > 2:
        raise ValueError("this launcher permits one or two GPUs only")
    if not seeds:
        raise ValueError("at least one seed is required")
    numeric_values = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "max_length": args.max_length,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "monitor_interval": args.monitor_interval,
        "dataloader_prefetch_factor": args.dataloader_prefetch_factor,
    }
    invalid = {name: value for name, value in numeric_values.items() if value <= 0}
    optional_sample_limits = {
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
    }
    invalid.update(
        {
            name: value
            for name, value in optional_sample_limits.items()
            if value is not None and value <= 0
        }
    )
    if args.acquire_sampling_target is not None and not (
        0.0 < args.acquire_sampling_target < 1.0
    ):
        invalid["acquire_sampling_target"] = args.acquire_sampling_target
    if invalid or args.early_stopping_patience < 0:
        raise ValueError(f"invalid training settings: {invalid}")
    if args.dataloader_num_workers < 0:
        raise ValueError("dataloader-num-workers must be non-negative")
    if args.dataloader_persistent_workers and args.dataloader_num_workers == 0:
        raise ValueError("persistent workers require a nonzero worker count")
    for path in (args.model, args.train_jsonl, args.val_jsonl):
        if not path.exists():
            raise FileNotFoundError(path)
    initialization_adapter = (
        adapter_initialization_manifest(args.init_adapter)
        if args.init_adapter is not None
        else None
    )
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    audits = {
        "train": audit_sft_jsonl(args.train_jsonl),
        "val": audit_sft_jsonl(args.val_jsonl),
    }
    validate_data_contract(args, audits)
    occupied = {gpu: gpu_processes(gpu) for gpu in gpus}
    occupied = {gpu: rows for gpu, rows in occupied.items() if rows}
    if occupied:
        raise RuntimeError(f"refusing occupied GPUs: {occupied}")
    schedule = assignments(seeds, gpus)
    manifest = {
        "schema_version": "semantic-vlm-sft-launch-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model.resolve()),
        "initialization_adapter": initialization_adapter,
        "train_jsonl": str(args.train_jsonl.resolve()),
        "val_jsonl": str(args.val_jsonl.resolve()),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "effective_batch_size": args.batch_size * args.gradient_accumulation,
        "max_length": args.max_length,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "early_stopping_patience": args.early_stopping_patience,
        "acquire_sampling_target": args.acquire_sampling_target,
        "monitor_interval_seconds": args.monitor_interval,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_prefetch_factor": args.dataloader_prefetch_factor,
        "dataloader_persistent_workers": args.dataloader_persistent_workers,
        "attention_implementation": args.attn_implementation,
        "data_audit": audits,
        "schedule": [{"seed": seed, "gpu": gpu} for seed, gpu in schedule],
        "test_assets_read": False,
    }
    print(json.dumps(manifest, indent=2), flush=True)
    if args.dry_run:
        return
    args.output_root.mkdir(parents=True, exist_ok=False)
    write_json(args.output_root / "launch_manifest.json", manifest)
    queues = {gpu: [seed for seed, assigned in schedule if assigned == gpu] for gpu in gpus}
    all_results = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {
            executor.submit(run_worker, args, gpu, queue): gpu
            for gpu, queue in queues.items()
            if queue
        }
        for future in as_completed(futures):
            all_results.extend(future.result())
    all_results.sort(key=lambda row: seeds.index(int(row["seed"])))
    write_json(args.output_root / "process_results.json", all_results)
    if any(result["returncode"] for result in all_results) or len(all_results) != len(seeds):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
