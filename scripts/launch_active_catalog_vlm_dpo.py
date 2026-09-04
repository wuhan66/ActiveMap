#!/usr/bin/env python3
"""Launch one multimodal DPO seed with GPU ownership and resource logging."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("sft_adapter", type=Path)
    parser.add_argument("train_jsonl", type=Path)
    parser.add_argument("val_jsonl", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--python", default=sys.executable)
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
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--monitor-interval", type=float, default=5.0)
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


def gpu_snapshot(gpu: int) -> dict[str, Any]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    memory, utilization, power, temperature = [
        value.strip() for value in result.stdout.split(",")
    ]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "memory_used_mib": int(memory),
        "utilization_percent": int(utilization),
        "power_watts": float(power),
        "temperature_c": int(temperature),
    }


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def last_history(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    return json.loads(lines[-1]) if lines else None


def trainer_command(args: argparse.Namespace, output: Path) -> list[str]:
    command = [
        args.python,
        "scripts/train_active_catalog_vlm_dpo.py",
        str(args.sft_adapter),
        str(args.train_jsonl),
        str(output),
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
        "--beta",
        str(args.beta),
        "--family-balance",
        str(args.family_balance),
        "--family-balance-power",
        str(args.family_balance_power),
        "--safe-acquire-boost",
        str(args.safe_acquire_boost),
        "--unsafe-stop-scale",
        str(args.unsafe_stop_scale),
        "--seed",
        str(args.seed),
        "--logging-steps",
        str(args.logging_steps),
        "--eval-steps",
        str(args.eval_steps),
        "--save-steps",
        str(args.save_steps),
    ]
    if args.max_train_samples is not None:
        command.extend(["--max-train-samples", str(args.max_train_samples)])
    if args.max_eval_samples is not None:
        command.extend(["--max-eval-samples", str(args.max_eval_samples)])
    return command


def main() -> None:
    args = parse_args()
    numeric = {
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "max_length": args.max_length,
        "beta": args.beta,
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "monitor_interval": args.monitor_interval,
    }
    invalid = {name: value for name, value in numeric.items() if value <= 0}
    if invalid:
        raise ValueError(f"invalid DPO launch settings: {invalid}")
    for path in (args.sft_adapter, args.train_jsonl, args.val_jsonl):
        if not path.exists():
            raise FileNotFoundError(path)
    if not (args.sft_adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(args.sft_adapter / "adapter_config.json")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    occupied = gpu_processes(args.gpu)
    if occupied:
        raise RuntimeError(f"refusing occupied GPU {args.gpu}: {occupied}")
    seed_output = args.output_root / f"seed{args.seed}"
    command = trainer_command(args, seed_output)
    manifest = {
        "schema_version": "active-catalog-vlm-dpo-launch-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "physical_gpu": args.gpu,
        "sft_adapter": str(args.sft_adapter.resolve()),
        "train_jsonl": str(args.train_jsonl.resolve()),
        "val_jsonl": str(args.val_jsonl.resolve()),
        "trainer_command": command,
        "test_assets_read": False,
    }
    print(json.dumps(manifest, indent=2), flush=True)
    if args.dry_run:
        return
    args.output_root.mkdir(parents=True)
    write_json(args.output_root / "launch_manifest.json", manifest)
    log_path = args.output_root / "train.log"
    resource_path = args.output_root / "resource_history.jsonl"
    state_path = args.output_root / "run_state.json"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env.setdefault("PYTHONPATH", "src:.")
    env["PYTHONUNBUFFERED"] = "1"
    started = datetime.now(timezone.utc).isoformat()
    peak_memory = 0
    peak_utilization = 0
    samples = 0
    with (
        log_path.open("w", encoding="utf-8", buffering=1) as log,
        resource_path.open("w", encoding="utf-8", buffering=1) as resources,
    ):
        process = subprocess.Popen(
            command, env=env, stdout=log, stderr=subprocess.STDOUT, text=True
        )
        while process.poll() is None:
            try:
                sample = gpu_snapshot(args.gpu)
                peak_memory = max(peak_memory, int(sample["memory_used_mib"]))
                peak_utilization = max(
                    peak_utilization, int(sample["utilization_percent"])
                )
            except (subprocess.SubprocessError, ValueError) as error:
                sample = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "error": str(error),
                }
            samples += 1
            resources.write(json.dumps(sample, separators=(",", ":")) + "\n")
            write_json(
                state_path,
                {
                    "status": "running",
                    "pid": process.pid,
                    "seed": args.seed,
                    "physical_gpu": args.gpu,
                    "started_utc": started,
                    "last_resource_sample": sample,
                    "last_training_event": last_history(seed_output / "history.jsonl"),
                    "peak_memory_used_mib": peak_memory,
                    "peak_utilization_percent": peak_utilization,
                },
            )
            time.sleep(args.monitor_interval)
        returncode = process.wait()
    result = {
        "status": "completed" if returncode == 0 else "failed",
        "returncode": returncode,
        "pid": process.pid,
        "seed": args.seed,
        "physical_gpu": args.gpu,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "peak_memory_used_mib": peak_memory,
        "peak_utilization_percent": peak_utilization,
        "resource_samples": samples,
        "output": str(seed_output.resolve()),
        "log": str(log_path.resolve()),
        "resource_history": str(resource_path.resolve()),
    }
    write_json(args.output_root / "process_result.json", result)
    write_json(state_path, result)
    if returncode:
        raise SystemExit(returncode)


if __name__ == "__main__":
    main()
