#!/usr/bin/env python3
"""Launch one multimodal contextual-RL seed with GPU ownership logging."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts.launch_active_catalog_vlm_dpo import (
    gpu_processes,
    gpu_snapshot,
    last_history,
    write_json,
)


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
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--monitor-interval", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def command(args: argparse.Namespace, output: Path) -> list[str]:
    result = [
        args.python, "scripts/train_active_catalog_vlm_rl.py",
        str(args.sft_adapter), str(args.train_jsonl), str(output),
        "--eval-jsonl", str(args.val_jsonl), "--epochs", str(args.epochs),
        "--learning-rate", str(args.learning_rate),
        "--gradient-accumulation", str(args.gradient_accumulation),
        "--max-length", str(args.max_length), "--action-limit", str(args.action_limit),
        "--kl-beta", str(args.kl_beta), "--entropy-weight", str(args.entropy_weight),
        "--temperature", str(args.temperature), "--seed", str(args.seed),
        "--utility-scale", str(getattr(args, "utility_scale", 1.0)),
        "--acquire-fraction", str(args.acquire_fraction),
        "--pairwise-weight", str(args.pairwise_weight),
        "--pairwise-margin", str(args.pairwise_margin),
        "--logging-steps", str(args.logging_steps), "--eval-steps", str(args.eval_steps),
        "--save-steps", str(args.save_steps),
    ]
    if args.max_train_samples is not None:
        result.extend(["--max-train-samples", str(args.max_train_samples)])
    if args.max_eval_samples is not None:
        result.extend(["--max-eval-samples", str(args.max_eval_samples)])
    if args.length_normalize_action_score:
        result.append("--length-normalize-action-score")
    return result


def main() -> None:
    args = parse_args()
    for path in (args.sft_adapter, args.train_jsonl, args.val_jsonl):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    if occupied := gpu_processes(args.gpu):
        raise RuntimeError(f"refusing occupied GPU {args.gpu}: {occupied}")
    seed_output = args.output_root / f"seed{args.seed}"
    trainer_command = command(args, seed_output)
    manifest = {
        "schema_version": "active-catalog-vlm-contextual-rl-launch-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "physical_gpu": args.gpu,
        "trainer_command": trainer_command,
        "claim_boundary": "offline multimodal contextual RL",
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
    peak_memory = peak_utilization = samples = 0
    with log_path.open("w", encoding="utf-8", buffering=1) as log, resource_path.open(
        "w", encoding="utf-8", buffering=1
    ) as resources:
        process = subprocess.Popen(
            trainer_command, env=env, stdout=log, stderr=subprocess.STDOUT, text=True
        )
        while process.poll() is None:
            try:
                sample = gpu_snapshot(args.gpu)
                peak_memory = max(peak_memory, int(sample["memory_used_mib"]))
                peak_utilization = max(peak_utilization, int(sample["utilization_percent"]))
            except (subprocess.SubprocessError, ValueError) as error:
                sample = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "error": str(error)}
            samples += 1
            resources.write(json.dumps(sample, separators=(",", ":")) + "\n")
            write_json(state_path, {
                "status": "running", "pid": process.pid, "seed": args.seed,
                "physical_gpu": args.gpu, "started_utc": started,
                "last_resource_sample": sample,
                "last_training_event": last_history(seed_output / "history.jsonl"),
                "peak_memory_used_mib": peak_memory,
                "peak_utilization_percent": peak_utilization,
            })
            time.sleep(args.monitor_interval)
        returncode = process.wait()
    result = {
        "status": "completed" if returncode == 0 else "failed",
        "returncode": returncode, "pid": process.pid, "seed": args.seed,
        "physical_gpu": args.gpu, "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "peak_memory_used_mib": peak_memory,
        "peak_utilization_percent": peak_utilization,
        "resource_samples": samples, "output": str(seed_output.resolve()),
        "log": str(log_path.resolve()), "resource_history": str(resource_path.resolve()),
    }
    write_json(args.output_root / "process_result.json", result)
    write_json(state_path, result)
    if returncode:
        raise SystemExit(returncode)


if __name__ == "__main__":
    main()
