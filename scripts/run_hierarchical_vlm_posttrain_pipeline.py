#!/usr/bin/env python3
"""Wait for SFT, smoke hidden states, then run hierarchical validation."""

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

from scripts.launch_hierarchical_semantic_vlm_seeds import gpu_processes
from scripts.run_semantic_vlm_posttrain_pipeline import validate_training_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("training_root", type=Path)
    parser.add_argument("train_sft", type=Path)
    parser.add_argument("val_sft", type=Path)
    parser.add_argument("rollout_jsonl", type=Path)
    parser.add_argument("evaluation_root", type=Path)
    parser.add_argument("--gpu", type=int, action="append", required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--pooling", choices=("last", "last_mean"), default="last_mean")
    parser.add_argument(
        "--gate-selection-objective",
        choices=("f0_5", "proxy_utility"),
        default="f0_5",
    )
    parser.add_argument(
        "--gate-fit-weighting",
        choices=("balanced", "utility_magnitude"),
        default="balanced",
    )
    parser.add_argument("--feature-batch-size", type=int, default=2)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-hours", type=float, default=12.0)
    parser.add_argument("--expected-train-records", type=int)
    parser.add_argument("--expected-val-records", type=int)
    parser.add_argument("--expected-train-use-tool", type=int)
    parser.add_argument("--expected-val-use-tool", type=int)
    parser.add_argument("--expected-rollout-records", type=int)
    parser.add_argument("--expected-rollout-pre", type=int)
    parser.add_argument("--expected-rollout-post", type=int)
    return parser.parse_args()


def _write_state(path: Path, **values: Any) -> None:
    payload = {
        "schema_version": "hierarchical-vlm-posttrain-pipeline-v1",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "test_assets_read": False,
        **values,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def hierarchical_command(args: argparse.Namespace) -> list[str]:
    command = [
        args.python,
        "scripts/launch_hierarchical_semantic_vlm_seeds.py",
        str(args.model),
        str(args.training_root),
        str(args.train_sft),
        str(args.val_sft),
        str(args.rollout_jsonl),
        str(args.evaluation_root),
        "--feature-batch-size",
        str(args.feature_batch_size),
        "--pooling",
        args.pooling,
        "--gate-selection-objective",
        args.gate_selection_objective,
        "--gate-fit-weighting",
        args.gate_fit_weighting,
    ]
    for gpu in list(dict.fromkeys(args.gpu)):
        command.extend(["--gpu", str(gpu)])
    for seed in list(dict.fromkeys(args.seed)):
        command.extend(["--seed", str(seed)])
    optional = {
        "--expected-train-records": args.expected_train_records,
        "--expected-val-records": args.expected_val_records,
        "--expected-train-use-tool": args.expected_train_use_tool,
        "--expected-val-use-tool": args.expected_val_use_tool,
        "--expected-rollout-records": args.expected_rollout_records,
        "--expected-rollout-pre": args.expected_rollout_pre,
        "--expected-rollout-post": args.expected_rollout_post,
    }
    for flag, value in optional.items():
        if value is not None:
            command.extend([flag, str(value)])
    return command


def validate_hidden_smoke(summary: dict[str, Any], pooling: str) -> None:
    checks = {
        "sample_count": summary.get("sample_count") == 2,
        "pooling": summary.get("pooling") == pooling,
        "assistant_tokens_seen": summary.get("assistant_tokens_seen") is False,
        "test_assets_read": summary.get("test_assets_read") is False,
        "feature_dim": int(summary.get("feature_dim", 0)) > 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"hidden-state smoke failed checks: {failed}")


def main() -> None:
    args = parse_args()
    seeds = list(dict.fromkeys(args.seed))
    gpus = list(dict.fromkeys(args.gpu))
    if not seeds or not gpus or len(gpus) > 2:
        raise ValueError("fixed seeds and one or two GPUs are required")
    if args.poll_seconds < 10 or args.timeout_hours <= 0 or args.feature_batch_size <= 0:
        raise ValueError("poll, timeout, or feature batch size is invalid")
    for path in (
        args.model,
        args.training_root,
        args.train_sft,
        args.val_sft,
        args.rollout_jsonl,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.evaluation_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.evaluation_root}")

    state_path = args.training_root / "hierarchical_posttrain_pipeline_state.json"
    results_path = args.training_root / "process_results.json"
    deadline = time.monotonic() + args.timeout_hours * 3600
    _write_state(
        state_path,
        status="waiting_for_training",
        seeds=seeds,
        gpus=gpus,
        evaluation_root=str(args.evaluation_root.resolve()),
    )
    poll_count = 0
    while not results_path.is_file():
        if time.monotonic() >= deadline:
            _write_state(state_path, status="timed_out_waiting_for_training")
            raise TimeoutError("training process results did not appear before timeout")
        poll_count += 1
        _write_state(
            state_path,
            status="waiting_for_training",
            seeds=seeds,
            gpus=gpus,
            evaluation_root=str(args.evaluation_root.resolve()),
            poll_count=poll_count,
            training_results_present=False,
        )
        time.sleep(args.poll_seconds)

    results = json.loads(results_path.read_text(encoding="utf-8"))
    validate_training_results(results, seeds, args.training_root)
    occupied = {gpu: gpu_processes(gpu) for gpu in gpus}
    occupied = {gpu: rows for gpu, rows in occupied.items() if rows}
    if occupied:
        _write_state(state_path, status="failed_closed_occupied_gpu", occupied=occupied)
        raise RuntimeError(f"refusing occupied GPUs: {occupied}")

    smoke_root = args.training_root / "posttrain_hidden_state_smoke" / f"seed{seeds[0]}"
    smoke_log = args.training_root / "posttrain_hidden_state_smoke.log"
    adapter = args.training_root / f"seed{seeds[0]}" / "final"
    smoke_command = [
        args.python,
        "scripts/extract_semantic_vlm_gate_features.py",
        str(args.model),
        str(adapter),
        str(args.val_sft),
        str(smoke_root),
        "--device",
        "cuda",
        "--batch-size",
        "2",
        "--pooling",
        args.pooling,
        "--limit",
        "2",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpus[0])
    env.setdefault("PYTHONPATH", "src:.")
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    _write_state(state_path, status="running_hidden_state_smoke", command=smoke_command)
    with smoke_log.open("w", encoding="utf-8", buffering=1) as log:
        completed = subprocess.run(
            smoke_command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode:
        _write_state(
            state_path,
            status="hidden_state_smoke_failed_closed",
            returncode=completed.returncode,
            log=str(smoke_log.resolve()),
        )
        raise SystemExit(completed.returncode)
    smoke_summary = json.loads((smoke_root / "summary.json").read_text(encoding="utf-8"))
    validate_hidden_smoke(smoke_summary, args.pooling)

    command = hierarchical_command(args)
    _write_state(
        state_path,
        status="launching_hierarchical_validation",
        hidden_state_smoke=smoke_summary,
        command=command,
    )
    completed = subprocess.run(command, env=env, text=True)
    if completed.returncode:
        _write_state(
            state_path,
            status="hierarchical_validation_failed_closed",
            returncode=completed.returncode,
        )
        raise SystemExit(completed.returncode)
    aggregate = json.loads(
        (args.evaluation_root / "three_seed_aggregate.json").read_text(encoding="utf-8")
    )
    _write_state(
        state_path,
        status="hierarchical_validation_complete",
        hidden_state_smoke=smoke_summary,
        single_or_all_seed_gate_passed=aggregate["all_seed_gates_passed"],
        expand_to_three_seeds=bool(
            len(seeds) == 1 and aggregate["all_seed_gates_passed"]
        ),
        rl_ready=False,
        frozen_test_ready=False,
        paper_claim_ready=False,
    )


if __name__ == "__main__":
    main()
