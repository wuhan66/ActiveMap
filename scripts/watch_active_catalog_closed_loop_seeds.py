#!/usr/bin/env python3
"""Run matched active-catalog evaluations after controlled multi-seed SFT."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("training_root", type=Path)
    parser.add_argument("states", type=Path)
    parser.add_argument("episodes", type=Path)
    parser.add_argument("val_sft", type=Path)
    parser.add_argument("val_evaluation_index", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--gpu", type=int, action="append", required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--checkpoint-step", type=int, default=2000)
    parser.add_argument("--ranker-checkpoint", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout-hours", type=float, default=36.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument("--max-acquisitions", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    return parser.parse_args()


def write_state(path: Path, **values: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema_version": "active-catalog-closed-loop-seed-watcher-v1",
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "test_assets_read": False,
                **values,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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


def evaluation_command(
    args: argparse.Namespace,
    *,
    seed: int,
    gpu: int,
    output: Path,
) -> list[str]:
    adapter = (
        args.training_root
        / f"seed{seed}"
        / "checkpoints"
        / f"checkpoint-{args.checkpoint_step}"
    )
    return [
        args.python,
        "scripts/launch_active_catalog_closed_loop.py",
        str(args.model),
        str(adapter),
        str(args.states),
        str(args.episodes),
        str(args.val_sft),
        str(args.val_evaluation_index),
        str(output),
        "--gpu",
        str(gpu),
        "--seed",
        str(seed),
        "--python",
        args.python,
        "--max-candidates",
        str(args.max_candidates),
        "--max-acquisitions",
        str(args.max_acquisitions),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--bootstrap-repetitions",
        str(args.bootstrap_repetitions),
        "--split",
        "val",
        "--policy-mode",
        "gate_ranker",
        "--ranker-checkpoint",
        str(args.ranker_checkpoint),
        "--tool-mode",
        "none",
    ]


def run_evaluation(
    args: argparse.Namespace,
    *,
    seed: int,
    gpu: int,
    deadline: float,
) -> dict[str, Any]:
    output = args.output_root / f"seed{seed}"
    while gpu_processes(gpu):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"GPU {gpu} remained occupied")
        time.sleep(args.poll_seconds)
    command = evaluation_command(args, seed=seed, gpu=gpu, output=output)
    completed = subprocess.run(command, text=True, check=False)
    summary = output / "evaluation" / "summary.json"
    if completed.returncode or not summary.is_file():
        raise RuntimeError(
            f"seed {seed} evaluation failed with return code {completed.returncode}"
        )
    return {
        "seed": seed,
        "gpu": gpu,
        "returncode": completed.returncode,
        "summary": str(summary.resolve()),
        "metrics": json.loads(summary.read_text(encoding="utf-8"))["metrics"],
    }


def main() -> None:
    args = parse_args()
    if len(args.gpu) != len(args.seed) or len(args.gpu) > 2:
        raise ValueError("provide one unique GPU for each of at most two seeds")
    if len(set(args.gpu)) != len(args.gpu) or len(set(args.seed)) != len(args.seed):
        raise ValueError("GPU and seed values must be unique")
    if (
        args.checkpoint_step <= 0
        or args.timeout_hours <= 0
        or args.poll_seconds <= 0
    ):
        raise ValueError("checkpoint, timeout, and poll values must be positive")
    required = [
        args.model,
        args.training_root,
        args.states,
        args.episodes,
        args.val_sft,
        args.val_evaluation_index,
        args.ranker_checkpoint,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    args.output_root.mkdir(parents=True)
    state_path = args.output_root / "watcher_state.json"
    deadline = time.monotonic() + args.timeout_hours * 3600.0
    process_results = args.training_root / "process_results.json"
    write_state(state_path, status="waiting_for_training")
    while not process_results.is_file():
        if time.monotonic() >= deadline:
            write_state(state_path, status="training_timeout")
            raise TimeoutError("multi-seed SFT did not complete before timeout")
        time.sleep(args.poll_seconds)

    training_results = json.loads(process_results.read_text(encoding="utf-8"))
    by_seed = {int(row["seed"]): row for row in training_results}
    for seed in args.seed:
        if seed not in by_seed or int(by_seed[seed]["returncode"]) != 0:
            write_state(
                state_path,
                status="training_failed",
                training_results=training_results,
            )
            raise RuntimeError(f"training seed {seed} did not complete successfully")
        checkpoint = (
            args.training_root
            / f"seed{seed}"
            / "checkpoints"
            / f"checkpoint-{args.checkpoint_step}"
        )
        if not (checkpoint / "adapter_config.json").is_file():
            raise FileNotFoundError(checkpoint / "adapter_config.json")

    schedule = [
        {"seed": seed, "gpu": gpu} for seed, gpu in zip(args.seed, args.gpu, strict=True)
    ]
    write_state(state_path, status="evaluations_running", schedule=schedule)
    results = []
    with ThreadPoolExecutor(max_workers=len(schedule)) as executor:
        futures = {
            executor.submit(
                run_evaluation,
                args,
                seed=item["seed"],
                gpu=item["gpu"],
                deadline=deadline,
            ): item
            for item in schedule
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: args.seed.index(int(row["seed"])))
    (args.output_root / "results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    write_state(state_path, status="complete", schedule=schedule, results=results)


if __name__ == "__main__":
    main()
