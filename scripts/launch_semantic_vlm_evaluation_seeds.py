#!/usr/bin/env python3
"""Evaluate completed visual-policy seeds on at most two free GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from scripts.aggregate_semantic_vlm_seeds import aggregate
from scripts.bootstrap_semantic_vlm_results import paired_bootstrap
from scripts.launch_semantic_vlm_sft_seeds import assignments, gpu_processes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("training_root", type=Path)
    parser.add_argument("validation_jsonl", type=Path)
    parser.add_argument("--rollout-jsonl", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--gpu", type=int, action="append", required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _run_worker(args: argparse.Namespace, gpu: int, seeds: list[int]) -> list[dict[str, Any]]:
    results = []
    for seed in seeds:
        adapter = args.training_root / f"seed{seed}" / "final"
        if not (adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"missing final adapter for seed {seed}: {adapter}")
        seed_output = args.output_root / f"seed{seed}"
        output = seed_output / "static"
        log_path = args.output_root / f"seed{seed}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env.setdefault("PYTHONPATH", "src:.")
        command = [
            args.python,
            "scripts/evaluate_semantic_vlm_actions.py",
            str(args.model),
            str(adapter),
            str(args.validation_jsonl),
            str(output),
            "--seed",
            str(seed),
        ]
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if completed.returncode == 0 and args.rollout_jsonl is not None:
                rollout_command = [
                    args.python,
                    "scripts/evaluate_semantic_vlm_rollouts.py",
                    str(args.model),
                    str(adapter),
                    str(args.rollout_jsonl),
                    str(seed_output / "rollout"),
                    "--seed",
                    str(seed),
                ]
                completed = subprocess.run(
                    rollout_command,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
        result = {
            "seed": seed,
            "gpu": gpu,
            "returncode": completed.returncode,
            "adapter": str(adapter.resolve()),
            "output": str(output.resolve()),
            "log": str(log_path.resolve()),
        }
        results.append(result)
        if completed.returncode:
            break
    return results


def main() -> None:
    args = parse_args()
    gpus = list(dict.fromkeys(args.gpu))
    seeds = list(dict.fromkeys(args.seed))
    if not gpus or len(gpus) > 2:
        raise ValueError("evaluation permits one or two GPUs only")
    occupied = {gpu: gpu_processes(gpu) for gpu in gpus}
    occupied = {gpu: rows for gpu, rows in occupied.items() if rows}
    if occupied:
        raise RuntimeError(f"refusing occupied GPUs: {occupied}")
    schedule = assignments(seeds, gpus)
    manifest = {
        "schema_version": "semantic-vlm-evaluation-launch-v1",
        "model": str(args.model.resolve()),
        "training_root": str(args.training_root.resolve()),
        "validation_jsonl": str(args.validation_jsonl.resolve()),
        "rollout_jsonl": str(args.rollout_jsonl.resolve())
        if args.rollout_jsonl is not None
        else None,
        "schedule": [{"seed": seed, "gpu": gpu} for seed, gpu in schedule],
        "test_assets_read": False,
    }
    print(json.dumps(manifest, indent=2), flush=True)
    if args.dry_run:
        return
    args.output_root.mkdir(parents=True, exist_ok=False)
    (args.output_root / "launch_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    queues = {gpu: [seed for seed, assigned in schedule if assigned == gpu] for gpu in gpus}
    results = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {
            executor.submit(_run_worker, args, gpu, queue): gpu
            for gpu, queue in queues.items()
            if queue
        }
        for future in as_completed(futures):
            results.extend(future.result())
    results.sort(key=lambda row: seeds.index(int(row["seed"])))
    (args.output_root / "process_results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    if any(result["returncode"] for result in results) or len(results) != len(seeds):
        raise SystemExit(1)
    summaries = [
        json.loads(
            (args.output_root / f"seed{seed}" / "static" / "summary.json").read_text()
        )
        for seed in seeds
    ]
    aggregate_result = aggregate(summaries, seeds)
    (args.output_root / "three_seed_aggregate.json").write_text(
        json.dumps(aggregate_result, indent=2) + "\n", encoding="utf-8"
    )
    static_bootstrap = paired_bootstrap(
        {
            seed: args.output_root / f"seed{seed}" / "static" / "predictions.jsonl"
            for seed in seeds
        },
        repetitions=args.bootstrap_repetitions,
        bootstrap_seed=20260716,
    )
    (args.output_root / "three_seed_static_paired_bootstrap.json").write_text(
        json.dumps(static_bootstrap, indent=2) + "\n", encoding="utf-8"
    )
    if args.rollout_jsonl is not None:
        rollout_summaries = [
            json.loads(
                (args.output_root / f"seed{seed}" / "rollout" / "summary.json").read_text()
            )
            for seed in seeds
        ]
        rollout_aggregate = aggregate(rollout_summaries, seeds)
        (args.output_root / "three_seed_rollout_aggregate.json").write_text(
            json.dumps(rollout_aggregate, indent=2) + "\n", encoding="utf-8"
        )
        rollout_bootstrap = paired_bootstrap(
            {
                seed: args.output_root / f"seed{seed}" / "rollout" / "traces.jsonl"
                for seed in seeds
            },
            repetitions=args.bootstrap_repetitions,
            bootstrap_seed=20260716,
        )
        (args.output_root / "three_seed_rollout_paired_bootstrap.json").write_text(
            json.dumps(rollout_bootstrap, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
