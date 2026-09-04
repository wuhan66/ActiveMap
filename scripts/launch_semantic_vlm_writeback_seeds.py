#!/usr/bin/env python3
"""Run frozen-updater vector writeback for promoted visual-policy seeds."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from scripts.aggregate_semantic_vlm_writeback_seeds import aggregate
from scripts.launch_semantic_vlm_sft_seeds import assignments, gpu_processes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("updater_checkpoint", type=Path)
    parser.add_argument("episodes", type=Path)
    parser.add_argument("evaluation_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--gpu", type=int, action="append", required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--simplify-tolerance", type=float, default=0.0)
    parser.add_argument("--min-delta-component-pixels", type=int, default=0)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--asset-root-map", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _worker(args: argparse.Namespace, gpu: int, seeds: list[int]) -> list[dict[str, Any]]:
    results = []
    for seed in seeds:
        trace_path = args.evaluation_root / f"seed{seed}" / "rollout" / "traces.jsonl"
        if not trace_path.is_file():
            raise FileNotFoundError(trace_path)
        seed_root = args.output_root / f"seed{seed}"
        seed_root.mkdir(parents=True)
        converted = seed_root / "writeback_input.jsonl"
        log_path = seed_root / "writeback.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env.setdefault("PYTHONPATH", "src:.")
        commands = [
            [
                args.python,
                "scripts/convert_semantic_vlm_rollouts_for_writeback.py",
                str(trace_path),
                str(converted),
            ],
            [
                args.python,
                "scripts/evaluate_agent_map_writeback.py",
                str(args.updater_checkpoint),
                str(args.episodes),
                str(converted),
                str(seed_root / "writeback"),
                "--device",
                "cuda",
                "--split",
                "val",
                "--image-size",
                str(args.image_size),
                "--threshold",
                str(args.threshold),
                "--simplify-tolerance",
                str(args.simplify_tolerance),
                "--min-delta-component-pixels",
                str(args.min_delta_component_pixels),
            ],
        ]
        for mapping in args.asset_root_map:
            commands[1].extend(["--asset-root-map", mapping])
        returncode = 0
        with log_path.open("w", encoding="utf-8") as log:
            for command in commands:
                completed = subprocess.run(
                    command,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                returncode = completed.returncode
                if returncode:
                    break
        results.append(
            {
                "seed": seed,
                "gpu": gpu,
                "returncode": returncode,
                "trace": str(trace_path.resolve()),
                "output": str(seed_root.resolve()),
                "log": str(log_path.resolve()),
            }
        )
        if returncode:
            break
    return results


def main() -> None:
    args = parse_args()
    gpus = list(dict.fromkeys(args.gpu))
    seeds = list(dict.fromkeys(args.seed))
    if not gpus or len(gpus) > 2 or not seeds:
        raise ValueError("one or two GPUs and fixed seeds are required")
    recurrent_path = args.evaluation_root / "three_seed_rollout_aggregate.json"
    if not recurrent_path.is_file():
        raise FileNotFoundError(recurrent_path)
    recurrent = json.loads(recurrent_path.read_text(encoding="utf-8"))
    if recurrent.get("all_static_gates_passed") is not True:
        raise PermissionError("recurrent visual-policy gate did not pass")
    occupied = {gpu: gpu_processes(gpu) for gpu in gpus}
    occupied = {gpu: rows for gpu, rows in occupied.items() if rows}
    if occupied:
        raise RuntimeError(f"refusing occupied GPUs: {occupied}")
    schedule = assignments(seeds, gpus)
    manifest = {
        "schema_version": "semantic-vlm-writeback-launch-v1",
        "updater_checkpoint": str(args.updater_checkpoint.resolve()),
        "episodes": str(args.episodes.resolve()),
        "evaluation_root": str(args.evaluation_root.resolve()),
        "schedule": [{"seed": seed, "gpu": gpu} for seed, gpu in schedule],
        "image_size": args.image_size,
        "threshold": args.threshold,
        "simplify_tolerance": args.simplify_tolerance,
        "min_delta_component_pixels": args.min_delta_component_pixels,
        "asset_root_maps": args.asset_root_map,
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
    process_results = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {
            executor.submit(_worker, args, gpu, queue): gpu
            for gpu, queue in queues.items()
            if queue
        }
        for future in as_completed(futures):
            process_results.extend(future.result())
    process_results.sort(key=lambda row: seeds.index(int(row["seed"])))
    (args.output_root / "process_results.json").write_text(
        json.dumps(process_results, indent=2) + "\n", encoding="utf-8"
    )
    if any(row["returncode"] for row in process_results) or len(process_results) != len(seeds):
        raise SystemExit(1)
    paths = {
        seed: args.output_root / f"seed{seed}" / "writeback" / "writeback.jsonl"
        for seed in seeds
    }
    aggregate_result = aggregate(
        paths,
        repetitions=args.bootstrap_repetitions,
        bootstrap_seed=20260716,
    )
    (args.output_root / "three_seed_writeback_aggregate.json").write_text(
        json.dumps(aggregate_result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
