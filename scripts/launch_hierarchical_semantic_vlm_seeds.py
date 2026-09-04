#!/usr/bin/env python3
"""Extract visual states, calibrate CALL/STOP, and evaluate fixed VLM seeds."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.launch_semantic_vlm_sft_seeds import audit_sft_jsonl, validate_data_contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("training_root", type=Path)
    parser.add_argument("train_sft", type=Path)
    parser.add_argument("val_sft", type=Path)
    parser.add_argument("rollout_jsonl", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--gpu", type=int, action="append", required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--feature-batch-size", type=int, default=2)
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
    parser.add_argument("--expected-train-records", type=int)
    parser.add_argument("--expected-val-records", type=int)
    parser.add_argument("--expected-train-use-tool", type=int)
    parser.add_argument("--expected-val-use-tool", type=int)
    parser.add_argument("--expected-rollout-records", type=int)
    parser.add_argument("--expected-rollout-pre", type=int)
    parser.add_argument("--expected-rollout-post", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def assignments(seeds: list[int], gpus: list[int]) -> list[tuple[int, int]]:
    return [(seed, gpus[index % len(gpus)]) for index, seed in enumerate(seeds)]


def gpu_processes(gpu: int) -> list[str]:
    completed = subprocess.run(
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
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def validate_rollout_contract(args: argparse.Namespace, audit: dict[str, Any]) -> None:
    checks = {
        "rollout.records": (args.expected_rollout_records, audit["records"]),
        "rollout.PRE_TOOL": (
            args.expected_rollout_pre,
            audit["stage_counts"].get("PRE_TOOL", 0),
        ),
        "rollout.POST_TOOL": (
            args.expected_rollout_post,
            audit["stage_counts"].get("POST_TOOL", 0),
        ),
        "rollout.invalid_action_records": (0, audit["invalid_action_records"]),
    }
    mismatches = {
        name: {"expected": expected, "actual": actual}
        for name, (expected, actual) in checks.items()
        if expected is not None and expected != actual
    }
    if mismatches:
        raise ValueError(f"rollout data contract mismatch: {mismatches}")


def _run_phase(
    command: list[str], log_path: Path, *, env: dict[str, str]
) -> dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command, env=env, stdout=log, stderr=subprocess.STDOUT, text=True
        )
    return {
        "command": command,
        "returncode": completed.returncode,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "log": str(log_path.resolve()),
    }


def _worker(args: argparse.Namespace, gpu: int, seeds: list[int]) -> list[dict[str, Any]]:
    results = []
    for seed in seeds:
        root = args.output_root / f"seed{seed}"
        root.mkdir(parents=True, exist_ok=False)
        adapter = args.training_root / f"seed{seed}" / "final"
        if not (adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"missing adapter: {adapter}")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env.setdefault("PYTHONPATH", "src:.")
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        phases = [
            (
                "extract_train",
                [
                    args.python,
                    "scripts/extract_semantic_vlm_gate_features.py",
                    str(args.model),
                    str(adapter),
                    str(args.train_sft),
                    str(root / "features" / "train"),
                    "--device",
                    "cuda",
                    "--batch-size",
                    str(args.feature_batch_size),
                    "--pooling",
                    args.pooling,
                ],
            ),
            (
                "extract_val",
                [
                    args.python,
                    "scripts/extract_semantic_vlm_gate_features.py",
                    str(args.model),
                    str(adapter),
                    str(args.val_sft),
                    str(root / "features" / "val"),
                    "--device",
                    "cuda",
                    "--batch-size",
                    str(args.feature_batch_size),
                    "--pooling",
                    args.pooling,
                ],
            ),
            (
                "train_gate",
                [
                    args.python,
                    "scripts/train_visual_tool_gate.py",
                    str(root / "features" / "train"),
                    str(root / "features" / "val"),
                    str(root / "gate"),
                    "--seed",
                    str(seed),
                    "--selection-objective",
                    args.gate_selection_objective,
                    "--fit-weighting",
                    args.gate_fit_weighting,
                ],
            ),
            (
                "evaluate",
                [
                    args.python,
                    "scripts/evaluate_hierarchical_semantic_vlm.py",
                    str(args.model),
                    str(adapter),
                    str(args.rollout_jsonl),
                    str(root / "features" / "val"),
                    str(root / "gate"),
                    str(root / "evaluation"),
                    "--device",
                    "cuda",
                    "--seed",
                    str(seed),
                ],
            ),
        ]
        phase_results = []
        for phase, command in phases:
            result = _run_phase(command, root / f"{phase}.log", env=env)
            result["phase"] = phase
            phase_results.append(result)
            if result["returncode"]:
                break
        payload = {
            "seed": seed,
            "gpu": gpu,
            "phases": phase_results,
            "returncode": next(
                (phase["returncode"] for phase in phase_results if phase["returncode"]), 0
            ),
        }
        (root / "process_result.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        results.append(payload)
        if payload["returncode"]:
            break
    return results


def main() -> None:
    args = parse_args()
    gpus = list(dict.fromkeys(args.gpu))
    seeds = list(dict.fromkeys(args.seed))
    if not seeds or not gpus or len(gpus) > 2:
        raise ValueError("fixed seeds and one or two GPUs are required")
    if args.feature_batch_size <= 0:
        raise ValueError("feature-batch-size must be positive")
    for path in (args.model, args.train_sft, args.val_sft, args.rollout_jsonl):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    audits = {
        "train": audit_sft_jsonl(args.train_sft),
        "val": audit_sft_jsonl(args.val_sft),
        "rollout": audit_sft_jsonl(args.rollout_jsonl),
    }
    validate_data_contract(args, audits)
    validate_rollout_contract(args, audits["rollout"])
    schedule = assignments(seeds, gpus)
    manifest = {
        "schema_version": "hierarchical-semantic-vlm-launch-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model.resolve()),
        "training_root": str(args.training_root.resolve()),
        "train_sft": str(args.train_sft.resolve()),
        "val_sft": str(args.val_sft.resolve()),
        "rollout_jsonl": str(args.rollout_jsonl.resolve()),
        "feature_batch_size": args.feature_batch_size,
        "pooling": args.pooling,
        "gate_selection_objective": args.gate_selection_objective,
        "data_audit": audits,
        "schedule": [{"seed": seed, "gpu": gpu} for seed, gpu in schedule],
        "test_assets_read": False,
    }
    print(json.dumps(manifest, indent=2), flush=True)
    if args.dry_run:
        return
    occupied = {gpu: gpu_processes(gpu) for gpu in gpus}
    occupied = {gpu: rows for gpu, rows in occupied.items() if rows}
    if occupied:
        raise RuntimeError(f"refusing occupied GPUs: {occupied}")
    args.output_root.mkdir(parents=True, exist_ok=False)
    (args.output_root / "launch_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    queues = {gpu: [seed for seed, assigned in schedule if assigned == gpu] for gpu in gpus}
    results = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {
            executor.submit(_worker, args, gpu, queue): gpu
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

    from scripts.aggregate_hierarchical_semantic_vlm_seeds import aggregate

    summaries = [
        json.loads(
            (args.output_root / f"seed{seed}" / "evaluation" / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        for seed in seeds
    ]
    aggregate_payload = aggregate(summaries, seeds)
    aggregate_payload["sources"] = [
        str(
            (
                args.output_root / f"seed{seed}" / "evaluation" / "summary.json"
            ).resolve()
        )
        for seed in seeds
    ]
    (args.output_root / "three_seed_aggregate.json").write_text(
        json.dumps(aggregate_payload, indent=2) + "\n", encoding="utf-8"
    )
    if len(seeds) >= 2:
        from scripts.bootstrap_semantic_vlm_results import paired_bootstrap

        bootstrap = paired_bootstrap(
            {
                seed: args.output_root / f"seed{seed}" / "evaluation" / "traces.jsonl"
                for seed in seeds
            },
            repetitions=2000,
            bootstrap_seed=20260716,
        )
        bootstrap["comparison"] = "hierarchical-policy-versus-direct-vlm"
        (args.output_root / "task_paired_bootstrap.json").write_text(
            json.dumps(bootstrap, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(aggregate_payload, indent=2))


if __name__ == "__main__":
    main()
