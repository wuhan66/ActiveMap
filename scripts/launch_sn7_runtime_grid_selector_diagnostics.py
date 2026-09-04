#!/usr/bin/env python3
"""Run validation-only selector diagnostics on the matched online runtime grid."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

BASE_CONFIG = Path("configs/selector/sn7_online_runtime_grid_two_stage_v1.yaml")
VARIANTS: dict[str, dict[str, Any]] = {
    "context_rank_balanced": {
        "candidate_value_head": False,
        "candidate_decision_mode": "rank",
        "terminal_gate_mode": "context",
        "target_sampling_power": 1.0,
        "acquire_loss_weight": 1.0,
        "utility_regression_weight": 0.0,
        "candidate_value_sign_weight": 0.0,
        "candidate_value_positive_weight": 1.0,
        "candidate_utility_positive_weight": 1.0,
        "context_gate_loss_weight": 1.0,
        "gate_utility_weight": 1.0,
    },
    "value_rank_pos16": {
        "candidate_value_head": True,
        "candidate_decision_mode": "rank",
        "terminal_gate_mode": "value",
        "target_sampling_power": 1.0,
        "acquire_loss_weight": 1.0,
        "utility_regression_weight": 1.0,
        "candidate_value_sign_weight": 1.0,
        "candidate_value_positive_weight": 16.0,
        "candidate_utility_positive_weight": 8.0,
        "context_gate_loss_weight": 0.0,
        "gate_utility_weight": 0.0,
    },
    "value_rank_pos32": {
        "candidate_value_head": True,
        "candidate_decision_mode": "rank",
        "terminal_gate_mode": "value",
        "target_sampling_power": 1.0,
        "acquire_loss_weight": 1.0,
        "utility_regression_weight": 1.0,
        "candidate_value_sign_weight": 1.0,
        "candidate_value_positive_weight": 32.0,
        "candidate_utility_positive_weight": 16.0,
        "context_gate_loss_weight": 0.0,
        "gate_utility_weight": 0.0,
    },
}


def gpu_processes(gpu: int) -> list[str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def prepare_job(
    repo: Path,
    output_root: Path,
    samples: Path,
    variant_name: str,
    variant: dict[str, Any],
    seed: int,
    gpu: int,
) -> dict[str, Any]:
    payload = yaml.safe_load((repo / BASE_CONFIG).read_text(encoding="utf-8"))
    run_dir = output_root / f"{variant_name}_seed{seed}"
    payload["seed"] = seed
    payload["data"]["samples"] = str(samples.resolve())
    payload["model"].update(
        {
            "candidate_value_head": variant["candidate_value_head"],
            "candidate_decision_mode": variant["candidate_decision_mode"],
            "terminal_gate_mode": variant["terminal_gate_mode"],
        }
    )
    payload["training"].update(
        {
            key: value
            for key, value in variant.items()
            if key
            not in {
                "candidate_value_head",
                "candidate_decision_mode",
                "terminal_gate_mode",
            }
        }
    )
    payload["training"]["epochs"] = 60
    payload["training"]["patience"] = 12
    payload["ablation"]["name"] = f"sn7_runtime_grid_{variant_name}_v1"
    payload["output_dir"] = str(run_dir.resolve())
    run_dir.mkdir(parents=True, exist_ok=False)
    config_path = run_dir / "resolved_config.yaml"
    config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    return {
        "variant": variant_name,
        "seed": seed,
        "gpu": gpu,
        "run_dir": str(run_dir),
        "config": str(config_path),
    }


def run_job(python: str, repo: Path, samples: Path, job: dict[str, Any]) -> dict[str, Any]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
    env["PYTHONPATH"] = str(repo / "src")
    run_dir = Path(job["run_dir"])
    train_command = [
        python,
        "-m",
        "activemap.cli",
        "train-selector",
        job["config"],
        "--output",
        str(run_dir),
    ]
    with (run_dir / "train.log").open("x", encoding="utf-8") as log:
        train_result = subprocess.run(
            train_command,
            cwd=repo,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    result = job | {"train_returncode": train_result.returncode}
    checkpoint = run_dir / "best.pt"
    if train_result.returncode != 0 or not checkpoint.is_file():
        return result | {"frontier_returncode": None}
    frontier_command = [
        python,
        str(repo / "scripts/diagnose_selector_stop_frontier.py"),
        str(checkpoint),
        str(samples),
        str(run_dir / "validation_stop_frontier.json"),
        "--split",
        "val",
        "--device",
        "cuda:0",
        "--max-false-call-rate",
        "0.02",
        "--max-harmful-call-fraction",
        "0.20",
        "--min-acquire-recall",
        "0.10",
    ]
    with (run_dir / "frontier.log").open("x", encoding="utf-8") as log:
        frontier_result = subprocess.run(
            frontier_command,
            cwd=repo,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return result | {"frontier_returncode": frontier_result.returncode}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--gpu", type=int, action="append", required=True)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    gpus = list(dict.fromkeys(args.gpu))
    if len(gpus) != len(VARIANTS):
        raise ValueError(f"expected {len(VARIANTS)} distinct GPUs")
    if not args.samples.is_file():
        raise FileNotFoundError(args.samples)
    provenance_candidates = (
        args.samples.with_suffix(".audit.json"),
        Path(f"{args.samples}.summary.json"),
    )
    provenance = next(
        (path for path in provenance_candidates if path.is_file()), None
    )
    if provenance is None:
        raise FileNotFoundError(
            f"missing selector provenance: {provenance_candidates}"
        )
    audit_payload = json.loads(provenance.read_text(encoding="utf-8"))
    if bool(audit_payload.get("test_assets_read", True)):
        raise RuntimeError("runtime-grid diagnostic rejects test provenance")
    occupied = {gpu: gpu_processes(gpu) for gpu in gpus}
    occupied = {gpu: pids for gpu, pids in occupied.items() if pids}
    if occupied:
        raise RuntimeError(f"refusing occupied GPUs: {occupied}")

    plan = [
        {"variant": name, "gpu": gpu, "seed": args.seed}
        for (name, _), gpu in zip(VARIANTS.items(), gpus, strict=True)
    ]
    print(
        json.dumps(
            {
                "jobs": plan,
                "provenance": str(provenance.resolve()),
                "test_assets_read": False,
            },
            indent=2,
        )
    )
    if args.dry_run:
        return
    args.output_root.mkdir(parents=True, exist_ok=False)
    jobs = [
        prepare_job(
            repo,
            args.output_root,
            args.samples,
            name,
            variant,
            args.seed,
            gpu,
        )
        for (name, variant), gpu in zip(VARIANTS.items(), gpus, strict=True)
    ]
    (args.output_root / "launch_manifest.json").write_text(
        json.dumps(
            {
                "jobs": jobs,
                "provenance": str(provenance.resolve()),
                "test_assets_read": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    results = []
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = {
            executor.submit(run_job, args.python, repo, args.samples, job): job
            for job in jobs
        }
        for future in as_completed(futures):
            results.append(future.result())
    (args.output_root / "process_results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    if any(
        row["train_returncode"] != 0 or row["frontier_returncode"] != 0
        for row in results
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
