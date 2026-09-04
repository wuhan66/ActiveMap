#!/usr/bin/env python3
"""Cache branch advantages and hidden states from completed cross-fit adapters."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.launch_semantic_vlm_sft_seeds import gpu_processes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("folds_root", type=Path)
    parser.add_argument("training_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--gpu", type=int, action="append", required=True)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--feature-batch-size", type=int, default=2)
    parser.add_argument("--pooling", choices=("last", "last_mean"), default="last_mean")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def branch_command(
    args: argparse.Namespace,
    fold: int,
    adapter: Path,
    rollout: Path,
    output: Path,
    expected_pairs: int,
    *,
    resume: bool,
) -> list[str]:
    command = [
        args.python,
        "scripts/cache_policy_relative_vlm_branches.py",
        str(args.model),
        str(adapter),
        str(rollout),
        str(output),
        "--seed",
        str(args.seed),
        "--expected-pairs",
        str(expected_pairs),
        "--checkpoint-every",
        "25",
    ]
    if resume:
        command.append("--resume")
    return command


def feature_command(
    args: argparse.Namespace,
    adapter: Path,
    rollout: Path,
    output: Path,
) -> list[str]:
    return [
        args.python,
        "scripts/extract_semantic_vlm_gate_features.py",
        str(args.model),
        str(adapter),
        str(rollout),
        str(output),
        "--batch-size",
        str(args.feature_batch_size),
        "--pooling",
        args.pooling,
    ]


def _worker(
    args: argparse.Namespace,
    gpu: int,
    specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results = []
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("PYTHONPATH", "src:.")
    env["PYTHONUNBUFFERED"] = "1"
    for spec in specs:
        fold = int(spec["fold"])
        adapter = args.training_root / f"fold{fold}" / f"seed{args.seed}" / "final"
        rollout = args.folds_root / f"fold{fold}" / "holdout_rollout.jsonl"
        branch_output = args.output_root / "branches" / f"fold{fold}"
        feature_output = args.output_root / "features" / f"fold{fold}"
        expected = int(spec["files"]["holdout_rollout"]["unique_examples"])
        fold_result = {"fold": fold, "gpu": gpu, "phases": []}
        for phase, output, command in (
            (
                "branches",
                branch_output,
                branch_command(
                    args,
                    fold,
                    adapter,
                    rollout,
                    branch_output,
                    expected,
                    resume=branch_output.exists()
                    and not (branch_output / "summary.json").is_file(),
                ),
            ),
            (
                "features",
                feature_output,
                feature_command(args, adapter, rollout, feature_output),
            ),
        ):
            summary_path = output / "summary.json"
            if summary_path.is_file():
                fold_result["phases"].append(
                    {"phase": phase, "status": "reused", "summary": str(summary_path.resolve())}
                )
                continue
            active = gpu_processes(gpu)
            if active:
                raise RuntimeError(f"refusing newly occupied GPU {gpu}: {active}")
            log_path = args.output_root / f"fold{fold}_{phase}.log"
            started = datetime.now(timezone.utc).isoformat()
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            phase_result = {
                "phase": phase,
                "status": "complete" if completed.returncode == 0 else "failed",
                "returncode": completed.returncode,
                "started_utc": started,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "command": command,
                "log": str(log_path.resolve()),
            }
            fold_result["phases"].append(phase_result)
            if completed.returncode:
                break
        results.append(fold_result)
        if any(phase.get("returncode", 0) for phase in fold_result["phases"]):
            break
    return results


def main() -> None:
    args = parse_args()
    gpus = list(dict.fromkeys(args.gpu))
    if not gpus or len(gpus) > 2:
        raise ValueError("cross-fit labeling requires one or two GPUs")
    if args.output_root.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    fold_summary_path = args.folds_root / "summary.json"
    fold_summary = json.loads(fold_summary_path.read_text(encoding="utf-8"))
    if fold_summary.get("test_assets_read") is not False:
        raise ValueError("fold manifest violates the frozen-test protocol")
    training_results = json.loads(
        (args.training_root / "process_results.json").read_text(encoding="utf-8")
    )
    specs = sorted(fold_summary["folds"], key=lambda row: int(row["fold"]))
    if len(training_results) != len(specs) or any(row["returncode"] for row in training_results):
        raise ValueError("cross-fit SFT training is incomplete or failed")
    for spec in specs:
        fold = int(spec["fold"])
        adapter = args.training_root / f"fold{fold}" / f"seed{args.seed}" / "final"
        if not (adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"missing fold adapter: {adapter}")
    occupied = {gpu: gpu_processes(gpu) for gpu in gpus}
    occupied = {gpu: rows for gpu, rows in occupied.items() if rows}
    if occupied:
        raise RuntimeError(f"refusing occupied GPUs: {occupied}")
    schedule = [
        {"fold": int(spec["fold"]), "gpu": gpus[index % len(gpus)]}
        for index, spec in enumerate(specs)
    ]
    manifest = {
        "schema_version": "policy-relative-crossfit-labeling-launch-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model.resolve()),
        "folds_root": str(args.folds_root.resolve()),
        "fold_summary_sha256": _sha256(fold_summary_path),
        "training_root": str(args.training_root.resolve()),
        "training_results_sha256": _sha256(args.training_root / "process_results.json"),
        "seed": args.seed,
        "pooling": args.pooling,
        "feature_batch_size": args.feature_batch_size,
        "schedule": schedule,
        "test_assets_read": False,
    }
    print(json.dumps(manifest, indent=2), flush=True)
    if args.dry_run:
        return
    args.output_root.mkdir(parents=True, exist_ok=args.resume)
    manifest_path = args.output_root / "launch_manifest.json"
    if args.resume and manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparable = {key: value for key, value in previous.items() if key != "created_utc"}
        current = {key: value for key, value in manifest.items() if key != "created_utc"}
        if comparable != current:
            raise ValueError("resume labeling manifest differs from the requested protocol")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    queues = {
        gpu: [
            spec
            for spec, assigned in zip(specs, schedule, strict=True)
            if assigned["gpu"] == gpu
        ]
        for gpu in gpus
    }
    results = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {
            executor.submit(_worker, args, gpu, queue): gpu
            for gpu, queue in queues.items()
            if queue
        }
        for future in as_completed(futures):
            results.extend(future.result())
    results.sort(key=lambda row: int(row["fold"]))
    (args.output_root / "process_results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    failures = [
        (row["fold"], phase["phase"])
        for row in results
        for phase in row["phases"]
        if phase.get("returncode", 0)
    ]
    if len(results) != len(specs) or failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
