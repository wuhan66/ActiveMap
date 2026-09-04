#!/usr/bin/env python3
"""Train task-cross-fitted semantic VLM adapters on at most two GPUs."""

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
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--gpu", type=int, action="append", required=True)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def fold_command(
    args: argparse.Namespace,
    spec: dict[str, Any],
    gpu: int,
    output_dir: Path,
) -> list[str]:
    fold = int(spec["fold"])
    fold_root = args.folds_root / f"fold{fold}"
    train = spec["files"]["train_sft"]
    val = spec["files"]["inner_val_sft"]
    return [
        args.python,
        "scripts/launch_semantic_vlm_sft_seeds.py",
        str(args.model),
        str(fold_root / "train_sft.jsonl"),
        str(fold_root / "inner_val_sft.jsonl"),
        str(output_dir),
        "--gpu",
        str(gpu),
        "--seed",
        str(args.seed),
        "--epochs",
        str(args.epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--batch-size",
        str(args.batch_size),
        "--gradient-accumulation",
        str(args.gradient_accumulation),
        "--eval-steps",
        str(args.eval_steps),
        "--save-steps",
        str(args.save_steps),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--expected-train-records",
        str(train["records"]),
        "--expected-val-records",
        str(val["records"]),
        "--expected-train-use-tool",
        str(train["pre_tool_positive_records"]),
        "--expected-val-use-tool",
        str(val["pre_tool_positive_records"]),
    ]


def _validate_folds(args: argparse.Namespace, summary: dict[str, Any]) -> None:
    if summary.get("test_assets_read") is not False:
        raise ValueError("cross-fit folds do not preserve the frozen-test protocol")
    if summary.get("schema_version") != "policy-relative-crossfit-folds-v1":
        raise ValueError("unexpected cross-fit fold schema")
    for spec in summary["folds"]:
        fold_root = args.folds_root / f"fold{spec['fold']}"
        for name in ("train_sft", "inner_val_sft", "holdout_rollout"):
            path = fold_root / f"{name}.jsonl"
            if not path.is_file():
                raise FileNotFoundError(path)
            if _sha256(path) != spec["files"][name]["sha256"]:
                raise ValueError(f"fold {spec['fold']} {name} hash mismatch")


def _worker(
    args: argparse.Namespace,
    gpu: int,
    specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results = []
    for spec in specs:
        fold = int(spec["fold"])
        active = gpu_processes(gpu)
        if active:
            raise RuntimeError(f"refusing newly occupied GPU {gpu}: {active}")
        output_dir = args.output_root / f"fold{fold}"
        command = fold_command(args, spec, gpu, output_dir)
        log_path = args.output_root / f"fold{fold}_launcher.log"
        env = os.environ.copy()
        env.setdefault("PYTHONPATH", "src:.")
        env["PYTHONUNBUFFERED"] = "1"
        started = datetime.now(timezone.utc).isoformat()
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        result = {
            "fold": fold,
            "gpu": gpu,
            "seed": args.seed,
            "returncode": completed.returncode,
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "command": command,
            "output_dir": str(output_dir.resolve()),
            "log": str(log_path.resolve()),
        }
        results.append(result)
        if completed.returncode:
            break
    return results


def main() -> None:
    args = parse_args()
    gpus = list(dict.fromkeys(args.gpu))
    if not gpus or len(gpus) > 2:
        raise ValueError("cross-fit training requires one or two GPUs")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    summary_path = args.folds_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _validate_folds(args, summary)
    specs = sorted(summary["folds"], key=lambda row: int(row["fold"]))
    occupied = {gpu: gpu_processes(gpu) for gpu in gpus}
    occupied = {gpu: rows for gpu, rows in occupied.items() if rows}
    if occupied:
        raise RuntimeError(f"refusing occupied GPUs: {occupied}")
    schedule = [
        {"fold": int(spec["fold"]), "gpu": gpus[index % len(gpus)]}
        for index, spec in enumerate(specs)
    ]
    manifest = {
        "schema_version": "policy-relative-crossfit-sft-launch-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model.resolve()),
        "folds_root": str(args.folds_root.resolve()),
        "fold_summary_sha256": _sha256(summary_path),
        "seed": args.seed,
        "schedule": schedule,
        "protocol": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "eval_steps": args.eval_steps,
            "save_steps": args.save_steps,
            "early_stopping_patience": args.early_stopping_patience,
        },
        "test_assets_read": False,
    }
    print(json.dumps(manifest, indent=2), flush=True)
    if args.dry_run:
        return
    args.output_root.mkdir(parents=True)
    (args.output_root / "launch_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
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
    if len(results) != len(specs) or any(result["returncode"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
