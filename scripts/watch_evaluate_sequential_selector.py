#!/usr/bin/env python3
"""Evaluate one sequential selector after its controlled SFT run succeeds."""

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

from scripts.launch_semantic_vlm_sft_seeds import gpu_processes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("training_root", type=Path)
    parser.add_argument("val_jsonl", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout-hours", type=float, default=4.0)
    return parser.parse_args()


def evaluation_command(args: argparse.Namespace, adapter: Path) -> list[str]:
    return [
        args.python,
        "scripts/evaluate_sequential_selector.py",
        str(args.model),
        str(adapter),
        str(args.val_jsonl),
        str(args.output_root / f"seed{args.seed}"),
        "--device",
        "cuda",
        "--seed",
        str(args.seed),
        "--expected-records",
        "414",
        "--bootstrap-repetitions",
        "2000",
    ]


def _write_state(path: Path, **values: Any) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "sequential-selector-eval-watcher-v1",
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "test_assets_read": False,
                **values,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    args.output_root.mkdir(parents=True)
    state = args.output_root / "watcher_state.json"
    results_path = args.training_root / "process_results.json"
    deadline = time.monotonic() + args.timeout_hours * 3600.0
    _write_state(state, status="waiting_for_training")
    while not results_path.is_file():
        if time.monotonic() >= deadline:
            _write_state(state, status="training_timeout")
            raise TimeoutError("selector SFT did not complete before timeout")
        time.sleep(30)
    results = json.loads(results_path.read_text(encoding="utf-8"))
    matches = [row for row in results if int(row["seed"]) == args.seed]
    if len(matches) != 1 or matches[0]["returncode"] != 0:
        _write_state(state, status="training_failed", process_results=results)
        raise RuntimeError("selector SFT did not complete successfully")
    adapter = args.training_root / f"seed{args.seed}" / "final"
    if not (adapter / "adapter_model.safetensors").is_file():
        _write_state(state, status="missing_final_adapter")
        raise FileNotFoundError(adapter / "adapter_model.safetensors")
    _write_state(state, status="waiting_for_gpu", adapter=str(adapter.resolve()))
    while gpu_processes(args.gpu):
        if time.monotonic() >= deadline:
            _write_state(state, status="gpu_timeout")
            raise TimeoutError("selector evaluation GPU remained occupied")
        time.sleep(15)
    command = evaluation_command(args, adapter)
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu)}
    environment.setdefault("PYTHONPATH", "src:.")
    log_path = args.output_root / "evaluation.log"
    _write_state(state, status="evaluation_running", command=command, gpu=args.gpu)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    summary = args.output_root / f"seed{args.seed}" / "summary.json"
    if completed.returncode or not summary.is_file():
        _write_state(
            state,
            status="evaluation_failed",
            returncode=completed.returncode,
            log=str(log_path.resolve()),
        )
        raise RuntimeError("sequential selector evaluation failed")
    result = json.loads(summary.read_text(encoding="utf-8"))
    _write_state(
        state,
        status="complete",
        summary=str(summary.resolve()),
        promotion_gate=result["promotion_gate"],
        joint_sft_ready=bool(result["promotion_gate"]["passed"]),
        rl_ready=False,
        frozen_test_ready=False,
    )


if __name__ == "__main__":
    main()
