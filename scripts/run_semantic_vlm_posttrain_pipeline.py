#!/usr/bin/env python3
"""Wait for three-seed SFT completion, then launch frozen visual-policy evaluation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("training_root", type=Path)
    parser.add_argument("validation_jsonl", type=Path)
    parser.add_argument("rollout_jsonl", type=Path)
    parser.add_argument("evaluation_root", type=Path)
    parser.add_argument("--gpu", type=int, action="append", required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-hours", type=float, default=12.0)
    return parser.parse_args()


def validate_training_results(
    results: list[dict[str, Any]], seeds: list[int], training_root: Path
) -> None:
    by_seed = {int(result["seed"]): result for result in results}
    if sorted(by_seed) != sorted(seeds) or len(by_seed) != len(results):
        raise ValueError("training process results do not match fixed seeds")
    failed = [seed for seed in seeds if int(by_seed[seed]["returncode"]) != 0]
    if failed:
        raise RuntimeError(f"training failed for seeds: {failed}")
    missing = [
        seed
        for seed in seeds
        if not (training_root / f"seed{seed}" / "final" / "adapter_config.json").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing final adapters for seeds: {missing}")


def _write_state(path: Path, **values: Any) -> None:
    state = {
        "schema_version": "semantic-vlm-posttrain-pipeline-v1",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "test_assets_read": False,
        **values,
    }
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.poll_seconds < 10 or args.timeout_hours <= 0.0:
        raise ValueError("poll interval or timeout is invalid")
    seeds = list(dict.fromkeys(args.seed))
    gpus = list(dict.fromkeys(args.gpu))
    if not seeds or not gpus or len(gpus) > 2:
        raise ValueError("fixed seeds and one or two GPUs are required")
    state_path = args.training_root / "posttrain_pipeline_state.json"
    deadline = time.monotonic() + args.timeout_hours * 3600.0
    results_path = args.training_root / "process_results.json"
    _write_state(
        state_path,
        status="waiting_for_training",
        seeds=seeds,
        gpus=gpus,
        evaluation_root=str(args.evaluation_root.resolve()),
    )
    while not results_path.is_file():
        if time.monotonic() >= deadline:
            _write_state(state_path, status="timed_out_waiting_for_training")
            raise TimeoutError("training process results did not appear before timeout")
        time.sleep(args.poll_seconds)
    results = json.loads(results_path.read_text(encoding="utf-8"))
    validate_training_results(results, seeds, args.training_root)
    _write_state(state_path, status="launching_evaluation", seeds=seeds, gpus=gpus)
    command = [
        args.python,
        "scripts/launch_semantic_vlm_evaluation_seeds.py",
        str(args.model),
        str(args.training_root),
        str(args.validation_jsonl),
        str(args.evaluation_root),
        "--rollout-jsonl",
        str(args.rollout_jsonl),
    ]
    for gpu in gpus:
        command.extend(["--gpu", str(gpu)])
    for seed in seeds:
        command.extend(["--seed", str(seed)])
    completed = subprocess.run(command, text=True)
    if completed.returncode:
        _write_state(
            state_path,
            status="evaluation_failed_closed",
            evaluation_returncode=completed.returncode,
        )
        raise SystemExit(completed.returncode)
    static = json.loads(
        (args.evaluation_root / "three_seed_aggregate.json").read_text(encoding="utf-8")
    )
    rollout = json.loads(
        (args.evaluation_root / "three_seed_rollout_aggregate.json").read_text(
            encoding="utf-8"
        )
    )
    ready = bool(
        static["all_static_gates_passed"] and rollout["all_static_gates_passed"]
    )
    _write_state(
        state_path,
        status="evaluation_complete",
        static_gate_passed=static["all_static_gates_passed"],
        recurrent_gate_passed=rollout["all_static_gates_passed"],
        vector_writeback_ready=ready,
        rl_ready=False,
        paper_claim_ready=False,
    )


if __name__ == "__main__":
    main()
