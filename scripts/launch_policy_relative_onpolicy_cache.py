#!/usr/bin/env python3
"""Launch deterministic policy-relative branch shards and merge them on success."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("rollout_jsonl", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--gpu", type=int, action="append", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--expected-pairs", type=int, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def shard_command(args: argparse.Namespace, shard_index: int) -> list[str]:
    command = [
        args.python,
        "scripts/cache_policy_relative_vlm_branches.py",
        str(args.model),
        str(args.adapter),
        str(args.rollout_jsonl),
        str(args.output_root / "shards" / f"shard{shard_index}"),
        "--device",
        "cuda",
        "--seed",
        str(args.seed),
        "--expected-pairs",
        str(args.expected_pairs),
        "--num-shards",
        str(len(args.gpu)),
        "--shard-index",
        str(shard_index),
    ]
    if args.resume:
        command.append("--resume")
    return command


def _write_state(path: Path, **values: Any) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "policy-relative-onpolicy-cache-launch-v1",
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
    gpus = list(dict.fromkeys(args.gpu))
    if len(gpus) not in (1, 2) or len(gpus) != len(args.gpu):
        raise ValueError("launcher requires one or two distinct GPUs")
    if args.output_root.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "shards").mkdir(exist_ok=True)
    state_path = args.output_root / "launcher_state.json"

    processes = []
    logs = []
    launch_rows = []
    for shard_index, gpu in enumerate(gpus):
        command = shard_command(args, shard_index)
        log_path = args.output_root / f"shard{shard_index}.log"
        log = log_path.open("a" if args.resume else "w", encoding="utf-8")
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
        )
        processes.append(process)
        logs.append(log)
        launch_rows.append(
            {
                "shard": shard_index,
                "physical_gpu": gpu,
                "pid": process.pid,
                "command": command,
                "log": str(log_path.resolve()),
            }
        )
    _write_state(state_path, status="running", processes=launch_rows)

    while any(process.poll() is None for process in processes):
        _write_state(
            state_path,
            status="running",
            processes=[
                {**row, "returncode": process.poll()}
                for row, process in zip(launch_rows, processes, strict=True)
            ],
        )
        time.sleep(30)
    for log in logs:
        log.close()
    results = [
        {**row, "returncode": process.returncode}
        for row, process in zip(launch_rows, processes, strict=True)
    ]
    (args.output_root / "process_results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    if any(row["returncode"] for row in results):
        _write_state(state_path, status="failed", processes=results)
        raise RuntimeError("one or more on-policy cache shards failed")

    merge_command = [
        args.python,
        "scripts/merge_policy_relative_vlm_branch_shards.py",
        str(args.output_root / "merged"),
        *[
            str(args.output_root / "shards" / f"shard{index}")
            for index in range(len(gpus))
        ],
    ]
    completed = subprocess.run(merge_command, check=False, text=True)
    if completed.returncode:
        _write_state(
            state_path,
            status="merge_failed",
            processes=results,
            merge_command=merge_command,
            merge_returncode=completed.returncode,
        )
        raise RuntimeError("on-policy cache merge failed")
    _write_state(
        state_path,
        status="complete",
        processes=results,
        merged_summary=str((args.output_root / "merged" / "summary.json").resolve()),
    )


if __name__ == "__main__":
    main()
