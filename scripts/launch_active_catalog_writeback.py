#!/usr/bin/env python3
"""Launch executable map writeback with GPU ownership and resource logging."""

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
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("episodes", type=Path)
    parser.add_argument("rollouts", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--simplify-tolerance", type=float, default=0.0)
    parser.add_argument("--min-delta-component-pixels", type=int, default=0)
    parser.add_argument("--delta-margin", type=float, default=0.0)
    parser.add_argument("--confidence-floor", type=float, default=0.0)
    parser.add_argument(
        "--evidence-fusion",
        choices=("confidence_weighted", "max_confidence"),
        default="confidence_weighted",
    )
    parser.add_argument("--protocol-name", default="sn7-active-catalog-vector-writeback-v1")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--frozen-test", action="store_true")
    parser.add_argument("--asset-root-map", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--monitor-interval", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def gpu_processes(gpu: int) -> list[str]:
    result = subprocess.run(
        [
            "nvidia-smi", "-i", str(gpu),
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def gpu_snapshot(gpu: int) -> dict[str, Any]:
    result = subprocess.run(
        [
            "nvidia-smi", "-i", str(gpu),
            "--query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    memory, utilization, power, temperature = (
        value.strip() for value in result.stdout.split(",")
    )
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "memory_used_mib": int(memory),
        "utilization_percent": int(utilization),
        "power_watts": float(power),
        "temperature_c": int(temperature),
    }


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def evaluator_command(args: argparse.Namespace, output: Path) -> list[str]:
    command = [
        args.python,
        "scripts/evaluate_agent_map_writeback.py",
        str(args.checkpoint),
        str(args.episodes),
        str(args.rollouts),
        str(output),
        "--device", "cuda:0",
        "--split", getattr(args, "split", "val"),
        "--image-size", str(args.image_size),
        "--threshold", str(args.threshold),
        "--simplify-tolerance", str(args.simplify_tolerance),
        "--min-delta-component-pixels", str(args.min_delta_component_pixels),
        "--delta-margin", str(getattr(args, "delta_margin", 0.0)),
        "--confidence-floor", str(getattr(args, "confidence_floor", 0.0)),
        "--evidence-fusion", str(getattr(args, "evidence_fusion", "confidence_weighted")),
        "--protocol-name", args.protocol_name,
    ]
    for mapping in args.asset_root_map:
        command.extend(["--asset-root-map", mapping])
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if getattr(args, "frozen_test", False):
        command.append("--frozen-test")
    return command


def main() -> None:
    args = parse_args()
    test_assets_read = args.split == "test"
    if test_assets_read:
        if not args.frozen_test:
            raise PermissionError("test writeback requires --frozen-test")
        from activemap.frozen_test import assert_frozen_test_access

        assert_frozen_test_access()
    elif args.frozen_test:
        raise ValueError("--frozen-test is valid only for the test split")
    for path in (args.checkpoint, args.episodes, args.rollouts):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    if args.monitor_interval <= 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("invalid monitor interval or limit")
    occupied = gpu_processes(args.gpu)
    if occupied:
        raise RuntimeError(f"refusing occupied GPU {args.gpu}: {occupied}")
    evaluation_output = args.output_root / "evaluation"
    command = evaluator_command(args, evaluation_output)
    manifest = {
        "schema_version": "active-catalog-writeback-launch-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "physical_gpu": args.gpu,
        "command": command,
        "protocol_name": args.protocol_name,
        "split": args.split,
        "test_assets_read": test_assets_read,
    }
    print(json.dumps(manifest, indent=2), flush=True)
    if args.dry_run:
        return

    args.output_root.mkdir(parents=True)
    write_json(args.output_root / "launch_manifest.json", manifest)
    log_path = args.output_root / "writeback.log"
    resource_path = args.output_root / "resource_history.jsonl"
    state_path = args.output_root / "run_state.json"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env.setdefault("PYTHONPATH", "src:.")
    env["PYTHONUNBUFFERED"] = "1"
    started = datetime.now(timezone.utc).isoformat()
    peak_memory = 0
    peak_utilization = 0
    samples = 0
    with (
        log_path.open("w", encoding="utf-8", buffering=1) as log,
        resource_path.open("w", encoding="utf-8", buffering=1) as resources,
    ):
        process = subprocess.Popen(
            command, env=env, stdout=log, stderr=subprocess.STDOUT, text=True
        )
        while process.poll() is None:
            try:
                sample = gpu_snapshot(args.gpu)
                peak_memory = max(peak_memory, sample["memory_used_mib"])
                peak_utilization = max(peak_utilization, sample["utilization_percent"])
            except (subprocess.SubprocessError, ValueError) as error:
                sample = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "error": str(error),
                }
            samples += 1
            resources.write(json.dumps(sample, separators=(",", ":")) + "\n")
            write_json(
                state_path,
                {
                    "status": "running",
                    "pid": process.pid,
                    "physical_gpu": args.gpu,
                    "started_utc": started,
                    "last_resource_sample": sample,
                    "peak_memory_used_mib": peak_memory,
                    "peak_utilization_percent": peak_utilization,
                },
            )
            time.sleep(args.monitor_interval)
        returncode = process.wait()
    result = {
        "status": "completed" if returncode == 0 else "failed",
        "returncode": returncode,
        "pid": process.pid,
        "physical_gpu": args.gpu,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "peak_memory_used_mib": peak_memory,
        "peak_utilization_percent": peak_utilization,
        "resource_samples": samples,
        "evaluation_output": str(evaluation_output.resolve()),
    }
    write_json(args.output_root / "process_result.json", result)
    write_json(state_path, result)
    if returncode:
        raise SystemExit(returncode)


if __name__ == "__main__":
    main()
