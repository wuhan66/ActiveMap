#!/usr/bin/env python3
"""Launch recurrent active-catalog evaluation with GPU and provenance logging."""

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
    parser.add_argument("states", type=Path)
    parser.add_argument("episodes", type=Path)
    parser.add_argument("val_sft", type=Path)
    parser.add_argument("val_evaluation_index", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument("--max-acquisitions", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--policy-mode",
        choices=(
            "full_action",
            "react",
            "geommagent_style",
            "sensesearch_style",
            "always_stop",
            "plan_execute",
            "gate_ranker",
            "utility_head_ranker",
            "residual_ranker",
            "hybrid_residual_ranker",
        ),
        default="full_action",
    )
    parser.add_argument("--ranker-checkpoint", type=Path)
    parser.add_argument("--utility-head", type=Path)
    parser.add_argument("--utility-head-summary", type=Path)
    parser.add_argument("--utility-threshold-override", type=float)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--frozen-test", action="store_true")
    parser.add_argument(
        "--tool-mode",
        choices=("none", "forced", "selective", "model"),
        default="none",
    )
    parser.add_argument(
        "--belief-mode",
        choices=("recurrent", "frozen_prior", "identity"),
        default="recurrent",
    )
    parser.add_argument("--tool-belief-checkpoint", type=Path)
    parser.add_argument("--tool-gate", type=Path)
    parser.add_argument("--tool-gate-summary", type=Path)
    parser.add_argument("--tool-artifact-root", type=Path)
    parser.add_argument("--tool-out-size", type=int, default=256)
    parser.add_argument("--asset-root-map", action="append", default=[])
    parser.add_argument("--monitor-interval", type=float, default=5.0)
    parser.add_argument(
        "--allow-shared-gpu",
        action="store_true",
        help="Allow a deliberate second process on an occupied GPU.",
    )
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
        "scripts/evaluate_active_catalog_closed_loop.py",
        str(args.model),
        str(args.adapter),
        str(args.states),
        str(args.episodes),
        str(args.val_sft),
        str(args.val_evaluation_index),
        str(output),
        "--device", "cuda:0",
        "--max-candidates", str(args.max_candidates),
        "--max-acquisitions", str(args.max_acquisitions),
        "--max-new-tokens", str(args.max_new_tokens),
        "--bootstrap-repetitions", str(args.bootstrap_repetitions),
        "--seed", str(args.seed),
        "--split", getattr(args, "split", "val"),
        "--policy-mode", getattr(args, "policy_mode", "full_action"),
        "--num-shards", str(getattr(args, "num_shards", 1)),
        "--shard-index", str(getattr(args, "shard_index", 0)),
    ]
    ranker_checkpoint = getattr(args, "ranker_checkpoint", None)
    if ranker_checkpoint is not None:
        command.extend(["--ranker-checkpoint", str(ranker_checkpoint)])
    utility_head = getattr(args, "utility_head", None)
    utility_head_summary = getattr(args, "utility_head_summary", None)
    if utility_head is not None:
        command.extend(["--utility-head", str(utility_head)])
    if utility_head_summary is not None:
        command.extend(["--utility-head-summary", str(utility_head_summary)])
    utility_threshold_override = getattr(args, "utility_threshold_override", None)
    if utility_threshold_override is not None:
        command.extend(["--utility-threshold-override", str(utility_threshold_override)])
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if getattr(args, "do_sample", False):
        command.extend([
            "--do-sample", "--temperature", str(getattr(args, "temperature", 1.0)),
            "--top-p", str(getattr(args, "top_p", 1.0))
        ])
    command.extend(
        [
            "--tool-mode",
            getattr(args, "tool_mode", "none"),
            "--belief-mode",
            getattr(args, "belief_mode", "recurrent"),
        ]
    )
    if getattr(args, "tool_mode", "none") != "none":
        command.extend([
            "--tool-belief-checkpoint", str(args.tool_belief_checkpoint),
            "--tool-artifact-root", str(args.tool_artifact_root),
            "--tool-out-size", str(args.tool_out_size),
        ])
        for root_map in getattr(args, "asset_root_map", []):
            command.extend(["--asset-root-map", str(root_map)])
    if getattr(args, "tool_mode", "none") == "selective":
        command.extend([
            "--tool-gate", str(args.tool_gate),
            "--tool-gate-summary", str(args.tool_gate_summary),
        ])
    if getattr(args, "frozen_test", False):
        command.append("--frozen-test")
    return command


def main() -> None:
    args = parse_args()
    test_assets_read = args.split == "test"
    if test_assets_read:
        if not args.frozen_test:
            raise PermissionError("test evaluation requires --frozen-test")
        from activemap.frozen_test import assert_frozen_test_access

        assert_frozen_test_access()
    elif args.frozen_test:
        raise ValueError("--frozen-test is valid only for the test split")
    required_paths = [
        args.model,
        args.adapter,
        args.states,
        args.episodes,
        args.val_sft,
        args.val_evaluation_index,
    ]
    if args.policy_mode in {"gate_ranker", "utility_head_ranker", "residual_ranker", "hybrid_residual_ranker"}:
        if args.ranker_checkpoint is None:
            raise ValueError(f"{args.policy_mode} launch requires --ranker-checkpoint")
        required_paths.append(args.ranker_checkpoint)
    elif args.ranker_checkpoint is not None:
        raise ValueError("ranker checkpoint requires a structured ranker policy")
    if args.policy_mode in {"utility_head_ranker", "hybrid_residual_ranker"}:
        if args.utility_head is None or args.utility_head_summary is None:
            raise ValueError("utility_head_ranker launch requires head and summary")
        required_paths.extend([args.utility_head, args.utility_head_summary])
    elif (
        args.utility_head is not None
        or args.utility_head_summary is not None
        or args.utility_threshold_override is not None
    ):
        raise ValueError("utility-head files require utility_head_ranker policy")
    model_tool_policies = {"react", "geommagent_style", "sensesearch_style"}
    if (args.policy_mode in model_tool_policies) != (args.tool_mode == "model"):
        raise ValueError(
            "ReAct/GeoMMAgent/SenseSearch policies require --tool-mode model, "
            "and model tool mode is restricted to those policies"
        )
    if args.belief_mode != "recurrent" and args.tool_mode == "none":
        raise ValueError("belief ablations require tool use")
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    if not (args.adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(args.adapter / "adapter_config.json")
    optional_required = []
    if args.tool_mode != "none":
        optional_required.extend([args.tool_belief_checkpoint, args.tool_artifact_root])
    if args.tool_mode == "selective":
        optional_required.extend([args.tool_gate, args.tool_gate_summary])
    if any(path is None for path in optional_required):
        raise ValueError("selected tool mode lacks required paths")
    for path in optional_required:
        if path is not None and path != args.tool_artifact_root and not path.exists():
            raise FileNotFoundError(path)
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    if args.monitor_interval <= 0 or args.bootstrap_repetitions < 0:
        raise ValueError("invalid monitoring or bootstrap settings")
    occupied = gpu_processes(args.gpu)
    if occupied and not args.allow_shared_gpu:
        raise RuntimeError(f"refusing occupied GPU {args.gpu}: {occupied}")

    evaluation_output = args.output_root / "evaluation"
    command = evaluator_command(args, evaluation_output)
    manifest = {
        "schema_version": "active-catalog-closed-loop-launch-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "physical_gpu": args.gpu,
        "allow_shared_gpu": args.allow_shared_gpu,
        "preexisting_gpu_processes": occupied,
        "seed": args.seed,
        "split": args.split,
        "tool_mode": args.tool_mode,
        "belief_mode": args.belief_mode,
        "command": command,
        "policy_mode": args.policy_mode,
        "ranker_checkpoint": (
            str(args.ranker_checkpoint.resolve())
            if args.ranker_checkpoint is not None
            else None
        ),
        "utility_head": (
            str(args.utility_head.resolve()) if args.utility_head is not None else None
        ),
        "utility_head_summary": (
            str(args.utility_head_summary.resolve())
            if args.utility_head_summary is not None
            else None
        ),
        "utility_threshold_override": args.utility_threshold_override,
        "model_selected_state_transitions": True,
        "oracle_next_state_replay": False,
        "test_assets_read": test_assets_read,
    }
    print(json.dumps(manifest, indent=2), flush=True)
    if args.dry_run:
        return

    args.output_root.mkdir(parents=True)
    write_json(args.output_root / "launch_manifest.json", manifest)
    log_path = args.output_root / "evaluation.log"
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
