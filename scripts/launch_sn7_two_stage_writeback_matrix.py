#!/usr/bin/env python3
"""Launch matched executable writebacks for STOP and three selector seeds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

POLICIES = ("always_stop", "edit_utility_s1", "edit_utility_s2", "edit_utility_s3")


def jobs(
    gpus: list[int], policies: tuple[str, ...] = POLICIES
) -> list[dict[str, Any]]:
    if len(gpus) != 4 or len(set(gpus)) != 4:
        raise ValueError("writeback matrix requires four distinct GPUs")
    if len(policies) != 4 or len(set(policies)) != 4:
        raise ValueError("writeback matrix requires four distinct policies")
    return [
        {"policy": policy, "gpu": gpus[index]}
        for index, policy in enumerate(policies)
    ]


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


def run_job(
    row: dict[str, Any],
    *,
    repo: Path,
    python: str,
    checkpoint: Path,
    episodes: Path,
    input_root: Path,
    output_root: Path,
    image_size: int,
    threshold: float,
    delta_margin: float,
    confidence_floor: float,
    protocol_name: str,
    asset_root_maps: list[str],
) -> dict[str, Any]:
    policy = str(row["policy"])
    command = [
        python,
        "scripts/launch_active_catalog_writeback.py",
        str(checkpoint),
        str(episodes),
        str(input_root / f"{policy}.jsonl"),
        str(output_root / policy),
        "--gpu",
        str(row["gpu"]),
        "--python",
        python,
        "--image-size",
        str(image_size),
        "--threshold",
        str(threshold),
        "--delta-margin",
        str(delta_margin),
        "--confidence-floor",
        str(confidence_floor),
        "--protocol-name",
        protocol_name,
        "--split",
        "val",
    ]
    for mapping in asset_root_maps:
        command.extend(["--asset-root-map", mapping])
    log_path = output_root / "launcher_logs" / f"{policy}.log"
    with log_path.open("x", encoding="utf-8") as log:
        result = subprocess.run(
            command, cwd=repo, stdout=log, stderr=subprocess.STDOUT
        )
    return row | {"returncode": result.returncode, "command": command}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("episodes", type=Path)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--gpu", type=int, action="append", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--delta-margin", type=float, default=0.0)
    parser.add_argument("--confidence-floor", type=float, default=0.0)
    parser.add_argument(
        "--protocol-name", default="sn7-two-stage-selector-vector-writeback-v1"
    )
    parser.add_argument("--asset-root-map", action="append", default=[])
    parser.add_argument(
        "--policy",
        action="append",
        default=[],
        help="Override the four input policy names; repeat exactly four times.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    selected_policies = tuple(args.policy) if args.policy else POLICIES
    plan = jobs(args.gpu, selected_policies)
    required = [args.checkpoint, args.episodes]
    required.extend(args.input_root / f"{policy}.jsonl" for policy in selected_policies)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing writeback inputs: {missing}")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    occupied = {row["gpu"]: gpu_processes(row["gpu"]) for row in plan}
    occupied = {gpu: pids for gpu, pids in occupied.items() if pids}
    if occupied:
        raise RuntimeError(f"refusing occupied GPUs: {occupied}")
    manifest = {
        "schema_version": "sn7-two-stage-writeback-matrix-v1",
        "jobs": plan,
        "checkpoint": str(args.checkpoint.resolve()),
        "episodes": str(args.episodes.resolve()),
        "input_root": str(args.input_root.resolve()),
        "image_size": args.image_size,
        "threshold": args.threshold,
        "delta_margin": args.delta_margin,
        "confidence_floor": args.confidence_floor,
        "protocol_name": args.protocol_name,
        "split": "val",
        "test_assets_read": False,
    }
    print(json.dumps(manifest, indent=2), flush=True)
    if args.dry_run:
        return
    args.output_root.mkdir(parents=True)
    (args.output_root / "launcher_logs").mkdir()
    (args.output_root / "launch_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(
                run_job,
                row,
                repo=repo,
                python=args.python,
                checkpoint=args.checkpoint,
                episodes=args.episodes,
                input_root=args.input_root,
                output_root=args.output_root,
                image_size=args.image_size,
                threshold=args.threshold,
                delta_margin=args.delta_margin,
                confidence_floor=args.confidence_floor,
                protocol_name=args.protocol_name,
                asset_root_maps=args.asset_root_map,
            ): row
            for row in plan
        }
        for future in as_completed(futures):
            results.append(future.result())
    (args.output_root / "process_results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    if any(row["returncode"] for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
