#!/usr/bin/env python3
"""Launch four matched SN7 writeback safety configurations on distinct GPUs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def _candidate(value: str) -> dict[str, Any]:
    parts = value.split(":")
    if len(parts) not in {3, 4}:
        raise argparse.ArgumentTypeError(
            "candidate must use "
            "LABEL:DELTA_MARGIN:CONFIDENCE_FLOOR[:MIN_COMPONENT_PIXELS]"
        )
    label, raw_margin, raw_floor = parts[:3]
    raw_min_component = parts[3] if len(parts) == 4 else "0"
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789_-"
    if not label or any(character not in allowed for character in label):
        raise argparse.ArgumentTypeError("candidate label must be lowercase filesystem-safe text")
    margin, floor = float(raw_margin), float(raw_floor)
    min_component_pixels = int(raw_min_component)
    if (
        not 0.0 <= margin < 0.5
        or not 0.0 <= floor <= 1.0
        or min_component_pixels < 0
    ):
        raise argparse.ArgumentTypeError("candidate margin/floor is outside the valid range")
    return {
        "label": label,
        "delta_margin": margin,
        "confidence_floor": floor,
        "min_delta_component_pixels": min_component_pixels,
    }


def jobs(candidates: list[dict[str, Any]], gpus: list[int]) -> list[dict[str, Any]]:
    if len(candidates) != 4 or len({row["label"] for row in candidates}) != 4:
        raise ValueError("safety sweep requires four uniquely labeled candidates")
    if len(gpus) != 4 or len(set(gpus)) != 4:
        raise ValueError("safety sweep requires four distinct GPUs")
    return [candidate | {"gpu": gpus[index]} for index, candidate in enumerate(candidates)]


def _gpu_processes(gpu: int) -> list[str]:
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


def _run(
    row: dict[str, Any],
    *,
    repo: Path,
    python: str,
    checkpoint: Path,
    episodes: Path,
    rollouts: Path,
    output_root: Path,
    asset_root_maps: list[str],
) -> dict[str, Any]:
    output = output_root / str(row["label"])
    command = [
        python,
        "scripts/launch_active_catalog_writeback.py",
        str(checkpoint),
        str(episodes),
        str(rollouts),
        str(output),
        "--gpu",
        str(row["gpu"]),
        "--python",
        python,
        "--image-size",
        "512",
        "--threshold",
        "0.5",
        "--delta-margin",
        str(row["delta_margin"]),
        "--confidence-floor",
        str(row["confidence_floor"]),
        "--min-delta-component-pixels",
        str(row["min_delta_component_pixels"]),
        "--protocol-name",
        "sn7-writeback-safety-sweep-v1",
        "--split",
        "val",
    ]
    for mapping in asset_root_maps:
        command.extend(["--asset-root-map", mapping])
    with (output_root / "launcher_logs" / f"{row['label']}.log").open(
        "x", encoding="utf-8"
    ) as log:
        result = subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
    return row | {"returncode": result.returncode, "command": command}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("episodes", type=Path)
    parser.add_argument("rollouts", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--candidate", action="append", type=_candidate, required=True)
    parser.add_argument("--gpu", action="append", type=int, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--asset-root-map", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    plan = jobs(args.candidate, args.gpu)
    for path in (args.checkpoint, args.episodes, args.rollouts):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    occupied = {row["gpu"]: _gpu_processes(row["gpu"]) for row in plan}
    occupied = {gpu: pids for gpu, pids in occupied.items() if pids}
    if occupied:
        raise RuntimeError(f"refusing occupied GPUs: {occupied}")
    manifest = {
        "schema_version": "sn7-writeback-safety-sweep-launch-v1",
        "jobs": plan,
        "checkpoint": str(args.checkpoint.resolve()),
        "episodes": str(args.episodes.resolve()),
        "rollouts": str(args.rollouts.resolve()),
        "selection_split": "validation",
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
                _run,
                row,
                repo=Path(__file__).resolve().parents[1],
                python=args.python,
                checkpoint=args.checkpoint,
                episodes=args.episodes,
                rollouts=args.rollouts,
                output_root=args.output_root,
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
