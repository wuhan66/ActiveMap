#!/usr/bin/env python3
"""Launch paired SN7 generic/edit-conditioned selector seeds on up to four GPUs."""

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

METHODS = {
    "generic": Path("configs/selector/sn7_evidence_generic_v1_server.yaml"),
    "edit_conditioned": Path(
        "configs/selector/sn7_evidence_edit_conditioned_v1_server.yaml"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--gpu", type=int, action="append", required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def jobs(seeds: list[int], gpus: list[int]) -> list[dict[str, Any]]:
    planned = [
        {"method": method, "seed": seed}
        for seed in seeds
        for method in METHODS
    ]
    if len(planned) > len(gpus):
        raise ValueError("one distinct GPU is required per concurrent job")
    return [row | {"gpu": gpus[index]} for index, row in enumerate(planned)]


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


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def prepare_job(
    repo: Path, output_root: Path, row: dict[str, Any]
) -> dict[str, Any]:
    template = repo / METHODS[str(row["method"])]
    payload = yaml.safe_load(template.read_text(encoding="utf-8"))
    run_dir = output_root / f"{row['method']}_seed{row['seed']}"
    payload["seed"] = int(row["seed"])
    payload["output_dir"] = str(run_dir.resolve())
    run_dir.mkdir(parents=True, exist_ok=False)
    config = run_dir / "resolved_config.yaml"
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return row | {"run_dir": str(run_dir), "config": str(config)}


def run_job(python: str, repo: Path, row: dict[str, Any]) -> dict[str, Any]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(row["gpu"])
    env["PYTHONPATH"] = str(repo / "src")
    command = [
        python,
        "-m",
        "activemap.cli",
        "train-selector",
        str(row["config"]),
        "--output",
        str(row["run_dir"]),
    ]
    log_path = Path(str(row["run_dir"])) / "train.log"
    with log_path.open("x", encoding="utf-8") as log:
        result = subprocess.run(
            command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT
        )
    return row | {"returncode": result.returncode, "command": command}


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    gpus = list(dict.fromkeys(args.gpu))
    seeds = list(dict.fromkeys(args.seed))
    if not 1 <= len(gpus) <= 4:
        raise ValueError("launcher permits one to four GPUs")
    plan = jobs(seeds, gpus)
    data = Path(
        "/home/wh/ActiveMap/processed/sn7_v1/agent/sequential_selector_v1/"
        "selector_states_train_val.jsonl"
    )
    required = [
        data,
        data.with_suffix(".audit.json"),
        data.with_suffix(".utility_audit.json"),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing audited selector inputs: {missing}")
    occupied = {gpu: gpu_processes(gpu) for gpu in gpus}
    occupied = {gpu: pids for gpu, pids in occupied.items() if pids}
    if occupied:
        raise RuntimeError(f"refusing occupied GPUs: {occupied}")
    manifest = {
        "schema_version": "sn7-selector-baseline-launch-v1",
        "jobs": plan,
        "data": str(data),
        "test_assets_read": False,
    }
    print(json.dumps(manifest, indent=2), flush=True)
    if args.dry_run:
        return
    args.output_root.mkdir(parents=True, exist_ok=False)
    prepared = [prepare_job(repo, args.output_root, row) for row in plan]
    write_json(args.output_root / "launch_manifest.json", manifest | {"jobs": prepared})
    results = []
    with ThreadPoolExecutor(max_workers=len(prepared)) as executor:
        futures = {
            executor.submit(run_job, args.python, repo, row): row for row in prepared
        }
        for future in as_completed(futures):
            results.append(future.result())
    write_json(args.output_root / "process_results.json", results)
    if any(row["returncode"] for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
