#!/usr/bin/env python3
"""Launch the four-card SN7 two-stage selector diagnostic matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

BASE_CONFIG = Path("configs/selector/sn7_evidence_two_stage_v3_server.yaml")
VARIANTS = {
    "generic_gate_rank": {
        "conditioned": False,
        "regression_weight": 0.0,
        "gate_utility_weight": 0.0,
        "candidate_value_head": False,
    },
    "generic_utility": {
        "conditioned": False,
        "regression_weight": 1.0,
        "gate_utility_weight": 0.5,
        "candidate_value_head": True,
    },
    "edit_gate_rank": {
        "conditioned": True,
        "regression_weight": 0.0,
        "gate_utility_weight": 0.0,
        "candidate_value_head": False,
    },
    "edit_utility": {
        "conditioned": True,
        "regression_weight": 1.0,
        "gate_utility_weight": 0.5,
        "candidate_value_head": True,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_samples(samples: Path) -> dict[str, Any]:
    required = {
        "samples": samples,
        "audit": samples.with_suffix(".audit.json"),
        "utility_audit": samples.with_suffix(".utility_audit.json"),
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing audited selector inputs: {missing}")
    audit = json.loads(required["audit"].read_text(encoding="utf-8"))
    if bool(audit.get("test_assets_read", True)):
        raise RuntimeError("selector audit indicates frozen test access")
    if int(audit.get("duplicate_source_budget_steps", -1)) != 0:
        raise RuntimeError(
            "selector audit contains duplicate source/budget/step states"
        )
    if int(audit.get("states", 0)) <= 0:
        raise RuntimeError("selector audit contains no states")
    return {
        "path": str(samples.resolve()),
        "sha256": _sha256(samples),
        "states": int(audit["states"]),
        "targets": audit.get("targets", {}),
        "audits": {
            name: str(path.resolve())
            for name, path in required.items()
            if name != "samples"
        },
    }


def jobs(seed: int, gpus: list[int]) -> list[dict[str, Any]]:
    if len(gpus) != len(VARIANTS):
        raise ValueError("the two-stage matrix requires four distinct GPUs")
    return [
        {"variant": variant, "seed": seed, "gpu": gpus[index]}
        for index, variant in enumerate(VARIANTS)
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


def prepare_job(
    repo: Path, output_root: Path, row: dict[str, Any], samples: Path | None = None
) -> dict[str, Any]:
    payload = yaml.safe_load((repo / BASE_CONFIG).read_text(encoding="utf-8"))
    variant = VARIANTS[str(row["variant"])]
    run_dir = output_root / f"{row['variant']}_seed{row['seed']}"
    payload["seed"] = int(row["seed"])
    if samples is not None:
        payload["data"]["samples"] = str(samples.resolve())
    payload["training"]["utility_regression_weight"] = float(
        variant["regression_weight"]
    )
    payload["training"]["gate_utility_weight"] = float(variant["gate_utility_weight"])
    uses_value_head = bool(variant["candidate_value_head"])
    payload["model"]["candidate_value_head"] = uses_value_head
    payload["model"]["candidate_decision_mode"] = "value" if uses_value_head else "rank"
    payload["model"]["terminal_gate_mode"] = "value" if uses_value_head else "context"
    payload["training"]["candidate_value_sign_weight"] = 1.0 if uses_value_head else 0.0
    payload["training"]["context_gate_loss_weight"] = 0.0 if uses_value_head else 1.0
    payload["ablation"]["name"] = f"sn7_{row['variant']}_v3"
    payload["ablation"]["condition_on_hypothesis"] = bool(variant["conditioned"])
    payload["output_dir"] = str(run_dir.resolve())
    run_dir.mkdir(parents=True, exist_ok=False)
    config_path = run_dir / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return row | {"run_dir": str(run_dir), "config": str(config_path)}


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
    with (Path(str(row["run_dir"])) / "train.log").open("x", encoding="utf-8") as log:
        result = subprocess.run(
            command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT
        )
    return row | {"returncode": result.returncode, "command": command}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--gpu", type=int, action="append", required=True)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    gpus = list(dict.fromkeys(args.gpu))
    plan = jobs(args.seed, gpus)
    samples_manifest = validate_samples(args.samples)
    occupied = {gpu: gpu_processes(gpu) for gpu in gpus}
    occupied = {gpu: pids for gpu, pids in occupied.items() if pids}
    if occupied:
        raise RuntimeError(f"refusing occupied GPUs: {occupied}")
    manifest = {
        "schema_version": "sn7-two-stage-selector-sweep-v3",
        "jobs": plan,
        "samples": samples_manifest,
        "test_assets_read": False,
    }
    print(json.dumps(manifest, indent=2), flush=True)
    if args.dry_run:
        return
    args.output_root.mkdir(parents=True, exist_ok=False)
    prepared = [prepare_job(repo, args.output_root, row, args.samples) for row in plan]
    (args.output_root / "launch_manifest.json").write_text(
        json.dumps(manifest | {"jobs": prepared}, indent=2) + "\n", encoding="utf-8"
    )
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(run_job, args.python, repo, row): row for row in prepared
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
