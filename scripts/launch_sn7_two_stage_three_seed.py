#!/usr/bin/env python3
"""Launch a three-seed SN7 two-stage selector replication."""

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

try:
    from scripts.launch_sn7_two_stage_selector_sweep import VARIANTS
except ModuleNotFoundError:  # Direct execution adds scripts/ to sys.path.
    from launch_sn7_two_stage_selector_sweep import VARIANTS

BASE_CONFIG = Path("configs/selector/sn7_evidence_two_stage_v3_server.yaml")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_samples(samples: Path) -> dict[str, Any]:
    audit_source = samples
    sanitizer_summary: Path | None = None
    direct_audit = samples.with_suffix(".audit.json")
    direct_utility_audit = samples.with_suffix(".utility_audit.json")
    if not direct_audit.is_file() or not direct_utility_audit.is_file():
        sanitizer_summary = samples.with_suffix(samples.suffix + ".summary.json")
        if not sanitizer_summary.is_file():
            raise FileNotFoundError(
                "selector inputs lack direct audits and a sanitizer summary: "
                f"{samples}"
            )
        summary = json.loads(sanitizer_summary.read_text(encoding="utf-8"))
        if summary.get("schema_version") != "online-observable-selector-features-v1":
            raise RuntimeError("unsupported selector sanitizer summary")
        if bool(summary.get("test_assets_read", True)):
            raise RuntimeError("selector sanitizer summary indicates frozen test access")
        contract = summary.get("online_state_contract", {})
        if contract.get("version") != "online-observable-state-v1":
            raise RuntimeError("selector sanitizer summary lacks the online-observable contract")
        output = summary.get("output", {})
        if Path(str(output.get("path", ""))).resolve() != samples.resolve():
            raise RuntimeError("selector sanitizer output path does not match the samples path")
        if output.get("sha256") != _sha256(samples):
            raise RuntimeError("selector sanitizer output hash does not match the samples file")
        source = summary.get("source", {})
        audit_source = Path(str(source.get("path", "")))
        if not audit_source.is_file() or source.get("sha256") != _sha256(audit_source):
            raise RuntimeError("selector sanitizer source is missing or has a mismatched hash")

    required = {
        "samples": samples,
        "audit": audit_source.with_suffix(".audit.json"),
        "utility_audit": audit_source.with_suffix(".utility_audit.json"),
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
    result = {
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
    if sanitizer_summary is not None:
        result["sanitizer_summary"] = {
            "path": str(sanitizer_summary.resolve()),
            "sha256": _sha256(sanitizer_summary),
        }
    return result


def jobs(seeds: list[int], gpus: list[int]) -> list[dict[str, int]]:
    if len(seeds) != 3 or len(gpus) != 3:
        raise ValueError("the formal replication requires three seeds and three GPUs")
    if len(set(seeds)) != 3 or len(set(gpus)) != 3:
        raise ValueError("seeds and GPUs must be distinct")
    return [{"seed": seed, "gpu": gpus[index]} for index, seed in enumerate(seeds)]


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
    repo: Path,
    output_root: Path,
    row: dict[str, int],
    *,
    base_config: Path,
    variant_name: str = "edit_utility",
    samples: Path | None = None,
    require_online_observable: bool = False,
) -> dict[str, Any]:
    payload = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    if require_online_observable:
        data_contract = payload.get("data_contract", {})
        if data_contract.get("version") != "online-observable-state-v1":
            raise ValueError(
                "the selected base config does not declare online-observable-state-v1"
            )
    variant = VARIANTS[variant_name]
    run_dir = output_root / f"{variant_name}_seed{row['seed']}"
    payload["seed"] = row["seed"]
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
    payload["ablation"]["name"] = f"sn7_{variant_name}_executable_v1"
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--gpu", type=int, action="append", required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="edit_utility")
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--require-online-observable", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    base_config = args.base_config if args.base_config.is_absolute() else repo / args.base_config
    if not base_config.is_file():
        raise FileNotFoundError(base_config)
    plan = jobs(args.seed, args.gpu)
    samples_manifest = validate_samples(args.samples)
    occupied = {row["gpu"]: gpu_processes(row["gpu"]) for row in plan}
    occupied = {gpu: pids for gpu, pids in occupied.items() if pids}
    if occupied:
        raise RuntimeError(f"refusing occupied GPUs: {occupied}")
    manifest = {
        "schema_version": "sn7-two-stage-selector-three-seed-v3",
        "jobs": plan,
        "variant": args.variant,
        "base_config": str(base_config.resolve()),
        "require_online_observable": bool(args.require_online_observable),
        "samples": samples_manifest,
        "test_assets_read": False,
    }
    print(json.dumps(manifest, indent=2), flush=True)
    if args.dry_run:
        return
    args.output_root.mkdir(parents=True, exist_ok=False)
    prepared = [
        prepare_job(
            repo,
            args.output_root,
            row,
            base_config=base_config,
            variant_name=args.variant,
            samples=args.samples,
            require_online_observable=args.require_online_observable,
        )
        for row in plan
    ]
    (args.output_root / "launch_manifest.json").write_text(
        json.dumps(manifest | {"jobs": prepared}, indent=2) + "\n", encoding="utf-8"
    )
    results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
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
