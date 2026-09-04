#!/usr/bin/env python3
"""Zero-test-access runtime preflight for the SN7 Step-0 frozen test."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from scripts.audit_sn7_step0_frozen_registry_v2 import audit


def _parse_gpu_rows(text: str) -> dict[int, dict[str, int]]:
    rows: dict[int, dict[str, int]] = {}
    for raw in text.splitlines():
        if not raw.strip():
            continue
        values = [item.strip() for item in raw.split(",")]
        if len(values) != 4:
            raise ValueError(f"unexpected nvidia-smi row: {raw}")
        index, used, total, utilization = map(int, values)
        rows[index] = {
            "memory_used_mib": used,
            "memory_total_mib": total,
            "utilization_percent": utilization,
        }
    return rows


def _command_checks(
    project_root: Path, python: Path
) -> list[dict[str, Any]]:
    commands = [
        [str(python), "-m", "activemap", "build-episodes-sn7", "--help"],
        [str(python), "-m", "activemap", "audit-episodes", "--help"],
        [str(python), "-m", "activemap", "build-selector-oracle", "--help"],
        [
            str(python),
            "scripts/evaluate_active_catalog_closed_loop_baselines.py",
            "--help",
        ],
        [
            str(python),
            "scripts/convert_active_catalog_closed_loop_for_writeback.py",
            "--help",
        ],
        [str(python), "scripts/launch_active_catalog_writeback.py", "--help"],
        [str(python), "scripts/summarize_sn7_step0_three_policy.py", "--help"],
        [str(python), "scripts/assess_sn7_step0_frontier_promotion.py", "--help"],
        [
            str(python),
            "scripts/aggregate_active_catalog_tool_writebacks.py",
            "--help",
        ],
        [
            str(python),
            "scripts/assess_active_catalog_tool_writeback_promotion.py",
            "--help",
        ],
        [
            str(python),
            "scripts/export_sn7_step0_frozen_test_tables.py",
            "--help",
        ],
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(project_root / "src"), str(project_root)]
    )
    results = []
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=project_root,
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=60,
        )
        results.append(
            {
                "command": command,
                "returncode": completed.returncode,
                "ready": completed.returncode == 0,
                "stderr_tail": completed.stderr[-500:],
            }
        )
    return results


def preflight(
    registry: Path,
    storage_root: Path,
    project_root: Path,
    python: Path,
    ledger: Path,
    run_root: Path,
    *,
    gpu_indices: list[int],
    minimum_free_gib: float,
) -> dict[str, Any]:
    registry_report = audit(registry, storage_root)
    command_checks = _command_checks(project_root, python)
    disk = shutil.disk_usage(storage_root)
    free_gib = disk.free / (1024**3)

    gpu_command = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=30,
    )
    gpu_rows = (
        _parse_gpu_rows(gpu_command.stdout) if gpu_command.returncode == 0 else {}
    )
    selected = {index: gpu_rows.get(index) for index in gpu_indices}
    gpu_ready = all(
        row is not None
        and row["memory_used_mib"] <= 512
        and row["utilization_percent"] <= 20
        for row in selected.values()
    )
    checks = {
        "registry_ready": registry_report.get("ready_for_frozen_test") is True,
        "all_cli_entrypoints_parse": all(row["ready"] for row in command_checks),
        "python_exists": python.is_file(),
        "ledger_absent": not ledger.exists(),
        "run_root_absent": not run_root.exists(),
        "authorization_token_absent": not bool(
            os.environ.get("ACTIVEMAP_FROZEN_TEST_TOKEN")
        ),
        "disk_free": free_gib >= minimum_free_gib,
        "selected_gpus_idle": gpu_ready,
    }
    return {
        "schema_version": "sn7-step0-frozen-test-runtime-preflight-v1",
        "ready": all(checks.values()),
        "checks": checks,
        "registry_summary": {
            "artifact_count": registry_report.get("artifact_count"),
            "blockers": registry_report.get("blockers"),
        },
        "runtime": {
            "selected_gpus": selected,
            "disk_free_gib": free_gib,
            "minimum_free_gib": minimum_free_gib,
            "ledger": str(ledger),
            "run_root": str(run_root),
        },
        "command_checks": command_checks,
        "test_assets_read": False,
        "note": "This preflight does not open test images, labels, or derived episodes.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("registry", type=Path)
    parser.add_argument("storage_root", type=Path)
    parser.add_argument("project_root", type=Path)
    parser.add_argument("python", type=Path)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--minimum-free-gib", type=float, default=20.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = preflight(
        args.registry,
        args.storage_root,
        args.project_root,
        args.python,
        args.ledger,
        args.run_root,
        gpu_indices=[int(item) for item in args.gpus.split(",") if item],
        minimum_free_gib=args.minimum_free_gib,
    )
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    raise SystemExit(0 if result["ready"] else 1)


if __name__ == "__main__":
    main()
