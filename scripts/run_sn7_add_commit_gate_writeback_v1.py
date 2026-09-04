#!/usr/bin/env python3
"""Evaluate STOP and three frozen h18 selectors with an ADD-only commit gate."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from apply_terminal_operation_gate import apply_gate


REPO = Path("/home/wh/projects/activemap-v1")
PYTHON = "/home/wh/ActiveMap/envs/activemap-agent/bin/python"
STORAGE = Path("/home/wh/ActiveMap")
CHECKPOINT = STORAGE / (
    "runs/updater/v4_hierarchical_vector_change_scratch_seed20260716/"
    "best_quality.pt"
)
EPISODES = STORAGE / (
    "processed/sn7_v1/agent/executable_selector_v3_512_sharded/"
    "closed_loop_val_bundle_v1/episodes_val.jsonl"
)
SOURCE_ROOT = STORAGE / (
    "runs/selector/sn7_value_aware_h18_terminal_evidence_writeback_v1/inputs"
)
RUN_ROOT = STORAGE / "runs/selector/sn7_value_aware_h18_add_commit_gate_writeback_v1"
POLICIES = (
    "always_stop",
    "add_gate_h18_s1",
    "add_gate_h18_s2",
    "add_gate_h18_s3",
)


def resume_gated() -> None:
    if not RUN_ROOT.exists():
        raise FileNotFoundError(RUN_ROOT)
    inputs = RUN_ROOT / "inputs"
    jobs = []
    for index, gpu in zip((1, 2, 3), (2, 3, 4), strict=True):
        label = f"add_gate_h18_s{index}_v2"
        apply_gate(
            SOURCE_ROOT / f"generic_value_h18_s{index}.jsonl",
            inputs / f"{label}.jsonl",
            {"ADD"},
        )
        command = [
            PYTHON,
            "scripts/launch_active_catalog_writeback.py",
            str(CHECKPOINT),
            str(EPISODES),
            str(inputs / f"{label}.jsonl"),
            str(RUN_ROOT / "writebacks" / label),
            "--gpu",
            str(gpu),
            "--python",
            PYTHON,
            "--image-size",
            "512",
            "--threshold",
            "0.5",
            "--delta-margin",
            "0.15",
            "--protocol-name",
            "sn7-add-commit-gate-writeback-v2",
            "--split",
            "val",
            "--asset-root-map",
            "/mnt/mydisk/wh/ActiveMap=/home/wh/ActiveMap",
        ]
        jobs.append((label, command))
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(
            executor.map(
                lambda job: subprocess.run(
                    job[1],
                    cwd=REPO,
                    env=os.environ | {"PYTHONPATH": ".:src"},
                ).returncode,
                jobs,
            )
        )
    (RUN_ROOT / "resume_gated_process_results.json").write_text(
        json.dumps(
            [
                {"policy": jobs[index][0], "returncode": returncode}
                for index, returncode in enumerate(results)
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if any(results):
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-gated", action="store_true")
    args = parser.parse_args()
    if args.resume_gated:
        resume_gated()
        return
    if RUN_ROOT.exists():
        raise FileExistsError(f"refusing to overwrite {RUN_ROOT}")
    inputs = RUN_ROOT / "inputs"
    inputs.mkdir(parents=True)
    shutil.copy2(SOURCE_ROOT / "always_stop.jsonl", inputs / "always_stop.jsonl")
    gate_counts = {}
    for index in (1, 2, 3):
        label = f"add_gate_h18_s{index}"
        gate_counts[label] = apply_gate(
            SOURCE_ROOT / f"generic_value_h18_s{index}.jsonl",
            inputs / f"{label}.jsonl",
            {"ADD"},
        )
    (RUN_ROOT / "gate_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "sn7-add-commit-gate-v1",
                "allowed_operations": ["ADD"],
                "uses_ground_truth": False,
                "spent_cost_retained": True,
                "gate_counts": gate_counts,
                "split": "val",
                "test_assets_read": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    command = [
        PYTHON,
        "scripts/launch_sn7_two_stage_writeback_matrix.py",
        str(CHECKPOINT),
        str(EPISODES),
        str(inputs),
        str(RUN_ROOT / "writebacks"),
        "--python",
        PYTHON,
        "--image-size",
        "512",
        "--threshold",
        "0.5",
        "--delta-margin",
        "0.15",
        "--protocol-name",
        "sn7-add-commit-gate-writeback-v1",
        "--asset-root-map",
        "/mnt/mydisk/wh/ActiveMap=/home/wh/ActiveMap",
    ]
    for gpu in (1, 2, 3, 4):
        command.extend(["--gpu", str(gpu)])
    for policy in POLICIES:
        command.extend(["--policy", policy])
    subprocess.run(
        command,
        cwd=REPO,
        env=os.environ | {"PYTHONPATH": ".:src"},
        check=True,
    )


if __name__ == "__main__":
    main()
