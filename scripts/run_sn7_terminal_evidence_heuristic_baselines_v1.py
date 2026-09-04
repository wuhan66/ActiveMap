#!/usr/bin/env python3
"""Run four matched SN7 heuristic controllers through executable writeback."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


REPO = Path("/home/wh/projects/activemap-v1")
PYTHON = "/home/wh/ActiveMap/envs/activemap-agent/bin/python"
STORAGE = Path("/home/wh/ActiveMap")
BUNDLE = STORAGE / (
    "processed/sn7_v1/agent/executable_selector_v3_512_sharded/"
    "closed_loop_val_bundle_v1"
)
STATES = BUNDLE / "states_val_step0.jsonl"
EPISODES = BUNDLE / "episodes_val.jsonl"
CHECKPOINT = STORAGE / (
    "runs/updater/v4_hierarchical_vector_change_scratch_seed20260716/"
    "best_quality.pt"
)
ROLLOUT_ROOT = STORAGE / "runs/selector/sn7_executable_closed_loop_three_seed_v1"
RUN_ROOT = STORAGE / "runs/baselines/sn7_terminal_evidence_heuristics_v1"
POLICIES = ("cheapest", "clear_per_cost", "uncertainty_gate", "random")


def main() -> None:
    if RUN_ROOT.exists():
        raise FileExistsError(f"refusing to overwrite {RUN_ROOT}")
    inputs = RUN_ROOT / "inputs"
    inputs.mkdir(parents=True)
    conversions = []
    for policy in POLICIES:
        output = inputs / f"{policy}.jsonl"
        command = [
            PYTHON,
            "scripts/prepare_terminal_evidence_writeback.py",
            str(STATES),
            str(ROLLOUT_ROOT / f"{policy}.jsonl"),
            str(output),
        ]
        result = subprocess.run(
            command,
            cwd=REPO,
            env=os.environ | {"PYTHONPATH": ".:src"},
            capture_output=True,
            text=True,
        )
        (inputs / f"{policy}.log").write_text(
            result.stdout + result.stderr, encoding="utf-8"
        )
        conversions.append({"policy": policy, "returncode": result.returncode})
    (RUN_ROOT / "conversion_results.json").write_text(
        json.dumps(conversions, indent=2) + "\n", encoding="utf-8"
    )
    if any(row["returncode"] for row in conversions):
        raise SystemExit("terminal-evidence conversion failed")

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
        "sn7-terminal-evidence-heuristic-baselines-v1",
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
