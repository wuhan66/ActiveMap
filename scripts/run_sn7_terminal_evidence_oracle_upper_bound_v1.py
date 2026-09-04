#!/usr/bin/env python3
"""Evaluate the SN7 shortlist oracle as a non-deployable writeback upper bound."""

from __future__ import annotations

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
ROLLOUT = STORAGE / (
    "runs/selector/sn7_executable_closed_loop_three_seed_v1/"
    "shortlist_oracle_upper_bound.jsonl"
)
RUN_ROOT = STORAGE / "runs/baselines/sn7_terminal_evidence_oracle_upper_bound_v1"
POLICY = "shortlist_oracle_upper_bound"


def main() -> None:
    inputs = RUN_ROOT / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    output = inputs / f"{POLICY}.jsonl"
    if not output.is_file():
        subprocess.run(
            [
                PYTHON,
                "scripts/prepare_terminal_evidence_writeback.py",
                str(STATES),
                str(ROLLOUT),
                str(output),
            ],
            cwd=REPO,
            env=os.environ | {"PYTHONPATH": ".:src"},
            check=True,
        )
    writeback_root = RUN_ROOT / "writebacks" / POLICY
    if writeback_root.exists():
        raise FileExistsError(f"refusing to overwrite {writeback_root}")
    subprocess.run(
        [
            PYTHON,
            "scripts/launch_active_catalog_writeback.py",
            str(CHECKPOINT),
            str(EPISODES),
            str(output),
            str(writeback_root),
            "--gpu",
            "1",
            "--python",
            PYTHON,
            "--image-size",
            "512",
            "--threshold",
            "0.5",
            "--delta-margin",
            "0.15",
            "--protocol-name",
            "sn7-terminal-evidence-oracle-upper-bound-v1",
            "--split",
            "val",
            "--asset-root-map",
            "/mnt/mydisk/wh/ActiveMap=/home/wh/ActiveMap",
        ],
        cwd=REPO,
        env=os.environ | {"PYTHONPATH": ".:src"},
        check=True,
    )


if __name__ == "__main__":
    main()
