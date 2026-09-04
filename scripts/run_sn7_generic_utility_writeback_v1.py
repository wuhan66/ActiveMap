#!/usr/bin/env python3
"""Run the generic-selector ablation rollout and matched SN7 writeback on GPU 1."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO = Path("/home/wh/projects/activemap-v1")
PYTHON = "/home/wh/ActiveMap/envs/activemap-agent/bin/python"
STATES = Path(
    "/home/wh/ActiveMap/processed/sn7_v1/agent/executable_selector_v2/"
    "states_train_val_executable_balanced_m15.jsonl"
)
EPISODES = Path(
    "/home/wh/ActiveMap/processed/sn7_v1/agent/executable_selector_v2/"
    "episodes_train_val_full_assets.jsonl"
)
UPDATER = Path(
    "/home/wh/ActiveMap/runs/updater/"
    "v4_hierarchical_vector_change_scratch_seed20260716/best_quality.pt"
)
SELECTOR = Path(
    "/home/wh/ActiveMap/runs/selector/sn7_executable_two_stage_sweep_seed20260721_v1/"
    "generic_utility_seed20260721/best.pt"
)
RUN_ROOT = Path(
    "/home/wh/ActiveMap/runs/selector/"
    "sn7_executable_generic_utility_writeback_seed20260721_v1"
)
POLICY = "generic_utility"


def main() -> None:
    if RUN_ROOT.exists():
        raise FileExistsError(f"refusing to overwrite {RUN_ROOT}")
    missing = [str(path) for path in (STATES, EPISODES, UPDATER, SELECTOR) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")
    RUN_ROOT.mkdir(parents=True)
    env = os.environ | {"PYTHONPATH": ".:src", "CUDA_VISIBLE_DEVICES": "1"}

    rollout_dir = RUN_ROOT / "rollout"
    with (RUN_ROOT / "rollout.log").open("x", encoding="utf-8") as log:
        subprocess.run(
            [
                PYTHON,
                "scripts/evaluate_active_catalog_closed_loop_baselines.py",
                str(STATES),
                str(rollout_dir),
                "--max-acquisitions",
                "2",
                "--bootstrap-repetitions",
                "0",
                "--policy",
                POLICY,
                "--device",
                "cuda:0",
                "--learned-selector",
                f"{POLICY}={SELECTOR}",
            ],
            cwd=REPO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )

    writeback_input = RUN_ROOT / f"{POLICY}.jsonl"
    subprocess.run(
        [
            PYTHON,
            "scripts/convert_active_catalog_closed_loop_for_writeback.py",
            str(rollout_dir / f"{POLICY}.jsonl"),
            str(writeback_input),
        ],
        cwd=REPO,
        env=env,
        check=True,
    )
    subprocess.run(
        [
            PYTHON,
            "scripts/launch_active_catalog_writeback.py",
            str(UPDATER),
            str(EPISODES),
            str(writeback_input),
            str(RUN_ROOT / "writeback"),
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
            "sn7-executable-generic-selector-polygon-writeback-v1",
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
