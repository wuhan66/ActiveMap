#!/usr/bin/env python3
"""Run four terminal-evidence-only SN7 writeback diagnostics in parallel."""

from __future__ import annotations

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


REPO = Path("/home/wh/projects/activemap-v1")
PYTHON = "/home/wh/ActiveMap/envs/activemap-agent/bin/python"
STORAGE = Path("/home/wh/ActiveMap")
STATES = STORAGE / "processed/sn7_v1/agent/executable_selector_v2/states_train_val_executable_balanced_m15.jsonl"
EPISODES = STORAGE / "processed/sn7_v1/agent/executable_selector_v2/episodes_train_val_full_assets.jsonl"
UPDATER = STORAGE / "runs/updater/v4_hierarchical_vector_change_scratch_seed20260716/best_quality.pt"
MAIN = STORAGE / "runs/selector/sn7_executable_closed_loop_writeback_v1"
GENERIC = STORAGE / "runs/selector/sn7_executable_generic_utility_writeback_seed20260721_v1"
ROOT = STORAGE / "runs/selector/sn7_terminal_evidence_writeback_matrix_v1"
JOBS = {
    "edit_utility_s1": (0, MAIN / "rollouts/edit_utility_s1/edit_utility_s1.jsonl"),
    "generic_utility": (1, GENERIC / "rollout/generic_utility.jsonl"),
    "edit_utility_s2": (2, MAIN / "rollouts/edit_utility_s2/edit_utility_s2.jsonl"),
    "edit_utility_s3": (4, MAIN / "rollouts/edit_utility_s3/edit_utility_s3.jsonl"),
}


def writeback(policy: str, gpu: int, rollout: Path) -> tuple[str, int]:
    input_path = ROOT / "inputs" / f"{policy}.jsonl"
    subprocess.run(
        [PYTHON, "scripts/prepare_terminal_evidence_writeback.py", str(STATES), str(rollout), str(input_path)],
        cwd=REPO,
        env=os.environ | {"PYTHONPATH": ".:src"},
        check=True,
    )
    log_path = ROOT / "logs" / f"{policy}.log"
    with log_path.open("x", encoding="utf-8") as log:
        result = subprocess.run(
            [
                PYTHON,
                "scripts/launch_active_catalog_writeback.py",
                str(UPDATER),
                str(EPISODES),
                str(input_path),
                str(ROOT / "writebacks" / policy),
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
                "sn7-terminal-evidence-only-diagnostic-v1",
                "--split",
                "val",
                "--asset-root-map",
                "/mnt/mydisk/wh/ActiveMap=/home/wh/ActiveMap",
            ],
            cwd=REPO,
            env=os.environ | {"PYTHONPATH": ".:src"},
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return policy, result.returncode


def main() -> None:
    if ROOT.exists():
        raise FileExistsError(f"refusing to overwrite {ROOT}")
    (ROOT / "inputs").mkdir(parents=True)
    (ROOT / "logs").mkdir()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(writeback, policy, gpu, rollout) for policy, (gpu, rollout) in JOBS.items()]
        results = [future.result() for future in as_completed(futures)]
    if any(code != 0 for _, code in results):
        raise SystemExit(f"writeback failure: {results}")


if __name__ == "__main__":
    main()
