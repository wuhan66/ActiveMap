#!/usr/bin/env python3
"""Run parallel selector rollouts, convert traces, then launch matched writebacks."""

from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
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
SELECTOR_ROOT = Path(
    "/home/wh/ActiveMap/runs/selector/sn7_executable_two_stage_three_seed_v1"
)
RUN_ROOT = Path(
    "/home/wh/ActiveMap/runs/selector/sn7_executable_closed_loop_writeback_v1"
)
POLICIES = {
    "always_stop": (None, None),
    "edit_utility_s1": (
        0,
        SELECTOR_ROOT / "edit_utility_seed20260721/best.pt",
    ),
    "edit_utility_s2": (
        2,
        SELECTOR_ROOT / "edit_utility_seed20260722/best.pt",
    ),
    "edit_utility_s3": (
        4,
        SELECTOR_ROOT / "edit_utility_seed20260723/best.pt",
    ),
}


def run_rollout(policy: str, gpu: int | None, checkpoint: Path | None) -> dict[str, object]:
    output = RUN_ROOT / "rollouts" / policy
    command = [
        PYTHON,
        "scripts/evaluate_active_catalog_closed_loop_baselines.py",
        str(STATES),
        str(output),
        "--max-acquisitions",
        "2",
        "--bootstrap-repetitions",
        "0",
        "--policy",
        policy,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = ".:src"
    if checkpoint is not None:
        command.extend(["--device", "cuda:0", "--learned-selector", f"{policy}={checkpoint}"])
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_path = RUN_ROOT / "logs" / f"rollout_{policy}.log"
    with log_path.open("x", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=REPO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return {"policy": policy, "gpu": gpu, "returncode": result.returncode}


def main() -> None:
    if RUN_ROOT.exists():
        raise FileExistsError(f"refusing to overwrite {RUN_ROOT}")
    RUN_ROOT.mkdir(parents=True)
    (RUN_ROOT / "logs").mkdir()
    (RUN_ROOT / "inputs").mkdir()
    required = [STATES, EPISODES, UPDATER]
    required.extend(path for _, path in POLICIES.values() if path is not None)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(run_rollout, policy, gpu, checkpoint): policy
            for policy, (gpu, checkpoint) in POLICIES.items()
        }
        results = [future.result() for future in as_completed(futures)]
    (RUN_ROOT / "rollout_process_results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    if any(int(row["returncode"]) != 0 for row in results):
        raise SystemExit("at least one rollout failed; writeback was not launched")

    for policy in POLICIES:
        subprocess.run(
            [
                PYTHON,
                "scripts/convert_active_catalog_closed_loop_for_writeback.py",
                str(RUN_ROOT / "rollouts" / policy / f"{policy}.jsonl"),
                str(RUN_ROOT / "inputs" / f"{policy}.jsonl"),
            ],
            cwd=REPO,
            env=os.environ | {"PYTHONPATH": ".:src"},
            check=True,
        )

    subprocess.run(
        [
            PYTHON,
            "scripts/launch_sn7_two_stage_writeback_matrix.py",
            str(UPDATER),
            str(EPISODES),
            str(RUN_ROOT / "inputs"),
            str(RUN_ROOT / "writebacks"),
            "--gpu",
            "0",
            "--gpu",
            "2",
            "--gpu",
            "4",
            "--gpu",
            "5",
            "--python",
            PYTHON,
            "--image-size",
            "512",
            "--threshold",
            "0.5",
            "--delta-margin",
            "0.15",
            "--protocol-name",
            "sn7-executable-selector-polygon-writeback-v1",
            "--asset-root-map",
            "/mnt/mydisk/wh/ActiveMap=/home/wh/ActiveMap",
        ],
        cwd=REPO,
        env=os.environ | {"PYTHONPATH": ".:src"},
        check=True,
    )


if __name__ == "__main__":
    main()
