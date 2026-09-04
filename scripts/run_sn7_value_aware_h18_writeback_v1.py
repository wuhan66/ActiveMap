#!/usr/bin/env python3
"""Run frozen h18 selector rollouts and matched 512px vector writebacks."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO = Path("/home/wh/projects/activemap-v1")
PYTHON = "/home/wh/ActiveMap/envs/activemap-agent/bin/python"
DATA_ROOT = Path(
    "/home/wh/ActiveMap/processed/sn7_v1/agent/"
    "executable_selector_v3_512_sharded"
)
BUNDLE = DATA_ROOT / "closed_loop_val_bundle_v1"
STATES = BUNDLE / "states_val_step0.jsonl"
EPISODES = BUNDLE / "episodes_val.jsonl"
UPDATER = Path(
    "/home/wh/ActiveMap/runs/updater/"
    "v4_hierarchical_vector_change_scratch_seed20260716/best_quality.pt"
)
SELECTOR_ROOT = Path(
    "/home/wh/ActiveMap/runs/selector/"
    "sn7_executable512_value_aware_generic_three_seed_h20_v1"
)
COMMON_WORKPOINT = SELECTOR_ROOT / "common_safety_workpoint_v1.json"
RUN_ROOT = Path(
    "/home/wh/ActiveMap/runs/selector/"
    "sn7_value_aware_h18_closed_loop_writeback_v1"
)
POLICIES = {
    "always_stop": (None, None),
    "generic_value_h18_s1": (
        1,
        SELECTOR_ROOT / "generic_utility_seed20260721/best.pt",
    ),
    "generic_value_h18_s2": (
        2,
        SELECTOR_ROOT / "generic_utility_seed20260722/best.pt",
    ),
    "generic_value_h18_s3": (
        3,
        SELECTOR_ROOT / "generic_utility_seed20260723/best.pt",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_margins() -> dict[Path, float]:
    payload = json.loads(COMMON_WORKPOINT.read_text(encoding="utf-8"))
    winner = payload.get("winner")
    if not payload.get("promoted") or not isinstance(winner, dict):
        raise ValueError("common safety workpoint is not promoted")
    if winner.get("label") != "h18_r10":
        raise ValueError(f"expected frozen h18_r10 winner, found {winner.get('label')}")
    return {
        Path(row["checkpoint"]["path"]): float(row["stop_margin"])
        for row in winner["per_seed"]
    }


def run_rollout(
    policy: str,
    gpu: int | None,
    checkpoint: Path | None,
    margins: dict[Path, float],
) -> dict[str, Any]:
    output = RUN_ROOT / "rollouts" / policy
    command = [
        PYTHON,
        "-m",
        "scripts.evaluate_active_catalog_closed_loop_baselines",
        str(STATES),
        str(output),
        "--max-acquisitions",
        "2",
        "--bootstrap-repetitions",
        "0",
        "--policy",
        policy,
    ]
    env = os.environ | {"PYTHONPATH": ".:src"}
    if checkpoint is not None:
        margin = margins.get(checkpoint)
        if margin is None:
            raise ValueError(f"no promoted margin for {checkpoint}")
        command.extend(
            [
                "--device",
                "cuda:0",
                "--learned-selector",
                f"{policy}={checkpoint}",
                "--learned-selector-stop-margin",
                f"{policy}={margin}",
            ]
        )
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
    return {
        "policy": policy,
        "gpu": gpu,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "stop_margin": margins.get(checkpoint) if checkpoint is not None else None,
        "returncode": result.returncode,
        "command": command,
    }


def main() -> None:
    if RUN_ROOT.exists():
        raise FileExistsError(f"refusing to overwrite {RUN_ROOT}")
    required = [STATES, EPISODES, UPDATER, COMMON_WORKPOINT]
    required.extend(path for _, path in POLICIES.values() if path is not None)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")
    margins = load_margins()
    RUN_ROOT.mkdir(parents=True)
    (RUN_ROOT / "logs").mkdir()
    (RUN_ROOT / "inputs").mkdir()
    manifest = {
        "schema_version": "sn7-value-aware-h18-writeback-v1",
        "states": {"path": str(STATES), "sha256": sha256(STATES)},
        "episodes": {"path": str(EPISODES), "sha256": sha256(EPISODES)},
        "updater": {"path": str(UPDATER), "sha256": sha256(UPDATER)},
        "common_workpoint": {
            "path": str(COMMON_WORKPOINT),
            "sha256": sha256(COMMON_WORKPOINT),
        },
        "policies": list(POLICIES),
        "split": "val",
        "test_assets_read": False,
    }
    (RUN_ROOT / "launch_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(run_rollout, policy, gpu, checkpoint, margins): policy
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
                "-m",
                "scripts.convert_active_catalog_closed_loop_for_writeback",
                str(RUN_ROOT / "rollouts" / policy / f"{policy}.jsonl"),
                str(RUN_ROOT / "inputs" / f"{policy}.jsonl"),
            ],
            cwd=REPO,
            env=os.environ | {"PYTHONPATH": ".:src"},
            check=True,
        )

    command = [
        PYTHON,
        "-m",
        "scripts.launch_sn7_two_stage_writeback_matrix",
        str(UPDATER),
        str(EPISODES),
        str(RUN_ROOT / "inputs"),
        str(RUN_ROOT / "writebacks"),
    ]
    for gpu in (1, 2, 3, 4):
        command.extend(["--gpu", str(gpu)])
    for policy in POLICIES:
        command.extend(["--policy", policy])
    command.extend(
        [
            "--python",
            PYTHON,
            "--image-size",
            "512",
            "--threshold",
            "0.5",
            "--delta-margin",
            "0.15",
            "--protocol-name",
            "sn7-value-aware-h18-polygon-writeback-v1",
            "--asset-root-map",
            "/mnt/mydisk/wh/ActiveMap=/home/wh/ActiveMap",
        ]
    )
    subprocess.run(
        command,
        cwd=REPO,
        env=os.environ | {"PYTHONPATH": ".:src"},
        check=True,
    )


if __name__ == "__main__":
    main()
