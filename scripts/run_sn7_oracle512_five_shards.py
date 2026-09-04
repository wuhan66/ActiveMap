#!/usr/bin/env python3
"""Build the full SN7 512px executable selector oracle on five GPUs."""

from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


REPO = Path("/home/wh/projects/activemap-v1")
PYTHON = "/home/wh/ActiveMap/envs/activemap-agent/bin/python"
STORAGE = Path("/home/wh/ActiveMap")
EPISODES = STORAGE / "processed/sn7_v1/agent/executable_selector_v2/episodes_train_val_full_assets.jsonl"
UPDATER = STORAGE / "runs/updater/v4_hierarchical_vector_change_scratch_seed20260716/best_quality.pt"
ROOT = STORAGE / "processed/sn7_v1/agent/executable_selector_v3_512_sharded"
GPUS = (1, 2, 3, 4, 5)


def gpu_pids(gpu: int) -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def build_shard(index: int, gpu: int) -> dict[str, object]:
    shard = ROOT / f"shard-{index:02d}"
    output = shard / "selector_states.jsonl"
    command = [
        PYTHON,
        "-m",
        "activemap.cli",
        "build-selector-oracle",
        str(UPDATER),
        str(shard / "episodes.jsonl"),
        str(output),
        "--device",
        "cuda:0",
        "--image-size",
        "512",
        "--budgets",
        "1.5,3.0,4.5",
        "--splits",
        "train,val",
        "--utility-mode",
        "executable",
        "--utility-profile",
        "balanced",
        "--writeback-delta-margin",
        "0.15",
        "--asset-root-map",
        "/mnt/mydisk/wh/ActiveMap=/home/wh/ActiveMap",
    ]
    env = os.environ | {"PYTHONPATH": ".:src", "CUDA_VISIBLE_DEVICES": str(gpu)}
    with (ROOT / "logs" / f"shard-{index:02d}.log").open("x", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    return {"shard": index, "gpu": gpu, "returncode": result.returncode, "output": str(output)}


def main() -> None:
    if ROOT.exists():
        raise FileExistsError(f"refusing to overwrite {ROOT}")
    occupied = {gpu: gpu_pids(gpu) for gpu in GPUS}
    if busy := {gpu: pids for gpu, pids in occupied.items() if pids}:
        raise RuntimeError(f"refusing occupied GPUs: {busy}")
    env = os.environ | {"PYTHONPATH": ".:src"}
    subprocess.run(
        [PYTHON, "scripts/shard_episodes.py", str(EPISODES), str(ROOT), "--num-shards", "5", "--seed", "20260722"],
        cwd=REPO,
        env=env,
        check=True,
    )
    (ROOT / "logs").mkdir()
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(build_shard, index, gpu) for index, gpu in enumerate(GPUS)]
        results = [future.result() for future in as_completed(futures)]
    (ROOT / "process_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    if any(int(row["returncode"]) != 0 for row in results):
        raise SystemExit("at least one 512 oracle shard failed")
    merged = ROOT / "states_train_val_executable_balanced_m15_512.jsonl"
    subprocess.run(
        [PYTHON, "scripts/merge_selector_state_shards.py", str(ROOT), str(merged)],
        cwd=REPO,
        env=env,
        check=True,
    )
    subprocess.run(
        [PYTHON, "scripts/audit_selector_states.py", str(merged), "--output", str(merged.with_suffix(".audit.json"))],
        cwd=REPO,
        env=env,
        check=True,
    )
    subprocess.run(
        [PYTHON, "scripts/audit_selector_utility_structure.py", str(merged), "--output", str(merged.with_suffix(".utility_audit.json"))],
        cwd=REPO,
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
