#!/usr/bin/env python3
"""Collect reproducible, scene-disjoint Habitat RGB-D development trajectories."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

SCENE_SPLITS = (
    ("train", "apartment_1"),
    ("val", "van-gogh-room"),
    ("test", "skokloster-castle"),
)


@dataclass(frozen=True)
class TrajectoryJob:
    split: str
    scene: str
    seed: int
    gpu_id: int
    output: Path
    command: tuple[str, ...]


def parse_gpu_ids(raw: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values:
        raise ValueError("at least one GPU ID is required")
    if len(set(values)) != len(values):
        raise ValueError("GPU IDs must be unique")
    if any(value < 0 for value in values):
        raise ValueError("GPU IDs must be non-negative")
    return values


def build_jobs(
    *,
    output: Path,
    scene_root: Path,
    episodes_per_scene: int,
    action_count: int,
    gpu_ids: tuple[int, ...],
    base_seed: int,
    resolution: int,
) -> list[TrajectoryJob]:
    if episodes_per_scene < 1:
        raise ValueError("episodes_per_scene must be positive")
    if action_count < 1:
        raise ValueError("action_count must be positive")
    recorder = Path(__file__).with_name("run_habitat_replica_rgbd_smoke.py")
    jobs: list[TrajectoryJob] = []
    for split_index, (split, scene) in enumerate(SCENE_SPLITS):
        scene_path = scene_root / f"{scene}.glb"
        if not scene_path.is_file():
            raise FileNotFoundError(scene_path)
        for trajectory_index in range(episodes_per_scene):
            seed = base_seed + split_index * 10_000 + trajectory_index
            gpu_id = gpu_ids[len(jobs) % len(gpu_ids)]
            job_output = output / "trajectories" / f"{split}_{scene}_seed{seed}"
            command = (
                sys.executable,
                str(recorder),
                str(job_output),
                "--scene",
                str(scene_path),
                "--gpu-id",
                str(gpu_id),
                "--resolution",
                str(resolution),
                "--start-mode",
                "random-navigable",
                "--seed",
                str(seed),
                "--random-action-count",
                str(action_count),
            )
            jobs.append(
                TrajectoryJob(
                    split=split,
                    scene=scene,
                    seed=seed,
                    gpu_id=gpu_id,
                    output=job_output,
                    command=command,
                )
            )
    return jobs


def run_job(job: TrajectoryJob) -> dict[str, object]:
    log_path = job.output.with_suffix(".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            job.command,
            check=False,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if result.returncode:
        raise RuntimeError(f"trajectory failed: {job.output}; see {log_path}")
    summary_path = job.output / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"trajectory did not write summary: {job.output}")
    return {
        "split": job.split,
        "scene": job.scene,
        "seed": job.seed,
        "gpu_id": job.gpu_id,
        "trajectory": str(job.output.resolve()),
        "summary": str(summary_path.resolve()),
        "log": str(log_path.resolve()),
    }


def run_gpu_queue(jobs: list[TrajectoryJob]) -> list[dict[str, object]]:
    """Run a serial queue per GPU so Habitat contexts never collide."""
    return [run_job(job) for job in jobs]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new suite output directory")
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--episodes-per-scene", type=int, default=12)
    parser.add_argument("--action-count", type=int, default=18)
    parser.add_argument("--gpus", default="1,2,3")
    parser.add_argument("--base-seed", type=int, default=20260830)
    parser.add_argument("--resolution", type=int, default=256)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    gpu_ids = parse_gpu_ids(args.gpus)
    jobs = build_jobs(
        output=args.output.resolve(),
        scene_root=args.scene_root.resolve(),
        episodes_per_scene=args.episodes_per_scene,
        action_count=args.action_count,
        gpu_ids=gpu_ids,
        base_seed=args.base_seed,
        resolution=args.resolution,
    )
    args.output.mkdir(parents=True)
    spec = {
        "schema_version": "activemap-habitat-scene-disjoint-suite-v1",
        "development_only": True,
        "scene_disjoint": True,
        "scene_splits": [{"split": split, "scene": scene} for split, scene in SCENE_SPLITS],
        "episodes_per_scene": args.episodes_per_scene,
        "action_count": args.action_count,
        "gpu_ids": list(gpu_ids),
        "base_seed": args.base_seed,
        "resolution": args.resolution,
        "jobs": [{**asdict(job), "output": str(job.output)} for job in jobs],
    }
    spec_path = args.output / "suite_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")

    jobs_by_gpu = {gpu_id: [job for job in jobs if job.gpu_id == gpu_id] for gpu_id in gpu_ids}
    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = [executor.submit(run_gpu_queue, gpu_jobs) for gpu_jobs in jobs_by_gpu.values()]
        for future in as_completed(futures):
            results.extend(future.result())
    results.sort(key=lambda record: (str(record["split"]), int(record["seed"])))
    summary = {
        "schema_version": "activemap-habitat-scene-disjoint-suite-v1",
        "development_only": True,
        "scene_disjoint": True,
        "trajectory_count": len(results),
        "split_counts": {
            split: sum(record["split"] == split for record in results) for split, _ in SCENE_SPLITS
        },
        "trajectories": results,
    }
    (args.output / "suite_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
