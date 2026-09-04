#!/usr/bin/env python3
"""Turn a completed scene-disjoint Habitat suite into split-safe map episodes."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

SPLITS = ("train", "val", "test")


def load_materializer():
    script = Path(__file__).with_name("materialize_habitat_rgbd_navigation.py")
    spec = importlib.util.spec_from_file_location("habitat_rgbd_materializer", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load materializer: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def group_suite_trajectories(
    summary: dict[str, object], *, selected_splits: tuple[str, ...] = SPLITS
) -> dict[str, list[tuple[str, Path]]]:
    if summary.get("schema_version") != "activemap-habitat-scene-disjoint-suite-v1":
        raise ValueError("unsupported suite summary")
    if summary.get("development_only") is not True or summary.get("scene_disjoint") is not True:
        raise ValueError("suite must be development-only and scene-disjoint")
    raw_rows = summary.get("trajectories")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("suite summary has no trajectories")
    if not selected_splits or len(set(selected_splits)) != len(selected_splits):
        raise ValueError("selected splits must be non-empty and unique")
    if any(split not in SPLITS for split in selected_splits):
        raise ValueError(f"unsupported selected splits: {selected_splits}")
    grouped: dict[str, list[tuple[str, Path]]] = {split: [] for split in selected_splits}
    scenes_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    seen_ids: set[str] = set()
    for row in raw_rows:
        if not isinstance(row, dict):
            raise ValueError("suite trajectory must be an object")
        split = row.get("split")
        scene = row.get("scene")
        trajectory = row.get("trajectory")
        seed = row.get("seed")
        if not isinstance(split, str) or split not in SPLITS:
            raise ValueError(f"invalid split: {split}")
        if (
            not isinstance(scene, str)
            or not isinstance(trajectory, str)
            or not isinstance(seed, int)
        ):
            raise ValueError("suite trajectory is missing scene, path, or seed")
        episode_id = f"{split}-{scene}-seed{seed}"
        if episode_id in seen_ids:
            raise ValueError(f"duplicate episode ID: {episode_id}")
        seen_ids.add(episode_id)
        scenes_by_split[split].add(scene)
        if split not in grouped:
            continue
        trajectory_path = Path(trajectory)
        if not (trajectory_path / "summary.json").is_file():
            raise FileNotFoundError(trajectory_path / "summary.json")
        grouped[split].append((episode_id, trajectory_path))
    if any(len(scenes) != 1 for scenes in scenes_by_split.values()):
        raise ValueError("each development split must contain exactly one scene")
    if len({next(iter(scenes)) for scenes in scenes_by_split.values()}) != 3:
        raise ValueError("scene leakage across train/val/test splits")
    return grouped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path, help="completed suite directory")
    parser.add_argument("output", type=Path, help="new materialized episode directory")
    parser.add_argument("--resolution-m", type=float, default=0.05)
    parser.add_argument("--max-range-m", type=float, default=5.0)
    parser.add_argument("--horizontal-fov-degrees", type=float, default=90.0)
    parser.add_argument("--ray-stride", type=int, default=4)
    parser.add_argument("--budget", type=float, default=8.0)
    parser.add_argument("--candidate-corruption-rate", type=float, default=0.0)
    parser.add_argument("--corruption-seed", type=int, default=20260830)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=SPLITS,
        help="splits to materialize; excluded split assets are not opened",
    )
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    summary_path = args.suite / "suite_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selected_splits = tuple(args.splits)
    grouped = group_suite_trajectories(summary, selected_splits=selected_splits)
    materializer = load_materializer()
    args.output.mkdir(parents=True)
    split_summaries: dict[str, object] = {}
    for split, trajectories in grouped.items():
        split_summaries[split] = materializer.materialize_episodes(
            trajectories=trajectories,
            output=args.output / "splits" / split,
            resolution_m=args.resolution_m,
            max_range_m=args.max_range_m,
            horizontal_fov_degrees=args.horizontal_fov_degrees,
            ray_stride=args.ray_stride,
            budget=args.budget,
            split=split,
            candidate_corruption_rate=args.candidate_corruption_rate,
            corruption_seed=args.corruption_seed,
        )
    manifest_root = args.output / "manifests"
    manifest_root.mkdir()
    for split in selected_splits:
        source = args.output / "splits" / split / "manifests" / f"{split}.jsonl"
        manifest_path = manifest_root / f"{split}.jsonl"
        manifest_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    result = {
        "schema_version": "activemap-habitat-scene-disjoint-navigation-v1",
        "development_only": True,
        "scene_disjoint": True,
        "materialized_splits": list(selected_splits),
        "suite": str(args.suite.resolve()),
        "output": str(args.output.resolve()),
        "split_counts": {split: len(rows) for split, rows in grouped.items()},
        "split_summaries": split_summaries,
        "test_assets_read": "test" in selected_splits,
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
