#!/usr/bin/env python3
"""Materialize development-only ActiveMap episodes from recorded Habitat RGB-D views.

The controller sees only the committed occupancy grid, robot pose, budget, and
candidate pose/footprint metadata.  Depth observations are loaded only after a
candidate is selected.  The fused reference map is an offline evaluation target
and never appears in policy inputs or candidate metadata.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from activemap.data.navigation_map import (
    NavigationEvidence,
    NavigationMapEpisode,
    NavigationPose,
    write_navigation_map_jsonl,
)

UNKNOWN = np.float32(0.5)


def parse_trajectory_spec(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError("trajectory must use EPISODE_ID=TRAJECTORY_DIRECTORY")
    episode_id, raw_path = raw.split("=", 1)
    if not episode_id:
        raise ValueError("trajectory episode ID must not be empty")
    path = Path(raw_path)
    if not (path / "summary.json").is_file():
        raise FileNotFoundError(path / "summary.json")
    return episode_id, path


def yaw_from_wxyz(rotation: list[float]) -> float:
    """Return the world-up yaw for Habitat's ``[w, x, y, z]`` quaternion."""
    if len(rotation) != 4:
        raise ValueError("rotation must contain [w, x, y, z]")
    w, x, y, z = (float(value) for value in rotation)
    return math.atan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z))


def _world_bounds(
    trajectories: list[dict[str, object]], *, max_range_m: float, resolution_m: float
) -> tuple[float, float, float, float, tuple[int, int]]:
    positions = [
        frame["agent_position"]
        for trajectory in trajectories
        for frame in trajectory["frames"]
        if isinstance(frame, dict)
    ]
    if not positions:
        raise ValueError("trajectory summaries contain no frames")
    world_x = [float(position[0]) for position in positions]
    world_z = [float(position[2]) for position in positions]
    x_min, x_max = min(world_x) - max_range_m, max(world_x) + max_range_m
    z_min, z_max = min(world_z) - max_range_m, max(world_z) + max_range_m
    width = int(math.ceil((x_max - x_min) / resolution_m)) + 1
    height = int(math.ceil((z_max - z_min) / resolution_m)) + 1
    return x_min, x_max, z_min, z_max, (height, width)


def _grid_index(
    x: float,
    z: float,
    *,
    x_min: float,
    z_max: float,
    resolution_m: float,
) -> tuple[int, int]:
    return round((z_max - z) / resolution_m), round((x - x_min) / resolution_m)


def _inside(row: int, col: int, shape: tuple[int, int]) -> bool:
    return 0 <= row < shape[0] and 0 <= col < shape[1]


def _write_ray(
    grid: np.ndarray,
    *,
    start: tuple[int, int],
    endpoint: tuple[int, int],
) -> None:
    distance = max(abs(endpoint[0] - start[0]), abs(endpoint[1] - start[1]))
    if distance == 0:
        if _inside(*start, grid.shape):
            grid[start] = 1.0
        return
    for index in range(distance):
        fraction = index / distance
        row = round(start[0] + fraction * (endpoint[0] - start[0]))
        col = round(start[1] + fraction * (endpoint[1] - start[1]))
        if _inside(row, col, grid.shape):
            grid[row, col] = 0.0
    if _inside(*endpoint, grid.shape):
        grid[endpoint] = 1.0


def project_depth_to_occupancy(
    depth: np.ndarray,
    *,
    position: list[float],
    rotation_wxyz: list[float],
    shape: tuple[int, int],
    x_min: float,
    z_max: float,
    resolution_m: float,
    horizontal_fov_degrees: float,
    max_range_m: float,
    ray_stride: int,
) -> np.ndarray:
    """Project center-row depth rays into an occupancy grid in world coordinates."""
    if depth.ndim != 2:
        raise ValueError(f"depth must be 2D, got {depth.shape}")
    if len(position) != 3:
        raise ValueError("agent_position must contain [x, y, z]")
    if ray_stride < 1:
        raise ValueError("ray_stride must be positive")
    occupancy = np.full(shape, UNKNOWN, dtype=np.float32)
    origin_x, origin_z = float(position[0]), float(position[2])
    start = _grid_index(
        origin_x,
        origin_z,
        x_min=x_min,
        z_max=z_max,
        resolution_m=resolution_m,
    )
    yaw = yaw_from_wxyz(rotation_wxyz)
    row = depth.shape[0] // 2
    center = (depth.shape[1] - 1) / 2.0
    focal = depth.shape[1] / (2.0 * math.tan(math.radians(horizontal_fov_degrees) / 2.0))
    for col in range(0, depth.shape[1], ray_stride):
        distance = float(depth[row, col])
        if not math.isfinite(distance) or distance <= 0.05 or distance > max_range_m:
            continue
        angle = math.atan2(col - center, focal)
        local_x = distance * math.sin(angle)
        local_z = -distance * math.cos(angle)
        world_x = origin_x + math.cos(yaw) * local_x + math.sin(yaw) * local_z
        world_z = origin_z - math.sin(yaw) * local_x + math.cos(yaw) * local_z
        endpoint = _grid_index(
            world_x,
            world_z,
            x_min=x_min,
            z_max=z_max,
            resolution_m=resolution_m,
        )
        _write_ray(occupancy, start=start, endpoint=endpoint)
    return occupancy


def fuse_observations(observations: list[np.ndarray]) -> np.ndarray:
    if not observations:
        raise ValueError("at least one observation is required")
    fused = np.full(observations[0].shape, UNKNOWN, dtype=np.float32)
    for observation in observations:
        if observation.shape != fused.shape:
            raise ValueError("occupancy observation shapes must match")
        known = observation != UNKNOWN
        fused[known] = observation[known]
    return fused


def corrupt_candidate_observation(
    observation: np.ndarray,
    *,
    error_rate: float,
    seed: int,
) -> np.ndarray:
    """Inject declared binary occupancy noise into an acquired observation only."""
    if not 0.0 <= error_rate < 1.0:
        raise ValueError("candidate_corruption_rate must be in [0, 1)")
    corrupted = observation.copy()
    if error_rate == 0.0:
        return corrupted
    generator = np.random.default_rng(seed)
    known = corrupted != UNKNOWN
    flips = known & (generator.random(corrupted.shape) < error_rate)
    corrupted[flips] = 1.0 - corrupted[flips]
    return corrupted


def _load_summary(trajectory_dir: Path) -> dict[str, object]:
    summary = json.loads((trajectory_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("schema_version") not in {
        "activemap-habitat-rgbd-smoke-v1",
        "activemap-habitat-rgbd-smoke-v2",
    }:
        raise ValueError(f"unsupported trajectory summary: {trajectory_dir}")
    frames = summary.get("frames")
    if not isinstance(frames, list) or len(frames) < 2:
        raise ValueError(f"trajectory needs initial plus candidate views: {trajectory_dir}")
    return summary


def materialize_episodes(
    *,
    trajectories: list[tuple[str, Path]],
    output: Path,
    resolution_m: float,
    max_range_m: float,
    horizontal_fov_degrees: float,
    ray_stride: int,
    budget: float,
    split: str = "val",
    candidate_corruption_rate: float = 0.0,
    corruption_seed: int = 20260830,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    if resolution_m <= 0.0 or max_range_m <= 0.0 or budget <= 0.0:
        raise ValueError("resolution_m, max_range_m, and budget must be positive")
    if split not in {"train", "val", "test"}:
        raise ValueError(f"unsupported split: {split}")
    if not 0.0 <= candidate_corruption_rate < 1.0:
        raise ValueError("candidate_corruption_rate must be in [0, 1)")

    raw_summaries = [(episode_id, path, _load_summary(path)) for episode_id, path in trajectories]
    reads_test_assets = split == "test"
    output.mkdir(parents=True)
    episodes: list[NavigationMapEpisode] = []
    for episode_id, trajectory_dir, summary in raw_summaries:
        x_min, _, _, z_max, shape = _world_bounds(
            [summary], max_range_m=max_range_m, resolution_m=resolution_m
        )
        frames = summary["frames"]
        assert isinstance(frames, list)
        projections: list[np.ndarray] = []
        for frame in frames:
            assert isinstance(frame, dict)
            depth_path = Path(str(frame["depth_path"]))
            projections.append(
                project_depth_to_occupancy(
                    np.load(depth_path),
                    position=[float(value) for value in frame["agent_position"]],
                    rotation_wxyz=[float(value) for value in frame["agent_rotation_wxyz"]],
                    shape=shape,
                    x_min=x_min,
                    z_max=z_max,
                    resolution_m=resolution_m,
                    horizontal_fov_degrees=horizontal_fov_degrees,
                    max_range_m=max_range_m,
                    ray_stride=ray_stride,
                )
            )
        reference = fuse_observations(projections)
        valid_space = reference != UNKNOWN
        episode_root = output / "episodes" / episode_id
        evidence_root = episode_root / "evidence"
        evidence_root.mkdir(parents=True)
        initial_path = episode_root / "initial_occupancy.npy"
        target_path = episode_root / "reference_fused_occupancy.npy"
        valid_path = episode_root / "reference_valid_space.npy"
        np.save(initial_path, projections[0])
        np.save(target_path, reference)
        np.save(valid_path, valid_space)

        initial_frame = frames[0]
        assert isinstance(initial_frame, dict)
        initial_rgb_path = str(initial_frame.get("rgb_path", "")) or None
        initial_depth_path = str(initial_frame.get("depth_path", "")) or None
        if initial_rgb_path is not None and not Path(initial_rgb_path).is_file():
            raise FileNotFoundError(initial_rgb_path)
        if initial_depth_path is not None and not Path(initial_depth_path).is_file():
            raise FileNotFoundError(initial_depth_path)
        initial_position = [float(value) for value in initial_frame["agent_position"]]
        initial_rotation = [float(value) for value in initial_frame["agent_rotation_wxyz"]]
        start_pose = NavigationPose(
            x=initial_position[0],
            y=initial_position[2],
            yaw=yaw_from_wxyz(initial_rotation),
        )
        evidence: list[NavigationEvidence] = []
        footprint = int(math.ceil(max_range_m / resolution_m))
        for index, (frame, observation) in enumerate(
            zip(frames[1:], projections[1:], strict=True), start=1
        ):
            assert isinstance(frame, dict)
            rgb_path = str(frame.get("rgb_path", "")) or None
            depth_path = str(frame.get("depth_path", "")) or None
            if rgb_path is not None and not Path(rgb_path).is_file():
                raise FileNotFoundError(rgb_path)
            if depth_path is not None and not Path(depth_path).is_file():
                raise FileNotFoundError(depth_path)
            position = [float(value) for value in frame["agent_position"]]
            rotation = [float(value) for value in frame["agent_rotation_wxyz"]]
            pose = NavigationPose(
                x=position[0],
                y=position[2],
                yaw=yaw_from_wxyz(rotation),
            )
            observation_path = evidence_root / f"rgbd_view_{index:02d}.npy"
            observation_seed = corruption_seed + sum(ord(char) for char in episode_id) + index
            np.save(
                observation_path,
                corrupt_candidate_observation(
                    observation,
                    error_rate=candidate_corruption_rate,
                    seed=observation_seed,
                ),
            )
            motion_cost = max(math.hypot(pose.x - start_pose.x, pose.y - start_pose.y), 0.05)
            evidence.append(
                NavigationEvidence(
                    evidence_id=f"{episode_id}-view-{index:02d}",
                    modality="camera",
                    path=str(observation_path.resolve()),
                    cost=motion_cost,
                    timestamp=index,
                    pose=pose,
                    footprint_radius_pixels=footprint,
                    rgb_path=rgb_path,
                    depth_path=depth_path,
                )
            )
        episodes.append(
            NavigationMapEpisode(
                schema_version="activemap-navigation-map-episode-v1",
                episode_id=episode_id,
                dataset="habitat_test_scenes",
                domain="indoor",
                split=split,
                map_representation="occupancy_grid",
                initial_map_path=str(initial_path.resolve()),
                target_map_path=str(target_path.resolve()),
                start_pose=start_pose,
                initial_rgb_path=initial_rgb_path,
                initial_depth_path=initial_depth_path,
                evidence=evidence,
                budget=budget,
                test_assets_read=reads_test_assets,
                metadata={
                    "valid_space_path": str(valid_path.resolve()),
                    "pixels_per_meter": 1.0 / resolution_m,
                    "grid_x_min": x_min,
                    "grid_z_max": z_max,
                    "occupancy_encoding": "free=0,unknown=0.5,occupied=1",
                    "observation_model": "habitat_center_row_depth_raycast_v1",
                    "reference_map": "offline_fusion_of_recorded_rgbd_views_v1",
                    "candidate_corruption_rate": candidate_corruption_rate,
                    "candidate_corruption_seed": corruption_seed,
                    "source_trajectory": str(trajectory_dir.resolve()),
                    "candidate_count": len(evidence),
                    "depth_resolution_m": resolution_m,
                    "max_range_m": max_range_m,
                    "horizontal_fov_degrees": horizontal_fov_degrees,
                    "ray_stride": ray_stride,
                    "development_only": True,
                    "test_assets_read": reads_test_assets,
                },
            )
        )
    manifest = output / "manifests" / f"{split}.jsonl"
    write_navigation_map_jsonl(episodes, manifest)
    result = {
        "schema_version": "activemap-habitat-rgbd-navigation-development-v1",
        "output": str(output.resolve()),
        "episode_count": len(episodes),
        "split_counts": {"train": 0, "val": 0, "test": 0} | {split: len(episodes)},
        "reference_map": "offline_fusion_of_recorded_rgbd_views_v1",
        "policy_observation_access": "after ACQUIRE only",
        "test_assets_read": reads_test_assets,
        "resolution_m": resolution_m,
        "max_range_m": max_range_m,
        "horizontal_fov_degrees": horizontal_fov_degrees,
        "ray_stride": ray_stride,
        "budget": budget,
        "candidate_corruption_rate": candidate_corruption_rate,
        "candidate_corruption_seed": corruption_seed,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new materialized episode output directory")
    parser.add_argument("--trajectory", action="append", required=True)
    parser.add_argument("--resolution-m", type=float, default=0.05)
    parser.add_argument("--max-range-m", type=float, default=5.0)
    parser.add_argument("--horizontal-fov-degrees", type=float, default=90.0)
    parser.add_argument("--ray-stride", type=int, default=4)
    parser.add_argument("--budget", type=float, default=8.0)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--candidate-corruption-rate", type=float, default=0.0)
    parser.add_argument("--corruption-seed", type=int, default=20260830)
    args = parser.parse_args()
    trajectories = [parse_trajectory_spec(raw) for raw in args.trajectory]
    if len({episode_id for episode_id, _ in trajectories}) != len(trajectories):
        parser.error("trajectory episode IDs must be unique")
    result = materialize_episodes(
        trajectories=trajectories,
        output=args.output.resolve(),
        resolution_m=args.resolution_m,
        max_range_m=args.max_range_m,
        horizontal_fov_degrees=args.horizontal_fov_degrees,
        ray_stride=args.ray_stride,
        budget=args.budget,
        split=args.split,
        candidate_corruption_rate=args.candidate_corruption_rate,
        corruption_seed=args.corruption_seed,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
