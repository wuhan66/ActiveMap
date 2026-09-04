#!/usr/bin/env python3
"""Execute a development-only online RGB-D mapping pilot in Habitat-Sim.

Each action is selected from the committed occupancy map, executed in the
simulator, and only then produces an RGB-D observation to acquire and commit.
The script intentionally separates a map-driven frontier controller from an
oracle-goal navigation control. It is not a benchmark training script.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Literal

import numpy as np

UNKNOWN = np.float32(0.5)
PolicyName = Literal["frontier", "goal_oracle", "random"]


def load_habitat_helpers() -> Any:
    script = Path(__file__).with_name("run_habitat_replica_rgbd_smoke.py")
    spec = importlib.util.spec_from_file_location("habitat_rgbd_smoke", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Habitat helpers from {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_projection_helpers() -> Any:
    script = Path(__file__).with_name("materialize_habitat_rgbd_navigation.py")
    spec = importlib.util.spec_from_file_location("habitat_rgbd_projection", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load occupancy projection helpers from {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def grid_spec(
    *, start_position: np.ndarray, radius_m: float, resolution_m: float
) -> tuple[float, float, tuple[int, int]]:
    """Return a fixed world-aligned occupancy grid centred on the start pose."""
    if radius_m <= 0.0 or resolution_m <= 0.0:
        raise ValueError("radius_m and resolution_m must be positive")
    side_cells = int(math.ceil(2.0 * radius_m / resolution_m)) + 1
    return (
        float(start_position[0] - radius_m),
        float(start_position[2] + radius_m),
        (side_cells, side_cells),
    )


def _grid_index(
    *, x: float, z: float, x_min: float, z_max: float, pixels_per_meter: float
) -> tuple[int, int]:
    return round((z_max - z) * pixels_per_meter), round((x - x_min) * pixels_per_meter)


def _heading_point(*, x: float, z: float, yaw: float, distance_m: float) -> tuple[float, float]:
    """World point in the forward camera direction used by the RGB-D projector."""
    return x - math.sin(yaw) * distance_m, z - math.cos(yaw) * distance_m


def unknown_visibility_score(
    committed_map: np.ndarray,
    *,
    x: float,
    z: float,
    yaw: float,
    x_min: float,
    z_max: float,
    pixels_per_meter: float,
    max_range_m: float,
) -> int:
    """Estimate unseen cells in a prospective camera cone without reading RGB-D."""
    cells: set[tuple[int, int]] = set()
    for angle_offset in (-0.35, 0.0, 0.35):
        for distance_m in np.arange(0.5, max_range_m + 1e-6, 0.5):
            point_x, point_z = _heading_point(
                x=x,
                z=z,
                yaw=yaw + angle_offset,
                distance_m=float(distance_m),
            )
            row, col = _grid_index(
                x=point_x,
                z=point_z,
                x_min=x_min,
                z_max=z_max,
                pixels_per_meter=pixels_per_meter,
            )
            if 0 <= row < committed_map.shape[0] and 0 <= col < committed_map.shape[1]:
                cells.add((row, col))
    return int(sum(committed_map[row, col] == UNKNOWN for row, col in cells))


def forward_clearance_m(depth: np.ndarray, *, central_fraction: float = 0.16) -> float:
    """Return a conservative near-field clearance from an acquired depth frame."""
    if depth.ndim != 2 or not 0.0 < central_fraction <= 1.0:
        raise ValueError("depth must be 2D and central_fraction must be in (0, 1]")
    row = depth.shape[0] // 2
    half_width = max(1, round(depth.shape[1] * central_fraction / 2.0))
    center = depth.shape[1] // 2
    values = np.asarray(depth[row, center - half_width : center + half_width], dtype=np.float32)
    valid = values[np.isfinite(values) & (values > 0.05)]
    return float(np.quantile(valid, 0.10)) if len(valid) else float("inf")


def select_frontier_action(
    committed_map: np.ndarray,
    *,
    x: float,
    z: float,
    yaw: float,
    x_min: float,
    z_max: float,
    pixels_per_meter: float,
    max_range_m: float,
    turn_angle_radians: float,
    forward_step_m: float,
    forward_clearance: float | None = None,
    min_forward_clearance_m: float = 0.125,
) -> str:
    """Choose a motion whose prospective view contains the most unknown cells."""
    candidate_yaws = {
        "move_forward": yaw,
        "turn_left": yaw + turn_angle_radians,
        "turn_right": yaw - turn_angle_radians,
    }
    scores = {
        action: unknown_visibility_score(
            committed_map,
            x=x,
            z=z,
            yaw=candidate_yaw,
            x_min=x_min,
            z_max=z_max,
            pixels_per_meter=pixels_per_meter,
            max_range_m=max_range_m,
        )
        for action, candidate_yaw in candidate_yaws.items()
    }
    ahead_x, ahead_z = _heading_point(x=x, z=z, yaw=yaw, distance_m=forward_step_m)
    ahead_row, ahead_col = _grid_index(
        x=ahead_x,
        z=ahead_z,
        x_min=x_min,
        z_max=z_max,
        pixels_per_meter=pixels_per_meter,
    )
    forward_safe = True
    if (
        0 <= ahead_row < committed_map.shape[0]
        and 0 <= ahead_col < committed_map.shape[1]
        and committed_map[ahead_row, ahead_col] == 1.0
    ):
        scores["move_forward"] = -1
        forward_safe = False
    if min_forward_clearance_m <= 0.0:
        raise ValueError("min_forward_clearance_m must be positive")
    if forward_clearance is not None and forward_clearance < min_forward_clearance_m:
        scores["move_forward"] = -1
        forward_safe = False
    if forward_safe and scores["move_forward"] > 0:
        return "move_forward"
    # When prospective coverage ties, keep moving through known-free space
    # rather than spinning in place and repeatedly acquiring the same view.
    action_order = ("move_forward", "turn_left", "turn_right")
    return max(action_order, key=lambda action: (scores[action], -action_order.index(action)))


def navigation_metrics(
    *,
    initial_geodesic_m: float,
    final_geodesic_m: float,
    path_length_m: float,
    goal_radius_m: float,
) -> dict[str, float | bool]:
    """Compute standard goal-conditioned success and SPL from executed motion."""
    if initial_geodesic_m < 0.0 or final_geodesic_m < 0.0 or path_length_m < 0.0:
        raise ValueError("navigation distances must be non-negative")
    if goal_radius_m <= 0.0:
        raise ValueError("goal_radius_m must be positive")
    success = final_geodesic_m <= goal_radius_m
    spl = float(success) * initial_geodesic_m / max(initial_geodesic_m, path_length_m, 1e-8)
    return {
        "success": success,
        "spl": spl,
        "initial_geodesic_m": initial_geodesic_m,
        "final_geodesic_m": final_geodesic_m,
        "executed_path_length_m": path_length_m,
    }


def _path_distance(pathfinder: Any, start: np.ndarray, goal: np.ndarray) -> float:
    import habitat_sim

    shortest_path = habitat_sim.ShortestPath()
    shortest_path.requested_start = start
    shortest_path.requested_end = goal
    if not pathfinder.find_path(shortest_path):
        return float("inf")
    return float(shortest_path.geodesic_distance)


def _sample_goal(
    *, pathfinder: Any, start: np.ndarray, min_distance_m: float, max_distance_m: float
) -> np.ndarray:
    for _ in range(256):
        goal = pathfinder.get_random_navigable_point()
        distance = _path_distance(pathfinder, start, goal)
        if min_distance_m <= distance <= max_distance_m:
            return np.asarray(goal, dtype=np.float32)
    raise RuntimeError("could not sample a reachable goal within the requested distance range")


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(np.asarray(rgb)[..., :3]).save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new online rollout directory")
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--policy", choices=("frontier", "goal_oracle", "random"), required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=48)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--grid-radius-m", type=float, default=8.0)
    parser.add_argument("--grid-resolution-m", type=float, default=0.1)
    parser.add_argument("--max-range-m", type=float, default=5.0)
    parser.add_argument("--ray-stride", type=int, default=4)
    parser.add_argument("--goal-min-distance-m", type=float, default=3.0)
    parser.add_argument("--goal-max-distance-m", type=float, default=6.0)
    parser.add_argument("--goal-radius-m", type=float, default=0.25)
    parser.add_argument("--min-forward-clearance-m", type=float, default=0.125)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.scene.is_file():
        raise FileNotFoundError(args.scene)
    if args.max_steps < 1 or args.resolution < 32:
        raise ValueError("max_steps must be positive and resolution must be at least 32")
    if args.min_forward_clearance_m <= 0.0:
        raise ValueError("min_forward_clearance_m must be positive")

    habitat = load_habitat_helpers()
    projection = load_projection_helpers()
    simulator = habitat.build_simulator(
        scene_config=None,
        scene_id=str(args.scene.resolve()),
        gpu_id=args.gpu_id,
        resolution=args.resolution,
    )
    args.output.mkdir(parents=True)
    try:
        agent = habitat.initialize_agent(
            simulator=simulator, start_mode="random-navigable", seed=args.seed
        )
        pathfinder = simulator.pathfinder
        start = np.asarray(agent.get_state().position, dtype=np.float32)
        pathfinder.seed(args.seed + 1)
        goal = _sample_goal(
            pathfinder=pathfinder,
            start=start,
            min_distance_m=args.goal_min_distance_m,
            max_distance_m=args.goal_max_distance_m,
        )
        initial_geodesic = _path_distance(pathfinder, start, goal)
        x_min, z_max, shape = grid_spec(
            start_position=start,
            radius_m=args.grid_radius_m,
            resolution_m=args.grid_resolution_m,
        )
        pixels_per_meter = 1.0 / args.grid_resolution_m
        committed = np.full(shape, UNKNOWN, dtype=np.float32)
        random_generator = np.random.default_rng(args.seed)
        follower = None
        if args.policy == "goal_oracle":
            import habitat_sim

            follower = habitat_sim.GreedyGeodesicFollower(pathfinder, agent, args.goal_radius_m)

        trace: list[dict[str, object]] = []
        path_length = 0.0
        collisions = 0
        for step in range(args.max_steps + 1):
            state = agent.get_state()
            position = np.asarray(state.position, dtype=np.float32)
            observation = simulator.get_sensor_observations()
            local_map = projection.project_depth_to_occupancy(
                np.asarray(observation["depth"], dtype=np.float32),
                position=[float(value) for value in position],
                rotation_wxyz=habitat.rotation_to_wxyz(state.rotation),
                shape=shape,
                x_min=x_min,
                z_max=z_max,
                resolution_m=args.grid_resolution_m,
                horizontal_fov_degrees=90.0,
                max_range_m=args.max_range_m,
                ray_stride=args.ray_stride,
            )
            known = local_map != UNKNOWN
            committed[known] = local_map[known]
            if step == 0:
                _save_rgb(args.output / "rgb_start.png", np.asarray(observation["rgb"]))
            if step == args.max_steps:
                break

            yaw = projection.yaw_from_wxyz(habitat.rotation_to_wxyz(state.rotation))
            current_clearance = forward_clearance_m(
                np.asarray(observation["depth"], dtype=np.float32)
            )
            if args.policy == "frontier":
                action = select_frontier_action(
                    committed,
                    x=float(position[0]),
                    z=float(position[2]),
                    yaw=yaw,
                    x_min=x_min,
                    z_max=z_max,
                    pixels_per_meter=pixels_per_meter,
                    max_range_m=args.max_range_m,
                    turn_angle_radians=math.radians(30.0),
                    forward_step_m=0.25,
                    forward_clearance=current_clearance,
                    min_forward_clearance_m=args.min_forward_clearance_m,
                )
            elif args.policy == "random":
                action = str(random_generator.choice(("move_forward", "turn_left", "turn_right")))
            else:
                assert follower is not None
                action = follower.next_action_along(goal)
                if action is None:
                    break

            simulator.step(action)
            next_position = np.asarray(agent.get_state().position, dtype=np.float32)
            transition = float(np.linalg.norm(next_position - position))
            collision = action == "move_forward" and transition < 0.10
            collisions += int(collision)
            path_length += transition
            trace.append(
                {
                    "step": step + 1,
                    "action": action,
                    "position": [float(value) for value in next_position],
                    "transition_distance_m": transition,
                    "collision_proxy": collision,
                    "pre_action_forward_clearance_m": current_clearance,
                    "known_cell_fraction": float(np.mean(committed != UNKNOWN)),
                }
            )

        final_state = agent.get_state()
        final_position = np.asarray(final_state.position, dtype=np.float32)
        _save_rgb(
            args.output / "rgb_final.png", np.asarray(simulator.get_sensor_observations()["rgb"])
        )
        np.save(args.output / "committed_occupancy.npy", committed)
        summary = {
            "schema_version": "activemap-habitat-online-mapping-pilot-v1",
            "development_only": True,
            "scene": str(args.scene.resolve()),
            "policy": args.policy,
            "seed": args.seed,
            "gpu_id": args.gpu_id,
            "start_position": [float(value) for value in start],
            "goal_position": [float(value) for value in goal],
            "grid": {
                "x_min": x_min,
                "z_max": z_max,
                "resolution_m": args.grid_resolution_m,
                "shape": list(shape),
            },
            "acquisition_protocol": "RGB-D is read only after each executed action",
            "commit_rule": "always_commit_observation",
            "min_forward_clearance_m": args.min_forward_clearance_m,
            "collision_proxy": "move_forward transition below 0.10m",
            "collision_count": collisions,
            "final_known_cell_fraction": float(np.mean(committed != UNKNOWN)),
            "trace": trace,
            **navigation_metrics(
                initial_geodesic_m=initial_geodesic,
                final_geodesic_m=_path_distance(pathfinder, final_position, goal),
                path_length_m=path_length,
                goal_radius_m=args.goal_radius_m,
            ),
        }
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        simulator.close()
    print(f"Wrote online Habitat mapping rollout to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
