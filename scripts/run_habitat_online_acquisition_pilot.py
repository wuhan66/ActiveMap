#!/usr/bin/env python3
"""Run online ActiveMap-style RGB-D acquisition on a real Habitat trajectory.

Motion is held fixed by Habitat's goal follower. The evaluated policy controls
only whether the next RGB-D observation is acquired and committed, using the
current map, pose, and heading. This isolates active evidence acquisition from
low-level collision avoidance.
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
AcquisitionPolicy = Literal["acquire_all", "unknown_gate", "novelty_gate", "stop_after_initial"]


def load_online_helpers() -> Any:
    script = Path(__file__).with_name("run_habitat_online_mapping_pilot.py")
    spec = importlib.util.spec_from_file_location("habitat_online_mapping", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load online mapping helpers from {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def should_acquire(
    policy: AcquisitionPolicy,
    *,
    unknown_score: int,
    min_unknown_score: int,
    novelty_score: int | None = None,
    min_novelty_score: int | None = None,
) -> bool:
    """Apply a pre-observation tool-acquisition decision."""
    if min_unknown_score < 0:
        raise ValueError("min_unknown_score must be non-negative")
    if policy == "acquire_all":
        return True
    if policy == "unknown_gate":
        return unknown_score >= min_unknown_score
    if policy == "novelty_gate":
        if novelty_score is None or min_novelty_score is None:
            raise ValueError("novelty_gate requires novelty_score and min_novelty_score")
        if min_novelty_score < 0:
            raise ValueError("min_novelty_score must be non-negative")
        return novelty_score >= min_novelty_score
    if policy == "stop_after_initial":
        return False
    raise ValueError(f"unsupported acquisition policy: {policy}")


def prospective_unknown_cells(
    committed_map: np.ndarray,
    *,
    x: float,
    z: float,
    yaw: float,
    x_min: float,
    z_max: float,
    pixels_per_meter: float,
    max_range_m: float,
) -> set[tuple[int, int]]:
    """List pre-observation unknown grid cells in the prospective camera cone."""
    cells: set[tuple[int, int]] = set()
    for angle_offset in (-0.35, 0.0, 0.35):
        for distance_m in np.arange(0.5, max_range_m + 1e-6, 0.5):
            point_x = x - math.sin(yaw + angle_offset) * float(distance_m)
            point_z = z - math.cos(yaw + angle_offset) * float(distance_m)
            row = round((z_max - point_z) * pixels_per_meter)
            col = round((point_x - x_min) * pixels_per_meter)
            if (
                0 <= row < committed_map.shape[0]
                and 0 <= col < committed_map.shape[1]
                and committed_map[row, col] == UNKNOWN
            ):
                cells.add((row, col))
    return cells


def reference_map_metrics(
    committed: np.ndarray, reference: np.ndarray | None
) -> dict[str, float | int | None]:
    """Compute class-aware occupancy metrics against an all-acquired reference."""
    empty = {
        "final_reference_map_quality": None,
        "reference_known_coverage": None,
        "reference_unknown_rate": None,
        "occupied_iou": None,
        "free_iou": None,
        "balanced_iou": None,
        "obstacle_precision": None,
        "obstacle_recall": None,
        "false_free_rate": None,
        "false_obstacle_rate": None,
        "reference_known_cells": None,
        "reference_covered_cells": None,
    }
    if reference is None:
        return empty
    if committed.shape != reference.shape:
        raise ValueError("committed/reference occupancy shapes must match")
    valid = reference != UNKNOWN
    if not np.any(valid):
        return empty

    predicted_known = committed != UNKNOWN
    reference_occupied = valid & (reference == 1.0)
    reference_free = valid & (reference == 0.0)
    predicted_occupied = valid & (committed == 1.0)
    predicted_free = valid & (committed == 0.0)

    def ratio(numerator: np.ndarray, denominator: np.ndarray) -> float | None:
        count = int(np.count_nonzero(denominator))
        return float(np.count_nonzero(numerator) / count) if count else None

    def iou(predicted: np.ndarray, target: np.ndarray) -> float | None:
        union = predicted | target
        return ratio(predicted & target, union)

    occupied_iou = iou(predicted_occupied, reference_occupied)
    free_iou = iou(predicted_free, reference_free)
    class_ious = [value for value in (occupied_iou, free_iou) if value is not None]
    reference_count = int(np.count_nonzero(valid))
    covered_count = int(np.count_nonzero(valid & predicted_known))
    return {
        "final_reference_map_quality": float(np.mean(committed[valid] == reference[valid])),
        "reference_known_coverage": float(covered_count / reference_count),
        "reference_unknown_rate": float(1.0 - covered_count / reference_count),
        "occupied_iou": occupied_iou,
        "free_iou": free_iou,
        "balanced_iou": float(np.mean(class_ious)) if class_ious else None,
        "obstacle_precision": ratio(predicted_occupied & reference_occupied, predicted_occupied),
        "obstacle_recall": ratio(predicted_occupied & reference_occupied, reference_occupied),
        "false_free_rate": ratio(predicted_free & reference_occupied, reference_occupied),
        "false_obstacle_rate": ratio(predicted_occupied & reference_free, reference_free),
        "reference_known_cells": reference_count,
        "reference_covered_cells": covered_count,
    }


def reference_map_quality(committed: np.ndarray, reference: np.ndarray | None) -> float | None:
    """Backward-compatible overall agreement helper."""
    value = reference_map_metrics(committed, reference)["final_reference_map_quality"]
    return float(value) if isinstance(value, int | float) else None


def _merge(committed: np.ndarray, observation: np.ndarray) -> None:
    known = observation != UNKNOWN
    committed[known] = observation[known]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new online acquisition rollout directory")
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument(
        "--policy",
        choices=("acquire_all", "unknown_gate", "novelty_gate", "stop_after_initial"),
        required=True,
    )
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
    parser.add_argument("--min-unknown-score", type=int, default=8)
    parser.add_argument(
        "--min-novelty-score",
        type=int,
        default=8,
        help="minimum previously unrequested unknown cells for novelty_gate",
    )
    parser.add_argument("--save-rgb", action="store_true")
    parser.add_argument(
        "--reference-occupancy",
        type=Path,
        help="optional all-acquired map from the same scene/seed/path, evaluation only",
    )
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.scene.is_file():
        raise FileNotFoundError(args.scene)
    if args.max_steps < 1 or args.resolution < 32 or args.min_novelty_score < 0:
        raise ValueError("max_steps must be positive and resolution must be at least 32")
    if args.reference_occupancy is not None and not args.reference_occupancy.is_file():
        raise FileNotFoundError(args.reference_occupancy)

    online = load_online_helpers()
    habitat = online.load_habitat_helpers()
    projection = online.load_projection_helpers()
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
        goal = online._sample_goal(
            pathfinder=pathfinder,
            start=start,
            min_distance_m=args.goal_min_distance_m,
            max_distance_m=args.goal_max_distance_m,
        )
        initial_geodesic = online._path_distance(pathfinder, start, goal)
        x_min, z_max, shape = online.grid_spec(
            start_position=start,
            radius_m=args.grid_radius_m,
            resolution_m=args.grid_resolution_m,
        )
        pixels_per_meter = 1.0 / args.grid_resolution_m
        committed = np.full(shape, UNKNOWN, dtype=np.float32)
        reference = (
            np.load(args.reference_occupancy).astype(np.float32)
            if args.reference_occupancy is not None
            else None
        )
        import habitat_sim

        follower = habitat_sim.GreedyGeodesicFollower(pathfinder, agent, args.goal_radius_m)
        trace: list[dict[str, object]] = []
        path_length = 0.0
        collisions = 0
        sensor_calls = 0
        acquired_cone_cells: set[tuple[int, int]] = set()

        # The initial state is always acquired to provide a real committed map.
        initial_state = agent.get_state()
        initial_yaw = projection.yaw_from_wxyz(habitat.rotation_to_wxyz(initial_state.rotation))
        acquired_cone_cells.update(
            prospective_unknown_cells(
                committed,
                x=float(initial_state.position[0]),
                z=float(initial_state.position[2]),
                yaw=initial_yaw,
                x_min=x_min,
                z_max=z_max,
                pixels_per_meter=pixels_per_meter,
                max_range_m=args.max_range_m,
            )
        )
        initial_observation = simulator.get_sensor_observations()
        initial_map = projection.project_depth_to_occupancy(
            np.asarray(initial_observation["depth"], dtype=np.float32),
            position=[float(value) for value in initial_state.position],
            rotation_wxyz=habitat.rotation_to_wxyz(initial_state.rotation),
            shape=shape,
            x_min=x_min,
            z_max=z_max,
            resolution_m=args.grid_resolution_m,
            horizontal_fov_degrees=90.0,
            max_range_m=args.max_range_m,
            ray_stride=args.ray_stride,
        )
        _merge(committed, initial_map)
        sensor_calls = 1
        initial_known_cells = int(np.count_nonzero(committed != UNKNOWN))
        if args.save_rgb:
            online._save_rgb(
                args.output / "rgb_initial.png", np.asarray(initial_observation["rgb"])
            )

        for step in range(args.max_steps):
            before = np.asarray(agent.get_state().position, dtype=np.float32)
            action = follower.next_action_along(goal)
            if action is None:
                break
            simulator.step(action)
            state = agent.get_state()
            position = np.asarray(state.position, dtype=np.float32)
            transition = float(np.linalg.norm(position - before))
            collision = action == "move_forward" and transition < 0.10
            collisions += int(collision)
            path_length += transition
            yaw = projection.yaw_from_wxyz(habitat.rotation_to_wxyz(state.rotation))
            candidate_cells = prospective_unknown_cells(
                committed,
                x=float(position[0]),
                z=float(position[2]),
                yaw=yaw,
                x_min=x_min,
                z_max=z_max,
                pixels_per_meter=pixels_per_meter,
                max_range_m=args.max_range_m,
            )
            score = len(candidate_cells)
            novelty_score = len(candidate_cells - acquired_cone_cells)
            acquire = should_acquire(
                args.policy,
                unknown_score=score,
                min_unknown_score=args.min_unknown_score,
                novelty_score=novelty_score,
                min_novelty_score=args.min_novelty_score,
            )
            if acquire:
                known_before = int(np.count_nonzero(committed != UNKNOWN))
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
                _merge(committed, local_map)
                known_gain = int(np.count_nonzero(committed != UNKNOWN)) - known_before
                acquired_cone_cells.update(candidate_cells)
                sensor_calls += 1
                if args.save_rgb:
                    online._save_rgb(
                        args.output / f"rgb_step_{step + 1:03d}.png",
                        np.asarray(observation["rgb"]),
                    )
            else:
                known_gain = 0
            trace.append(
                {
                    "step": step + 1,
                    "action": action,
                    "position": [float(value) for value in position],
                    "transition_distance_m": transition,
                    "collision_proxy": collision,
                    "acquisition_score": score,
                    "novelty_score": novelty_score,
                    "acquired": acquire,
                    "known_cell_gain": known_gain,
                    "known_cell_fraction": float(np.mean(committed != UNKNOWN)),
                }
            )

        final_position = np.asarray(agent.get_state().position, dtype=np.float32)
        np.save(args.output / "committed_occupancy.npy", committed)
        final_known_cells = int(np.count_nonzero(committed != UNKNOWN))
        post_initial_acquisitions = sensor_calls - 1
        post_initial_known_gain = final_known_cells - initial_known_cells
        reference_metrics = reference_map_metrics(committed, reference)
        reference_covered_cells = reference_metrics["reference_covered_cells"]
        summary = {
            "schema_version": "activemap-habitat-online-acquisition-pilot-v1",
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
            "acquisition_protocol": (
                "initial RGB-D is acquired; later RGB-D is read only after an "
                "executed action and a map/proprioception-only acquisition decision"
            ),
            "motion_controller": "Habitat GreedyGeodesicFollower oracle control",
            "collision_proxy": "move_forward transition below 0.10m",
            "min_unknown_score": args.min_unknown_score,
            "min_novelty_score": args.min_novelty_score,
            "sensor_calls": sensor_calls,
            "post_initial_acquisitions": post_initial_acquisitions,
            "rgb_artifacts_saved": args.save_rgb,
            "collision_count": collisions,
            "initial_known_cell_fraction": float(initial_known_cells / committed.size),
            "final_known_cell_fraction": float(np.mean(committed != UNKNOWN)),
            "post_initial_known_cell_gain": post_initial_known_gain,
            "new_known_cells_per_post_initial_acquisition": float(
                post_initial_known_gain / post_initial_acquisitions
            )
            if post_initial_acquisitions
            else 0.0,
            "reference_covered_cells_per_sensor_call": float(reference_covered_cells / sensor_calls)
            if isinstance(reference_covered_cells, int)
            else None,
            **reference_metrics,
            "trace": trace,
            **online.navigation_metrics(
                initial_geodesic_m=initial_geodesic,
                final_geodesic_m=online._path_distance(pathfinder, final_position, goal),
                path_length_m=path_length,
                goal_radius_m=args.goal_radius_m,
            ),
        }
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        simulator.close()
    print(f"Wrote online Habitat acquisition rollout to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
