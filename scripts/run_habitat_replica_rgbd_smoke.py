#!/usr/bin/env python3
"""Record a short, headless RGB-D navigation trajectory in a public Habitat scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_actions(raw_actions: str) -> tuple[str, ...]:
    actions = tuple(action.strip() for action in raw_actions.split(",") if action.strip())
    if not actions:
        raise ValueError("at least one action is required")
    return actions


def random_actions(*, count: int, seed: int) -> tuple[str, ...]:
    """Create a deterministic, navigation-compatible action sequence."""
    if count < 1:
        raise ValueError("random action count must be positive")
    generator = np.random.default_rng(seed)
    action_space = np.asarray(("move_forward", "turn_left", "turn_right"))
    return tuple(str(action) for action in generator.choice(action_space, size=count))


def _as_float_list(value: Any) -> list[float]:
    try:
        return [float(item) for item in value]
    except TypeError:
        return []


def rotation_to_wxyz(rotation: Any) -> list[float]:
    """Serialize Habitat's numpy-quaternion pose without adding a new dependency."""
    return [float(rotation.w), float(rotation.x), float(rotation.y), float(rotation.z)]


def build_simulator(
    *, scene_config: Path | None, scene_id: str, gpu_id: int, resolution: int
) -> Any:
    import habitat_sim

    simulator_config = habitat_sim.SimulatorConfiguration()
    if scene_config is not None:
        simulator_config.scene_dataset_config_file = str(scene_config)
    simulator_config.scene_id = scene_id
    simulator_config.gpu_device_id = gpu_id
    simulator_config.enable_physics = True

    agent_config = habitat_sim.agent.AgentConfiguration()
    sensor_specs = []
    for uuid, sensor_type in (
        ("rgb", habitat_sim.SensorType.COLOR),
        ("depth", habitat_sim.SensorType.DEPTH),
    ):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = sensor_type
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        spec.resolution = [resolution, resolution]
        spec.position = [0.0, 1.5, 0.0]
        spec.hfov = 90.0
        sensor_specs.append(spec)
    agent_config.sensor_specifications = sensor_specs
    return habitat_sim.Simulator(habitat_sim.Configuration(simulator_config, [agent_config]))


def initialize_agent(*, simulator: Any, start_mode: str, seed: int | None) -> Any:
    """Initialize the default agent, optionally at a seeded navigable point."""
    agent = simulator.initialize_agent(0)
    if start_mode == "scene-default":
        return agent
    if start_mode != "random-navigable":
        raise ValueError(f"unsupported start mode: {start_mode}")
    if seed is None:
        raise ValueError("--seed is required with --start-mode=random-navigable")
    if not simulator.pathfinder.is_loaded:
        raise RuntimeError("scene navigation mesh is not loaded")
    simulator.pathfinder.seed(seed)
    state = agent.get_state()
    state.position = simulator.pathfinder.get_random_navigable_point()
    agent.set_state(state, reset_sensors=True)
    return agent


def write_observation(
    *,
    output_dir: Path,
    step: int,
    observation: dict[str, np.ndarray],
    position: Any,
    rotation: Any,
    incoming_action: str | None,
    transition_distance_m: float,
) -> dict[str, object]:
    from PIL import Image

    rgb = np.asarray(observation["rgb"])
    if rgb.ndim != 3 or rgb.shape[-1] not in (3, 4):
        raise ValueError(f"expected RGB(A) observation, got {rgb.shape}")
    rgb_path = output_dir / f"rgb_{step:03d}.png"
    Image.fromarray(rgb[..., :3]).save(rgb_path)

    depth = np.asarray(observation["depth"], dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"expected depth observation, got {depth.shape}")
    depth_path = output_dir / f"depth_{step:03d}.npy"
    np.save(depth_path, depth)
    return {
        "step": step,
        "rgb_path": str(rgb_path),
        "depth_path": str(depth_path),
        "agent_position": _as_float_list(position),
        "agent_rotation_wxyz": rotation_to_wxyz(rotation),
        "incoming_action": incoming_action,
        "transition_distance_m": float(transition_distance_m),
        "depth_min": float(np.nanmin(depth)),
        "depth_max": float(np.nanmax(depth)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new trajectory output directory")
    scene_group = parser.add_mutually_exclusive_group(required=True)
    scene_group.add_argument(
        "--scene-config",
        type=Path,
        help="ReplicaCAD scene-dataset configuration JSON",
    )
    scene_group.add_argument(
        "--scene",
        type=Path,
        help="standalone Habitat-compatible GLB scene",
    )
    parser.add_argument("--scene-id", default="apt_1", help="scene name when using --scene-config")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument(
        "--start-mode",
        choices=("scene-default", "random-navigable"),
        default="scene-default",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--actions",
        default="move_forward,turn_left,move_forward,turn_right,move_forward",
    )
    parser.add_argument(
        "--random-action-count",
        type=int,
        help="replace the action string with a seeded random sequence",
    )
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    if args.scene_config is not None and not args.scene_config.is_file():
        raise FileNotFoundError(args.scene_config)
    if args.scene is not None and not args.scene.is_file():
        raise FileNotFoundError(args.scene)
    if args.resolution < 32:
        raise ValueError("resolution must be at least 32")
    if args.random_action_count is not None:
        if args.seed is None:
            parser.error("--seed is required with --random-action-count")
        actions = random_actions(count=args.random_action_count, seed=args.seed)
    else:
        actions = parse_actions(args.actions)
    scene_config = args.scene_config.resolve() if args.scene_config is not None else None
    scene_id = args.scene_id if args.scene is None else str(args.scene.resolve())

    args.output.mkdir(parents=True)
    simulator = build_simulator(
        scene_config=scene_config,
        scene_id=scene_id,
        gpu_id=args.gpu_id,
        resolution=args.resolution,
    )
    try:
        agent = initialize_agent(
            simulator=simulator,
            start_mode=args.start_mode,
            seed=args.seed,
        )
        frames: list[dict[str, object]] = []
        previous_position: list[float] | None = None
        for step, action in enumerate((None, *actions)):
            state = agent.get_state()
            observation = simulator.get_sensor_observations()
            position = _as_float_list(state.position)
            transition_distance_m = (
                0.0
                if previous_position is None
                else float(np.linalg.norm(np.asarray(position) - np.asarray(previous_position)))
            )
            frames.append(
                write_observation(
                    output_dir=args.output,
                    step=step,
                    observation=observation,
                    position=position,
                    rotation=state.rotation,
                    incoming_action=None if step == 0 else actions[step - 1],
                    transition_distance_m=transition_distance_m,
                )
            )
            previous_position = position
            if action is not None:
                simulator.step(action)
        summary = {
            "schema_version": "activemap-habitat-rgbd-smoke-v2",
            "scene_config": str(scene_config) if scene_config is not None else None,
            "scene": str(args.scene.resolve()) if args.scene is not None else None,
            "scene_id": scene_id,
            "gpu_id": args.gpu_id,
            "resolution": args.resolution,
            "start_mode": args.start_mode,
            "seed": args.seed,
            "navmesh_loaded": bool(simulator.pathfinder.is_loaded),
            "actions": actions,
            "frames": frames,
            "test_assets_read": False,
        }
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        simulator.close()
    print(f"Wrote {len(frames)} Habitat RGB-D observations to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
