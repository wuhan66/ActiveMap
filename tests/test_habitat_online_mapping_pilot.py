from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def load_online_pilot():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_habitat_online_mapping_pilot.py"
    spec = importlib.util.spec_from_file_location("habitat_online_mapping_pilot", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_grid_spec_is_world_aligned_and_centered_on_start() -> None:
    module = load_online_pilot()
    x_min, z_max, shape = module.grid_spec(
        start_position=np.array([4.0, 0.0, 7.0], dtype=np.float32),
        radius_m=2.0,
        resolution_m=0.5,
    )

    assert (x_min, z_max, shape) == (2.0, 9.0, (9, 9))


def test_frontier_controller_avoids_a_known_obstacle_ahead() -> None:
    module = load_online_pilot()
    occupancy = np.zeros((25, 25), dtype=np.float32)
    occupancy[14, 12] = 1.0

    action = module.select_frontier_action(
        occupancy,
        x=0.0,
        z=0.0,
        yaw=0.0,
        x_min=-2.5,
        z_max=2.5,
        pixels_per_meter=5.0,
        max_range_m=2.0,
        turn_angle_radians=np.pi / 6.0,
        forward_step_m=0.25,
    )

    assert action in {"turn_left", "turn_right"}


def test_frontier_controller_prefers_forward_motion_when_scores_tie() -> None:
    module = load_online_pilot()
    occupancy = np.zeros((25, 25), dtype=np.float32)

    action = module.select_frontier_action(
        occupancy,
        x=0.0,
        z=0.0,
        yaw=0.0,
        x_min=-2.5,
        z_max=2.5,
        pixels_per_meter=5.0,
        max_range_m=2.0,
        turn_angle_radians=np.pi / 6.0,
        forward_step_m=0.25,
    )

    assert action == "move_forward"


def test_frontier_controller_advances_into_safe_unknown_space() -> None:
    module = load_online_pilot()
    occupancy = np.full((25, 25), 0.5, dtype=np.float32)

    action = module.select_frontier_action(
        occupancy,
        x=0.0,
        z=0.0,
        yaw=0.0,
        x_min=-2.5,
        z_max=2.5,
        pixels_per_meter=5.0,
        max_range_m=2.0,
        turn_angle_radians=np.pi / 6.0,
        forward_step_m=0.25,
        forward_clearance=1.0,
    )

    assert action == "move_forward"


def test_frontier_controller_uses_acquired_depth_to_avoid_near_collision() -> None:
    module = load_online_pilot()
    occupancy = np.full((25, 25), 0.5, dtype=np.float32)

    action = module.select_frontier_action(
        occupancy,
        x=0.0,
        z=0.0,
        yaw=0.0,
        x_min=-2.5,
        z_max=2.5,
        pixels_per_meter=5.0,
        max_range_m=2.0,
        turn_angle_radians=np.pi / 6.0,
        forward_step_m=0.25,
        forward_clearance=0.1,
        min_forward_clearance_m=0.125,
    )

    assert action in {"turn_left", "turn_right"}


def test_forward_clearance_uses_the_near_quantile_of_the_center_view() -> None:
    module = load_online_pilot()
    depth = np.full((8, 20), 2.0, dtype=np.float32)
    depth[4, 9] = 0.2

    assert module.forward_clearance_m(depth, central_fraction=0.2) < 1.0


def test_navigation_metrics_reports_success_and_spl() -> None:
    module = load_online_pilot()

    metrics = module.navigation_metrics(
        initial_geodesic_m=4.0,
        final_geodesic_m=0.1,
        path_length_m=5.0,
        goal_radius_m=0.25,
    )

    assert metrics["success"] is True
    assert metrics["spl"] == 0.8
