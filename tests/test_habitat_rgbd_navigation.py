from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


def load_materializer():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "materialize_habitat_rgbd_navigation.py"
    )
    spec = importlib.util.spec_from_file_location("habitat_rgbd_navigation", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _summary(root: Path) -> None:
    for index in range(2):
        np.save(root / f"depth_{index:03d}.npy", np.full((8, 8), 1.0, dtype=np.float32))
    payload = {
        "schema_version": "activemap-habitat-rgbd-smoke-v1",
        "frames": [
            {
                "depth_path": str(root / "depth_000.npy"),
                "agent_position": [0.0, 0.0, 0.0],
                "agent_rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
            {
                "depth_path": str(root / "depth_001.npy"),
                "agent_position": [1.0, 0.0, 0.0],
                "agent_rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
        ],
    }
    (root / "summary.json").write_text(json.dumps(payload), encoding="utf-8")


def test_yaw_from_identity_quaternion_is_zero() -> None:
    module = load_materializer()
    assert module.yaw_from_wxyz([1.0, 0.0, 0.0, 0.0]) == 0.0


def test_candidate_corruption_is_seeded_and_preserves_unknown_cells() -> None:
    module = load_materializer()
    observation = np.array([[0.0, 1.0], [0.5, 0.0]], dtype=np.float32)

    first = module.corrupt_candidate_observation(observation, error_rate=0.5, seed=7)
    second = module.corrupt_candidate_observation(observation, error_rate=0.5, seed=7)

    assert np.array_equal(first, second)
    assert first[1, 0] == 0.5
    assert set(np.unique(first)).issubset({0.0, 0.5, 1.0})


def test_materializer_keeps_reference_out_of_candidate_metadata(tmp_path: Path) -> None:
    module = load_materializer()
    trajectory = tmp_path / "trajectory"
    trajectory.mkdir()
    _summary(trajectory)
    output = tmp_path / "episodes"

    result = module.materialize_episodes(
        trajectories=[("fixture", trajectory)],
        output=output,
        resolution_m=0.25,
        max_range_m=2.0,
        horizontal_fov_degrees=90.0,
        ray_stride=1,
        budget=4.0,
    )

    assert result["episode_count"] == 1
    record = json.loads((output / "manifests" / "val.jsonl").read_text())
    assert record["target_map_path"].endswith("reference_fused_occupancy.npy")
    assert all("reference" not in item["path"] for item in record["evidence"])
    assert record["test_assets_read"] is False
    assert isinstance(record["metadata"]["grid_x_min"], float)
    assert isinstance(record["metadata"]["grid_z_max"], float)
    assert record["metadata"]["candidate_corruption_rate"] == 0.0


def test_materializer_marks_held_out_test_assets(tmp_path: Path) -> None:
    module = load_materializer()
    trajectory = tmp_path / "trajectory"
    trajectory.mkdir()
    _summary(trajectory)

    module.materialize_episodes(
        trajectories=[("fixture", trajectory)],
        output=tmp_path / "episodes",
        resolution_m=0.25,
        max_range_m=2.0,
        horizontal_fov_degrees=90.0,
        ray_stride=1,
        budget=4.0,
        split="test",
    )

    record = json.loads((tmp_path / "episodes" / "manifests" / "test.jsonl").read_text())
    assert record["split"] == "test"
    assert record["test_assets_read"] is True
