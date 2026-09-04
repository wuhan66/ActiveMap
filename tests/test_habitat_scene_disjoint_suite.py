from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def load_suite_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_habitat_scene_disjoint_suite.py"
    spec = importlib.util.spec_from_file_location("habitat_scene_disjoint_suite", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_gpu_ids_rejects_empty_and_duplicates() -> None:
    module = load_suite_module()

    assert module.parse_gpu_ids("1, 3") == (1, 3)
    with pytest.raises(ValueError, match="at least one"):
        module.parse_gpu_ids(" ")
    with pytest.raises(ValueError, match="unique"):
        module.parse_gpu_ids("1,1")


def test_build_jobs_keeps_scenes_disjoint_and_deterministic(tmp_path: Path) -> None:
    module = load_suite_module()
    scene_root = tmp_path / "scenes"
    scene_root.mkdir()
    for _, scene in module.SCENE_SPLITS:
        (scene_root / f"{scene}.glb").write_bytes(b"scene")

    jobs = module.build_jobs(
        output=tmp_path / "suite",
        scene_root=scene_root,
        episodes_per_scene=2,
        action_count=4,
        gpu_ids=(1, 3),
        base_seed=100,
        resolution=128,
    )

    assert len(jobs) == 6
    assert {job.split for job in jobs} == {"train", "val", "test"}
    assert len({(job.split, job.scene) for job in jobs}) == 3
    assert len({job.seed for job in jobs}) == len(jobs)
    assert {job.gpu_id for job in jobs} == {1, 3}
    assert all("--start-mode" in job.command for job in jobs)
