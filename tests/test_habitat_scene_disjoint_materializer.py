from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def load_suite_materializer():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "materialize_habitat_scene_disjoint_suite.py"
    )
    spec = importlib.util.spec_from_file_location("habitat_scene_suite_materializer", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_group_suite_trajectories_rejects_scene_leakage(tmp_path: Path) -> None:
    module = load_suite_materializer()
    rows = []
    for split in ("train", "val", "test"):
        trajectory = tmp_path / split
        trajectory.mkdir()
        (trajectory / "summary.json").write_text("{}", encoding="utf-8")
        rows.append(
            {
                "split": split,
                "scene": "same-scene",
                "seed": len(rows),
                "trajectory": str(trajectory),
            }
        )
    payload = {
        "schema_version": "activemap-habitat-scene-disjoint-suite-v1",
        "development_only": True,
        "scene_disjoint": True,
        "trajectories": rows,
    }

    with pytest.raises(ValueError, match="scene leakage"):
        module.group_suite_trajectories(json.loads(json.dumps(payload)))


def test_group_suite_trajectories_does_not_open_excluded_test_assets(tmp_path: Path) -> None:
    module = load_suite_materializer()
    rows = []
    for index, split in enumerate(("train", "val")):
        trajectory = tmp_path / split
        trajectory.mkdir()
        (trajectory / "summary.json").write_text("{}", encoding="utf-8")
        rows.append(
            {
                "split": split,
                "scene": f"scene-{split}",
                "seed": index,
                "trajectory": str(trajectory),
            }
        )
    rows.append(
        {
            "split": "test",
            "scene": "scene-test",
            "seed": 2,
            "trajectory": str(tmp_path / "locked-test-does-not-exist"),
        }
    )
    payload = {
        "schema_version": "activemap-habitat-scene-disjoint-suite-v1",
        "development_only": True,
        "scene_disjoint": True,
        "trajectories": rows,
    }

    grouped = module.group_suite_trajectories(
        json.loads(json.dumps(payload)), selected_splits=("train", "val")
    )

    assert tuple(grouped) == ("train", "val")
    assert all(len(rows) == 1 for rows in grouped.values())
