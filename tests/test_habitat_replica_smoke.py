from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def load_smoke_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_habitat_replica_rgbd_smoke.py"
    spec = importlib.util.spec_from_file_location("habitat_replica_smoke", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_actions_accepts_stable_action_sequence() -> None:
    module = load_smoke_module()
    assert module.parse_actions("move_forward, turn_left,move_forward") == (
        "move_forward",
        "turn_left",
        "move_forward",
    )


def test_parse_actions_rejects_empty_sequence() -> None:
    module = load_smoke_module()
    with pytest.raises(ValueError, match="at least one action"):
        module.parse_actions(" , ")


def test_rotation_to_wxyz_preserves_habitat_quaternion_order() -> None:
    module = load_smoke_module()

    class Rotation:
        w = 1.0
        x = 0.1
        y = 0.2
        z = 0.3

    assert module.rotation_to_wxyz(Rotation()) == [1.0, 0.1, 0.2, 0.3]


def test_random_actions_are_seeded_and_navigation_compatible() -> None:
    module = load_smoke_module()

    first = module.random_actions(count=8, seed=7)
    second = module.random_actions(count=8, seed=7)

    assert first == second
    assert len(first) == 8
    assert set(first) <= {"move_forward", "turn_left", "turn_right"}
