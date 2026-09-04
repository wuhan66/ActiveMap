from types import SimpleNamespace

import pytest

from activemap.models import EditOperation
from scripts.evaluate_agent_map_writeback import (
    _executable_operation_errors,
    _road_width_source_pixels,
)


def test_sn7_polygon_episode_does_not_receive_default_road_width():
    episode = SimpleNamespace(metadata={})

    assert _road_width_source_pixels(episode) is None


def test_muno21_episode_uses_explicit_road_width():
    episode = SimpleNamespace(metadata={"road_width_source_pixels": 6.0})

    assert _road_width_source_pixels(episode) == 6.0


def test_invalid_explicit_road_width_is_rejected():
    episode = SimpleNamespace(metadata={"road_width_source_pixels": 0.0})

    with pytest.raises(ValueError, match="must be positive"):
        _road_width_source_pixels(episode)


def test_reject_with_executable_delete_is_false_edit():
    assert _executable_operation_errors("REJECT", EditOperation.DELETE) == (
        True,
        False,
        False,
    )


def test_commit_add_with_no_executable_delta_is_missed_edit():
    assert _executable_operation_errors("COMMIT:ADD", EditOperation.KEEP) == (
        False,
        True,
        False,
    )


def test_commit_add_with_executable_delete_is_wrong_edit():
    assert _executable_operation_errors("COMMIT:ADD", EditOperation.DELETE) == (
        False,
        False,
        True,
    )
