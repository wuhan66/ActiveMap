import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_oracle_writeback_router.py"
SPEC = importlib.util.spec_from_file_location("build_oracle_writeback_router", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _row(task_id, budget, raster_iou):
    return {
        "task_id": task_id,
        "budget": budget,
        "raster_iou": raster_iou,
        "episode_utility_v2_balanced": raster_iou - 0.1,
        "test_assets_read": False,
    }


def test_oracle_router_selects_higher_metric_and_records_provenance():
    rows, summary = MODULE.route(
        [_row("a", 1.0, 0.8), _row("b", 1.0, 0.7)],
        [_row("a", 1.0, 0.6), _row("b", 1.0, 0.9)],
        metric="raster_iou",
    )

    assert [row["oracle_routing_source"] for row in rows] == ["primary", "alternate"]
    assert [row["raster_iou"] for row in rows] == [0.8, 0.9]
    assert summary["alternate_selection_rate"] == 0.5
    assert summary["deployment_valid"] is False


def test_oracle_router_requires_identical_pairs():
    with pytest.raises(ValueError, match="identical"):
        MODULE.route([_row("a", 1.0, 0.8)], [_row("b", 1.0, 0.9)], metric="raster_iou")
