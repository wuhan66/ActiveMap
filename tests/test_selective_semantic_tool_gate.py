from types import SimpleNamespace

import numpy as np

from activemap.geo_tools.records import GeoToolName, GeoToolResult
from scripts.evaluate_selective_semantic_tool_gate import (
    _policy_metrics,
    _pre_call_features,
    _task_permutation_indices,
)


def test_task_permutation_never_uses_the_same_task() -> None:
    tasks = ["a", "a", "b", "c", "c"]
    indices = _task_permutation_indices(tasks)

    assert all(tasks[index] != tasks[source] for index, source in enumerate(indices))


def test_policy_metrics_charge_cost_only_when_tool_is_called() -> None:
    target = np.asarray([0, 1])
    baseline = np.asarray([0, 0])
    semantic = np.asarray([1, 1])
    costs = np.asarray([0.75, 0.75])

    no_tool = _policy_metrics(target, baseline, semantic, costs, np.asarray([False, False]))
    selective = _policy_metrics(target, baseline, semantic, costs, np.asarray([False, True]))

    assert no_tool["mean_tool_cost"] == 0.0
    assert selective["mean_tool_cost"] == 0.375
    assert selective["mean_terminal_reward"] > no_tool["mean_terminal_reward"]


def test_selector_features_do_not_read_semantic_outputs() -> None:
    belief = SimpleNamespace(
        edit_probabilities=[0.7, 0.1, 0.1, 0.1],
        confidence=0.8,
        uncertainty=0.2,
        geometry_delta=[0.0] * 8,
    )
    row = SimpleNamespace(
        post_acquisition_belief=belief,
        quality_result=GeoToolResult(
            call_id="quality",
            tool=GeoToolName.IMAGE_QUALITY,
            success=True,
            outputs={"sharpness": 0.5},
            cost=0.03,
        ),
        temporal_result=GeoToolResult(
            call_id="temporal",
            tool=GeoToolName.TEMPORAL_CHANGE,
            success=True,
            outputs={"changed_fraction": 0.2},
            cost=0.15,
        ),
    )
    first = _pre_call_features(row)
    row.semantic_result = object()
    second = _pre_call_features(row)
    assert np.array_equal(first, second)
