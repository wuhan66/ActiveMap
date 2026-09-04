import numpy as np

from scripts.audit_post_acquisition_tool_features import (
    cyclically_shuffle_tools,
    operation_metrics,
)


def test_cyclic_tool_shuffle_is_deterministic_and_nonidentity():
    tools = np.asarray([[1.0], [2.0], [3.0]])
    shuffled = cyclically_shuffle_tools(tools)
    assert shuffled[:, 0].tolist() == [3.0, 1.0, 2.0]


def test_operation_metrics_reports_false_and_missed_edits():
    target = np.asarray([0, 0, 1, 2])
    prediction = np.asarray([0, 1, 0, 2])
    metrics = operation_metrics(target, prediction)
    assert metrics["false_edit_rate"] == 0.5
    assert metrics["missed_edit_rate"] == 0.5
    assert len(metrics["confusion_matrix"]) == 4
