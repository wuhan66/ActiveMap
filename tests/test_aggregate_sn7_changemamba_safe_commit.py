import json

from scripts.aggregate_sn7_changemamba_safe_commit import (
    METRICS,
    aggregate_safe_commit,
)


def _write_run(path, offset):
    metric_values = {
        metric: 0.1 + offset + index * 0.01
        for index, metric in enumerate(METRICS)
    }
    summary = {
        "schema_version": "sn7-changemamba-safe-commit-v1",
        "feature_names": ["confidence"],
        "calibration": {
            "method": "AOI-grouped OOF balanced logistic regression",
            "folds": 5,
            "l2": 0.01,
            "beneficial_definition": "updater map_iou_delta > 1e-8",
            "threshold_selection": "train-only",
            "selected_threshold": 0.4 + offset,
        },
        "validation": {
            "always_commit": metric_values,
            "safe_commit": {
                key: value + 0.1 for key, value in metric_values.items()
            },
            "safe_minus_always": {key: 0.1 for key in metric_values},
        },
        "test_assets_read": False,
    }
    path.mkdir()
    (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_aggregate_safe_commit_reports_mean_and_sample_std(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_run(first, 0.0)
    _write_run(second, 0.2)

    result = aggregate_safe_commit([first, second])

    delta = result["aggregate"]["safe_minus_always"]["map_iou_delta"]
    assert result["run_count"] == 2
    assert abs(delta["mean"] - 0.1) < 1e-12
    assert delta["std"] == 0.0
    assert result["test_assets_read"] is False
