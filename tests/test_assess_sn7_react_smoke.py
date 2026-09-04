import json

from scripts.assess_sn7_react_smoke import assess


def _write_fixture(tmp_path, *, valid_action_rate=1.0, test_assets_read=False):
    summary = {
        "schema_version": "active-catalog-closed-loop-evaluation-v1",
        "sample_count": 2,
        "split": "val",
        "metrics": {
            "valid_action_rate": valid_action_rate,
            "fallback_episode_rate": 0.0,
            "mean_tool_calls": 0.5,
        },
        "protocol": {
            "policy_mode": "react",
            "controller_protocol": "react_observation_reason_action",
            "tool_mode": "model",
            "explicit_geospatial_tool_calls": True,
        },
        "test_assets_read": test_assets_read,
    }
    traces = [
        {
            "test_assets_read": test_assets_read,
            "events": [{"observable_state": {"controller_stage": "REACT"}}],
        }
        for _ in range(2)
    ]
    summary_path = tmp_path / "summary.json"
    traces_path = tmp_path / "traces.jsonl"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    traces_path.write_text(
        "\n".join(json.dumps(row) for row in traces) + "\n",
        encoding="utf-8",
    )
    return summary_path, traces_path


def test_react_smoke_assessment_accepts_valid_validation_trace(tmp_path):
    summary, traces = _write_fixture(tmp_path)
    result = assess(summary, traces)
    assert result["passed"] is True
    assert all(result["checks"].values())


def test_react_smoke_assessment_rejects_invalid_actions(tmp_path):
    summary, traces = _write_fixture(tmp_path, valid_action_rate=0.0)
    result = assess(summary, traces)
    assert result["passed"] is False
    assert result["checks"]["valid_action_rate"] is False


def test_react_smoke_assessment_rejects_test_access(tmp_path):
    summary, traces = _write_fixture(tmp_path, test_assets_read=True)
    result = assess(summary, traces)
    assert result["passed"] is False
    assert result["checks"]["validation_only"] is False
