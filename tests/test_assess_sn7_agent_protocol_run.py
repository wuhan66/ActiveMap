import json

from scripts.assess_sn7_agent_protocol_run import assess


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _make_run(tmp_path, protocol="geommagent_style"):
    specifications = {
        "geommagent_style": (
            "geommagent_plan_execute_self_evaluate",
            "geommagent_style_qwen",
            "PLAN",
        ),
        "sensesearch_style": (
            "sensesearch_iterative_search_crop_reason",
            "sensesearch_style_qwen",
            "SEARCH",
        ),
    }
    controller, policy, stage = specifications[protocol]
    registered_tools = (
        [
            "IMAGE_QUALITY",
            "TEMPORAL_CHANGE",
            "RASTER_CROP",
            "RASTER_SEGMENT",
            "VECTOR_INSPECT",
        ]
        if protocol == "sensesearch_style"
        else ["IMAGE_QUALITY", "TEMPORAL_CHANGE"]
    )
    _write_json(tmp_path / "run_state.json", {"status": "completed", "returncode": 0})
    _write_json(
        tmp_path / "evaluation" / "summary.json",
        {
            "schema_version": "active-catalog-closed-loop-evaluation-v1",
            "sample_count": 2,
            "split": "val",
            "metrics": {"valid_action_rate": 1.0},
            "protocol": {
                "policy_mode": protocol,
                "controller_protocol": controller,
                "tool_mode": "model",
                "explicit_geospatial_tool_calls": True,
                "registered_tools": registered_tools,
                "asset_root_maps": [{"source": "/old", "target": "/new"}],
            },
            "test_assets_read": False,
        },
    )
    rows = [
        {
            "split": "val",
            "policy": policy,
            "budget": 1.5,
            "spent_cost": 0.1,
            "acquisitions": 0,
            "tool_calls": 1,
            "predicted_edit": "KEEP",
            "events": [
                {
                    "controller_protocol": protocol,
                    "observable_state": {"controller_stage": stage},
                }
            ],
            "test_assets_read": False,
        }
        for _ in range(2)
    ]
    (tmp_path / "evaluation" / "traces.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_assessment_accepts_both_protocols(tmp_path):
    for protocol in ("geommagent_style", "sensesearch_style"):
        root = tmp_path / protocol
        _make_run(root, protocol)
        assert assess(root, protocol, 2, 0.5)["passed"] is True


def test_assessment_records_and_enforces_valid_action_threshold(tmp_path):
    _make_run(tmp_path)
    summary_path = tmp_path / "evaluation" / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["metrics"]["valid_action_rate"] = 0.75
    _write_json(summary_path, summary)

    result = assess(tmp_path, "geommagent_style", 2, 0.95)

    assert result["minimum_valid_action_rate"] == 0.95
    assert result["passed"] is False
    assert result["checks"]["valid_action_rate"] is False


def test_assessment_rejects_budget_violation(tmp_path):
    _make_run(tmp_path)
    path = tmp_path / "evaluation" / "traces.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["spent_cost"] = 2.0
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = assess(tmp_path, "geommagent_style", 2, 0.5)
    assert result["passed"] is False
    assert result["checks"]["budget_safe"] is False


def test_assessment_rejects_protocol_leakage(tmp_path):
    _make_run(tmp_path)
    path = tmp_path / "evaluation" / "traces.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["events"][0]["observable_state"]["target_edit"] = "KEEP"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = assess(tmp_path, "geommagent_style", 2, 0.5)
    assert result["passed"] is False
    assert result["checks"]["observable_protocol"] is False


def test_assessment_rejects_failed_tool_observation(tmp_path):
    _make_run(tmp_path, "sensesearch_style")
    path = tmp_path / "evaluation" / "traces.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["events"][0]["observable_state"]["tool_observations"] = [
        {"tool": "RASTER_CROP", "success": False}
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = assess(tmp_path, "sensesearch_style", 2, 0.5)
    assert result["passed"] is False
    assert result["checks"]["tool_execution_success"] is False
