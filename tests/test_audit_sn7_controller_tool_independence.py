import json

from scripts.audit_sn7_controller_tool_independence import audit, discover


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_audit_accepts_grounded_tool_results(tmp_path):
    trace = tmp_path / "run" / "tool_results.jsonl"
    _write(
        trace,
        {
            "sample_id": "sample-1",
            "results": [
                {
                    "tool": "TEMPORAL_CHANGE",
                    "success": True,
                    "outputs": {"changed_fraction": 0.2},
                    "cost": 0.15,
                }
            ],
            "test_assets_read": False,
        },
    )

    payload = audit(discover([tmp_path]))

    assert payload["status"] == "pass"
    assert payload["trace_count"] == 1
    assert payload["invocation_count"] == 1
    assert payload["tool_counts"] == {"TEMPORAL_CHANGE": 1}
    assert payload["tool_success_rates"] == {"TEMPORAL_CHANGE": 1.0}
    assert payload["failure_rate"] == 0.0
    assert payload["forbidden_hits"] == []


def test_audit_rejects_nested_target_fields(tmp_path):
    trace = tmp_path / "tool_results.jsonl"
    _write(
        trace,
        {
            "results": [
                {
                    "tool": "IMAGE_QUALITY",
                    "success": True,
                    "outputs": {"target_geometry": {"type": "Polygon"}},
                }
            ]
        },
    )

    payload = audit([trace])

    assert payload["status"] == "fail"
    assert payload["forbidden_hits"][0]["key_paths"] == [
        "$.results[0].outputs.target_geometry"
    ]
