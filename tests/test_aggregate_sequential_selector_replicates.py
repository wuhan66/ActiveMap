from scripts.aggregate_sequential_selector_replicates import trace_validity_rate


def test_v2_aggregation_uses_executable_validity_not_raw_schema_validity():
    traces = [
        {
            "valid_action": False,
            "raw_json_valid": True,
            "raw_schema_valid": False,
            "executable_action_valid": True,
        },
        {
            "valid_action": True,
            "raw_json_valid": True,
            "raw_schema_valid": True,
            "executable_action_valid": True,
        },
    ]

    assert trace_validity_rate(traces, "raw_json_valid") == 1.0
    assert trace_validity_rate(traces, "raw_schema_valid") == 0.5
    assert trace_validity_rate(traces, "executable_action_valid") == 1.0


def test_v1_receipts_fall_back_to_their_legacy_valid_action_field():
    traces = [{"valid_action": True}, {"valid_action": False}]

    assert trace_validity_rate(traces, "executable_action_valid") == 0.5
