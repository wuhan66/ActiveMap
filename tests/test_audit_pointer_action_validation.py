from scripts.audit_pointer_action_validation import audit_pointer_action_validation


def _summary(*, fallback: float = 0.0, pointer_count: int = 3) -> dict:
    return {
        "protocol": {
            "split": "val",
            "test_assets_read": False,
            "decoding": {"do_sample": False},
        },
        "results": [
            {
                "method": "qwen3_4b_sft_tool_to_belief",
                "sample_count": 20,
                "mean_tool_calls": 0.2,
            }
        ],
        "tool_positive_results": [
            {
                "method": "qwen3_4b_sft_tool_to_belief",
                "sample_count": 5,
                "mean_tool_calls": 0.8,
            }
        ],
        "llm_validity_by_method": {
            "qwen3_4b_sft_tool_to_belief": {
                "schema_valid_rate": 1.0,
                "executable_valid_rate": 1.0,
                "fallback_rate": fallback,
                "pointer_resolution_counts": {
                    "selected_evidence_index_v1": pointer_count
                },
            }
        },
    }


def test_pointer_validation_audit_passes_complete_greedy_controller() -> None:
    report = audit_pointer_action_validation(
        _summary(), method="qwen3_4b_sft_tool_to_belief"
    )

    assert report["ready_for_executable_grpo"] is True
    assert report["estimated_positive_subset_tool_calls"] == 4.0


def test_pointer_validation_audit_blocks_fallback_or_missing_pointer() -> None:
    report = audit_pointer_action_validation(
        _summary(fallback=0.1, pointer_count=0), method="qwen3_4b_sft_tool_to_belief"
    )

    assert report["ready_for_executable_grpo"] is False
    assert "pointer_resolved" in report["failed_gates"]
    assert "no_material_fallback" in report["failed_gates"]
