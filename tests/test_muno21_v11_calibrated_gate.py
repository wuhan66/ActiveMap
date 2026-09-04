from pathlib import Path

from scripts.assess_muno21_v11_diagnostic import CANDIDATE, assess
from scripts.select_muno21_v11_controller import select


def test_controller_selection_separates_terminal_quality_from_call_safety(
    tmp_path: Path,
) -> None:
    for label in ("checkpoint-300", "checkpoint-500"):
        adapter = tmp_path / "checkpoints" / label
        adapter.mkdir(parents=True)
        (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    static = {
        "checkpoints": [
            {
                "label": "checkpoint-300",
                "test_assets_read": False,
                "schema_valid_rate": 1.0,
                "executable_valid_rate": 1.0,
                "predicted_tool_calls": 44,
                "tool_positive_exact": 0.54,
                "macro_f1": 0.43,
                "exact_action_accuracy": 0.68,
            },
            {
                "label": "checkpoint-500",
                "test_assets_read": False,
                "schema_valid_rate": 1.0,
                "executable_valid_rate": 1.0,
                "predicted_tool_calls": 20,
                "tool_positive_exact": 0.54,
                "macro_f1": 0.53,
                "exact_action_accuracy": 0.67,
            },
        ]
    }
    gate = {
        "promotion_gate": {"passed": True},
        "validation": {"false_call_rate": 0.0, "recall": 0.5},
    }

    result = select(static, gate, run_dir=tmp_path)

    assert result["selected_checkpoint"] == "checkpoint-500"
    assert result["test_assets_read"] is False


def _row(method: str, utility: float, calls: float, delta: float, false: float = 0.01):
    return {
        "method": method,
        "mean_quality_cost_utility": utility,
        "false_edit_rate": false,
        "mean_tool_calls": calls,
        "mean_tool_belief_l1_delta": delta,
    }


def test_v11_diagnostic_requires_quality_cost_safety_dominance() -> None:
    summary = {
        "protocol": {
            "test_assets_read": False,
            "tool_need_gate": {"proposals": 8, "admitted": 3},
        },
        "results": [
            _row(CANDIDATE, 0.30, 0.2, 0.1),
            _row("qwen3_4b_sft_tool_to_belief", 0.20, 0.8, 0.1),
            _row("edit_conditioned_selector", 0.15, 0.0, 0.0),
            _row("qwen3_4b_sft_tools_no_belief", 0.18, 0.4, 0.0),
            _row("forced_tools", 0.10, 2.0, 0.1),
        ],
        "action_counts": {CANDIDATE: {"USE_TOOL": 3}},
    }

    result = assess(summary)

    assert result["diagnostic_passed"] is True
    assert result["checks"]["fewer_calls_than_uncalibrated"] is True
