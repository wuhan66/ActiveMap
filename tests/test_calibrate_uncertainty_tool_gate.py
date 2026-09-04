import json

from scripts.calibrate_uncertainty_tool_gate import calibrate


def _row(task: str, uncertainty: float) -> dict:
    state = {
        "controller_stage": "SELECT",
        "belief": {"uncertainty": uncertainty},
        "selected_evidence_ids": ["initial"],
    }
    return {
        "task_id": task,
        "split": "train",
        "test_assets_read": False,
        "prompt": [
            {
                "role": "user",
                "content": [{"type": "text", "text": json.dumps(state)}],
            }
        ],
    }


def test_calibration_uses_train_uncertainty_without_outcomes(tmp_path):
    source = tmp_path / "train.jsonl"
    source.write_text(
        "".join(
            json.dumps(_row(f"t{index}", value)) + "\n"
            for index, value in enumerate((0.1, 0.2, 0.3, 0.9))
        ),
        encoding="utf-8",
    )
    result = calibrate(source, target_call_rate=0.25)
    assert result["selected"]["threshold"] == 0.9
    assert result["observed_train_call_rate"] == 0.25
    assert result["outcome_labels_used"] is False
