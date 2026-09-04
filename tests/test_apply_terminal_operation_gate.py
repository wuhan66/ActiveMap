from scripts.apply_terminal_operation_gate import gate_row


def _row(prediction: str):
    return {
        "prediction": prediction,
        "selected_evidence_ids": ["ev-1"],
        "spent_cost": 2.0,
    }


def test_add_commit_passes_gate():
    result = gate_row(_row("COMMIT:ADD"), {"ADD"})

    assert result["prediction"] == "COMMIT:ADD"
    assert result["selected_evidence_ids"] == ["ev-1"]
    assert result["terminal_operation_gate"]["passed"] is True


def test_delete_commit_is_rejected_without_dropping_acquired_evidence_or_cost():
    result = gate_row(_row("COMMIT:DELETE"), {"ADD"})

    assert result["prediction"] == "REJECT"
    assert result["selected_evidence_ids"] == ["ev-1"]
    assert result["spent_cost"] == 2.0
    assert result["ungated_prediction"] == "COMMIT:DELETE"
    assert result["ungated_selected_evidence_ids"] == ["ev-1"]


def test_existing_reject_passes_gate():
    result = gate_row(_row("REJECT"), {"ADD"})

    assert result["prediction"] == "REJECT"
    assert result["terminal_operation_gate"]["passed"] is True
