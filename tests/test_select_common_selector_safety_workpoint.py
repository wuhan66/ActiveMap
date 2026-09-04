import pytest

from scripts.select_common_selector_safety_workpoint import select_common_workpoint


def _row(label, harmful_limit, utility, harmful, recall, *, train_ok=True):
    return {
        "label": label,
        "max_harmful_call_fraction": harmful_limit,
        "min_acquire_recall": 0.10,
        "calibration": {"constraints_satisfied": float(train_ok), "stop_margin": 0.5},
        "validation": {
            "mean_utility": utility,
            "mean_regret": 0.1 - utility,
            "call_rate": 0.1,
            "false_call_rate": 0.001,
            "harmful_call_fraction": harmful,
            "acquire_recall": recall,
            "exact_acquire_recall": 0.03,
        },
    }


def _payload(checkpoint, rows):
    return {"checkpoint": checkpoint, "rows": rows, "test_assets_read": False}


def test_selects_highest_utility_candidate_feasible_for_every_seed() -> None:
    payloads = [
        _payload(
            {"path": "seed1.pt"},
            [
                _row("h18", 0.18, 0.014, 0.29, 0.40),
                _row("h20", 0.20, 0.016, 0.31, 0.50),
            ],
        ),
        _payload(
            {"path": "seed2.pt"},
            [
                _row("h18", 0.18, 0.012, 0.27, 0.35),
                _row("h20", 0.20, 0.015, 0.29, 0.48),
            ],
        ),
    ]
    result = select_common_workpoint(
        payloads,
        max_false_call_rate=0.02,
        max_harmful_call_fraction=0.30,
        min_acquire_recall=0.01,
    )
    assert result["promoted"] is True
    assert result["winner"]["label"] == "h18"
    assert result["winner"]["mean"]["mean_utility"] == pytest.approx(0.013)


def test_rejects_mismatched_candidate_protocols() -> None:
    payloads = [
        _payload({"path": "seed1.pt"}, [_row("h18", 0.18, 0.01, 0.2, 0.2)]),
        _payload({"path": "seed2.pt"}, [_row("h20", 0.20, 0.01, 0.2, 0.2)]),
    ]
    with pytest.raises(ValueError, match="identical ordered candidates"):
        select_common_workpoint(
            payloads,
            max_false_call_rate=0.02,
            max_harmful_call_fraction=0.30,
            min_acquire_recall=0.01,
        )
