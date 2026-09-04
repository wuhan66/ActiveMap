import subprocess
import sys

from scripts.compare_active_catalog_sampling_ablation import compare_traces


def test_comparator_supports_direct_script_execution() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/compare_active_catalog_sampling_ablation.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def _row(index: int, aoi: str, target: str, predicted: str, utility: float):
    target_id = f"e{index}" if target == "ACQUIRE" else None
    predicted_id = f"e{index}" if predicted == "ACQUIRE" else None
    oracle = 1.0 if target == "ACQUIRE" else 0.0
    return {
        "example_id": f"x{index}",
        "task_id": f"t{index}",
        "aoi_id": aoi,
        "source_episode": f"episode{index}",
        "split": "val",
        "budget": 1.0,
        "oracle_step": 0,
        "gt_edit": "RESHAPE",
        "draft_edit": "RESHAPE",
        "candidate_count": 2,
        "target_selection": target,
        "target_evidence_id": target_id,
        "predicted_selection": predicted,
        "predicted_evidence_id": predicted_id,
        "valid_action": True,
        "stop_utility": 0.0,
        "oracle_utility": oracle,
        "policy_utility": utility,
        "policy_cost": 1.0 if predicted == "ACQUIRE" else 0.0,
        "regret": oracle - utility,
    }


def test_weighted_sampling_promotes_on_paired_utility_and_safety_gain():
    weighted = [
        _row(0, "a", "ACQUIRE", "ACQUIRE", 1.0),
        _row(1, "a", "STOP", "STOP", 0.0),
        _row(2, "b", "ACQUIRE", "ACQUIRE", 1.0),
        _row(3, "b", "STOP", "STOP", 0.0),
    ]
    unweighted = [
        _row(0, "a", "ACQUIRE", "STOP", 0.0),
        _row(1, "a", "STOP", "STOP", 0.0),
        _row(2, "b", "ACQUIRE", "STOP", 0.0),
        _row(3, "b", "STOP", "STOP", 0.0),
    ]

    report = compare_traces(weighted, unweighted, repetitions=100, seed=7)

    assert report["decision"] == "weighted"
    assert report["weighted_gate"]["passed"] is True
    assert report["paired_aoi_bootstrap"]["intervals"]["realized_utility_mean"]["ci95_low"] > 0


def test_sampling_comparison_rejects_protocol_mismatch():
    weighted = [_row(0, "a", "ACQUIRE", "ACQUIRE", 1.0), _row(1, "b", "STOP", "STOP", 0.0)]
    unweighted = [dict(row) for row in weighted]
    unweighted[0]["budget"] = 2.0

    try:
        compare_traces(weighted, unweighted, repetitions=10, seed=7)
    except ValueError as error:
        assert "protocol mismatch" in str(error)
    else:
        raise AssertionError("expected protocol mismatch")
