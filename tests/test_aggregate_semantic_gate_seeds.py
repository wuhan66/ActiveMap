import json

import pytest

from scripts.aggregate_semantic_gate_seeds import aggregate


def _result(seed, *, direct=True, interaction=True):
    metrics = {
        "accuracy": 0.6,
        "macro_f1": 0.55,
        "false_edit_rate": 0.1,
        "missed_edit_rate": 0.2,
    }
    variants = (
        "belief_only",
        "belief_plus_weak_tools",
        "belief_plus_semantic",
        "task_deranged_semantic_mismatch",
        "belief_plus_all",
        "all_with_task_deranged_semantic_mismatch",
    )
    return {
        "schema_version": "semantic-threshold-calibration-v1",
        "selection_protocol": "five-fold-stratified-group-CV-on-train-only",
        "semantic_mismatch_protocol": "seeded-task-derangement-on-validation",
        "fixed_threshold": 0.8,
        "C_grid": [0.01, 0.1, 1.0, 10.0],
        "train_examples": 100,
        "train_tasks": 50,
        "val_examples": 40,
        "model_training_seed": seed,
        "cv_seed": seed,
        "validation": {
            **{name: dict(metrics) for name in variants},
            "deltas": {"full_minus_belief_macro_f1": 0.05},
        },
        "gate": {"passed": direct},
        "interaction_gate": {"passed": interaction},
        "test_assets_read": False,
    }


def _write(tmp_path, seed, **kwargs):
    path = tmp_path / f"seed{seed}.json"
    path.write_text(json.dumps(_result(seed, **kwargs)), encoding="utf-8")
    return path


def test_aggregate_tracks_true_model_seeds_and_gate_counts(tmp_path):
    paths = [
        _write(tmp_path, 20260716),
        _write(tmp_path, 20260719),
        _write(tmp_path, 20260722, interaction=False),
    ]
    result = aggregate(paths, (20260716, 20260719, 20260722))
    assert result["seed_semantics"] == "upstream_model_training"
    assert result["all_direct_gates_passed"] is True
    assert result["all_interaction_gates_passed"] is False
    assert result["semantic_gate_ready"] is True
    assert result["paper_claim_ready"] is False
    assert result["aggregate_deltas"]["full_minus_belief_macro_f1"]["mean"] == pytest.approx(0.05)


def test_aggregate_rejects_seed_mismatch(tmp_path):
    paths = [_write(tmp_path, 1), _write(tmp_path, 2)]
    with pytest.raises(ValueError, match="seed mismatch"):
        aggregate(paths, (1, 3))


def test_aggregate_rejects_test_access(tmp_path):
    first = _result(1)
    first["test_assets_read"] = True
    first_path = tmp_path / "first.json"
    first_path.write_text(json.dumps(first), encoding="utf-8")
    second_path = _write(tmp_path, 2)
    with pytest.raises(ValueError, match="test assets frozen"):
        aggregate([first_path, second_path], (1, 2))
