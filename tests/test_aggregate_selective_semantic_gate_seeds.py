import json

import pytest

from scripts.aggregate_selective_semantic_gate_seeds import aggregate


def _payload(seed, *, promoted=True):
    metrics = {
        "accuracy": 0.6,
        "macro_f1": 0.5,
        "false_edit_rate": 0.1,
        "missed_edit_rate": 0.2,
        "tool_call_rate": 0.1,
        "mean_terminal_reward": 0.5,
        "mean_tool_cost": 0.075,
        "mean_utility": 0.425,
    }
    return {
        "schema_version": "selective-semantic-tool-gate-v1",
        "selection_protocol": "five-fold-stratified-group-OOF-on-train-only",
        "selector_feature_protocol": "pre-call-belief-quality-temporal-only",
        "tool_cost": [0.75],
        "train_examples": 100,
        "train_tasks": 50,
        "validation_examples": 40,
        "validation_tasks": 20,
        "model_training_seed": seed,
        "seed": seed,
        "train_beneficial_tool_rate": 0.1,
        "selected_gate": {"C": 1.0, "threshold": 0.5},
        "validation": {
            name: dict(metrics)
            for name in (
                "no_tool",
                "forced_tool",
                "selective_tool",
                "oracle_tool",
                "cross_task_mismatch",
            )
        },
        "promotion_gate": {"passed": promoted},
        "test_assets_read": False,
    }


def _write(tmp_path, seed, **kwargs):
    path = tmp_path / f"seed{seed}.json"
    path.write_text(json.dumps(_payload(seed, **kwargs)), encoding="utf-8")
    return path


def test_aggregate_counts_promoted_model_seeds(tmp_path):
    paths = [_write(tmp_path, 1), _write(tmp_path, 2, promoted=False)]
    result = aggregate(paths, (1, 2))
    assert result["promotion_passed_count"] == 1
    assert result["all_seeds_promoted"] is False
    assert result["paper_claim_ready"] is False


def test_aggregate_rejects_seed_mismatch(tmp_path):
    paths = [_write(tmp_path, 1), _write(tmp_path, 2)]
    with pytest.raises(ValueError, match="seed mismatch"):
        aggregate(paths, (1, 3))
