import json

import pytest

from scripts.aggregate_post_acquisition_tool_belief_seeds import METRICS, aggregate


def _summary(seed, promoted, full_f1, zero_f1):
    metrics = {name: 0.1 for name in METRICS}
    metrics.update(
        {
            "baseline_deployed_macro_f1": 0.5,
            "zero_tool_macro_f1": zero_f1,
            "full_macro_f1": full_f1,
            "baseline_deployed_false_edit_rate": 0.02,
            "full_false_edit_rate": 0.03,
        }
    )
    return {
        "seed": seed,
        "protocol": "post_acquisition_tool_dependent_residual_v1",
        "train_examples": 10,
        "val_examples": 4,
        "train_tasks": 3,
        "val_tasks": 2,
        "batch_size": 2,
        "learning_rate": 0.001,
        "hidden_dim": 8,
        "dropout": 0.0,
        "max_logit_delta": 1.5,
        "geometry_scale": 0.15,
        "safety_margin": 0.02,
        "min_quality_delta": 0.01,
        "min_tool_delta": 0.01,
        "max_calibration_degradation": 0.03,
        "loss_weights": {"false_edit": 1.5},
        "tool_contrastive_margin": 0.05,
        "promotion_passed": promoted,
        "best_epoch": 2,
        "best_val_metrics": metrics,
        "best_promoted_epoch": 3 if promoted else None,
        "best_promoted_val_metrics": metrics if promoted else None,
    }


def test_aggregate_retains_failed_seed_and_blocks_paper_claim(tmp_path):
    paths = []
    for row in (_summary(1, True, 0.54, 0.52), _summary(2, False, 0.51, 0.55)):
        path = tmp_path / f"seed{row['seed']}.json"
        path.write_text(json.dumps(row), encoding="utf-8")
        paths.append(path)
    result = aggregate(paths)
    assert result["promoted_seed_count"] == 1
    assert result["promotion_rate"] == 0.5
    assert result["paper_claim_ready"] is False
    assert result["per_seed"][1]["selection_kind"] == "best_non_promoted"
    delta = result["aggregate_deltas"]["full_minus_zero_tool_macro_f1"]["mean"]
    assert delta == pytest.approx(-0.01)


def test_aggregate_rejects_protocol_drift(tmp_path):
    first = _summary(1, True, 0.54, 0.52)
    second = _summary(2, True, 0.55, 0.53)
    second["safety_margin"] = 0.03
    paths = []
    for row in (first, second):
        path = tmp_path / f"seed{row['seed']}.json"
        path.write_text(json.dumps(row), encoding="utf-8")
        paths.append(path)
    with pytest.raises(ValueError, match="frozen protocol"):
        aggregate(paths)
