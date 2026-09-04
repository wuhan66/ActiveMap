import json
from pathlib import Path

from scripts.compare_tool_belief_reliability_gate import compare


def _summary(seed: int, *, gated: bool) -> dict:
    metrics = {
        "full_macro_f1": 0.70 + (0.01 if gated else 0.0),
        "full_false_edit_rate": 0.04 - (0.01 if gated else 0.0),
        "full_expected_calibration_error": 0.08 - (0.01 if gated else 0.0),
    }
    if gated:
        metrics.update(
            {
                "full_reliability_gate_mean": 0.4,
                "full_reliability_gate_p10": 0.2,
                "full_reliability_gate_p90": 0.7,
            }
        )
    return {
        "seed": seed,
        "reliability_gate": gated,
        "promotion_passed": True,
        "best_promoted_val_metrics": metrics,
        "train_examples": 10,
        "val_examples": 5,
        "epochs": 3,
        "batch_size": 2,
        "learning_rate": 0.001,
        "hidden_dim": 8,
        "dropout": 0.0,
        "max_logit_delta": 1.5,
        "geometry_scale": 0.15,
        "loss_weights": {},
        "tool_contrastive_margin": 0.05,
    }


def test_gate_comparison_requires_paired_noncollapsed_safe_gain(tmp_path: Path) -> None:
    ungated, gated = [], []
    for seed in (1, 2, 3):
        for is_gated, output in ((False, ungated), (True, gated)):
            path = tmp_path / f"{seed}-{is_gated}.json"
            path.write_text(json.dumps(_summary(seed, gated=is_gated)), encoding="utf-8")
            output.append(path)

    report = compare(ungated, gated)

    assert report["promotion_passed"] is True
    assert report["controlled_difference"] == "reliability_gate"
    assert report["next_stage"] == "closed_loop_quality_cost_safety"
    assert report["test_assets_read"] is False
