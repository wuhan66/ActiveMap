# ruff: noqa: E402

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from scripts.evaluate_tool_belief_stage_head import evaluate_stage_checkpoint
from scripts.train_tool_belief_decision_head import (
    FEATURE_DIM,
    HierarchicalDecisionHead,
    _feature_names,
)


def _details(path: Path) -> None:
    rows = []
    for sequence_id, target, probabilities in (
        ("keep", "KEEP", [0.8, 0.1, 0.05, 0.05]),
        ("add", "ADD", [0.2, 0.7, 0.05, 0.05]),
    ):
        for step in range(1, 4):
            rows.append(
                {
                    "sequence_id": sequence_id,
                    "split": "val",
                    "step": step,
                    "target": target,
                    "baseline": "KEEP",
                    "baseline_probabilities": [0.7, 0.1, 0.1, 0.1],
                    "baseline_recommended_edit": "KEEP",
                    "paired": "KEEP" if target == "KEEP" else "ADD",
                    "paired_probabilities": probabilities,
                    "paired_confidence": 0.8,
                }
            )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_stage_evaluator_checks_every_causal_prefix(tmp_path: Path) -> None:
    details = tmp_path / "val.jsonl"
    _details(details)
    model = HierarchicalDecisionHead(hidden_dim=16, dropout=0.0)
    checkpoint = tmp_path / "prefix.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": 1,
            "hidden_dim": 16,
            "dropout": 0.0,
            "feature_dim": FEATURE_DIM,
            "feature_names": _feature_names(),
            "decision_thresholds": {str(stage): 0.5 for stage in range(4)},
            "prefix_stages": [0, 1, 2, 3],
            "protocol": "hierarchical_decision_head_prefix_v2",
        },
        checkpoint,
    )

    report = evaluate_stage_checkpoint(
        checkpoint,
        details,
        minimum_final_macro_f1_gain=-1.0,
        max_false_edit_increase=1.0,
        max_missed_edit_increase=1.0,
    )

    assert report["protocol"]["sequence_count"] == 2
    assert report["protocol"]["state_count"] == 8
    assert set(report["stage_summaries"]) == {"stage_0", "stage_1", "stage_2", "stage_3"}
    assert all(report["causal_future_zero_checks"].values())
    assert report["gates"]["passed"] is True
