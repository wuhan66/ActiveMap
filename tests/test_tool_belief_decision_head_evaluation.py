# ruff: noqa: E402

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from scripts.evaluate_tool_belief_decision_head import evaluate_checkpoint
from scripts.train_tool_belief_decision_head import (
    FEATURE_DIM,
    HierarchicalDecisionHead,
    _feature_names,
)


def _rows(split: str) -> list[dict[str, object]]:
    rows = []
    for sequence_id, target, baseline, probabilities in (
        ("keep", "KEEP", "KEEP", [0.8, 0.1, 0.05, 0.05]),
        ("add", "ADD", "KEEP", [0.3, 0.6, 0.05, 0.05]),
    ):
        for step in range(1, 4):
            prediction = ("KEEP", "ADD", "DELETE", "RESHAPE")[
                max(range(4), key=probabilities.__getitem__)
            ]
            rows.append(
                {
                    "sequence_id": sequence_id,
                    "split": split,
                    "step": step,
                    "target": target,
                    "baseline": baseline,
                    "baseline_probabilities": [0.7, 0.1, 0.1, 0.1],
                    "baseline_recommended_edit": baseline,
                    "paired": prediction,
                    "paired_probabilities": probabilities,
                    "paired_confidence": 0.9,
                }
            )
    return rows


def test_decision_checkpoint_round_trip(tmp_path: Path) -> None:
    details = tmp_path / "val.jsonl"
    rows = _rows("val")
    for row in rows:
        row["no_quality"] = row["paired"]
        row["no_quality_probabilities"] = row["paired_probabilities"]
        row["no_quality_confidence"] = row["paired_confidence"]
    details.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    model = HierarchicalDecisionHead(hidden_dim=16, dropout=0.0)
    checkpoint = tmp_path / "head.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": 1,
            "hidden_dim": 16,
            "dropout": 0.0,
            "feature_dim": FEATURE_DIM,
            "feature_names": _feature_names(),
            "decision_threshold": 0.5,
            "protocol": "hierarchical_decision_head_v1",
        },
        checkpoint,
    )

    report, output_rows = evaluate_checkpoint(
        checkpoint,
        details,
        minimum_macro_f1_gain=-1.0,
        max_false_edit_increase=1.0,
        max_missed_edit_increase=1.0,
    )

    assert report["protocol"]["sequence_count"] == 2
    assert set(report["summaries"]) == {
        "identity",
        "residual_argmax",
        "hierarchical",
        "no_quality",
        "no_anchor",
        "no_current",
    }
    assert len(output_rows) == 2
