from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_sn7_fullval_controller_matrix import audit


def _write_trace(path: Path, *, quality_offset: float) -> None:
    rows = []
    budgets = (1.5, 3.0, 4.5)
    for index in range(6):
        rows.append(
            {
                "sample_id": f"sample-{index}",
                "source_episode": f"episode-{index}",
                "aoi_id": f"aoi-{index % 2}",
                "split": "val",
                "budget": budgets[index % len(budgets)],
                "target_edit": "KEEP",
                "predicted_edit": "KEEP",
                "terminal_correct": True,
                "false_edit": False,
                "missed_edit": False,
                "wrong_edit": False,
                "acquisitions": 0,
                "spent_cost": 0.0,
                "tool_calls": 0,
                "quality_gain": quality_offset,
                "quality_cost_utility": quality_offset,
                "episode_utility_v2_proxy_balanced": quality_offset,
                "episode_utility_v2_proxy_safety": quality_offset,
                "episode_utility_v2_proxy_cost_aware": quality_offset,
                "model_action_count": 1,
                "valid_action_count": 1,
                "fallback_count": 0,
                "events": [{"observable_state": {"belief": {}}}],
                "test_assets_read": False,
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_audit_matched_controller_matrix(tmp_path: Path) -> None:
    reference = tmp_path / "reference.jsonl"
    seed1 = tmp_path / "seed1.jsonl"
    seed2 = tmp_path / "seed2.jsonl"
    _write_trace(reference, quality_offset=0.0)
    _write_trace(seed1, quality_offset=0.1)
    _write_trace(seed2, quality_offset=0.2)
    result = audit(
        [
            ("old_vla", "reference", reference),
            ("hybrid_8k", "1", seed1),
            ("hybrid_8k", "2", seed2),
        ],
        reference_method="old_vla",
        expected_count=6,
        repetitions=200,
        bootstrap_seed=7,
    )
    assert result["all_checks_passed"] is True
    assert result["support_identical"] is True
    comparison = result["comparisons_vs_reference"]["hybrid_8k"]
    observed = comparison["intervals"]["mean_quality_cost_utility"]["observed_delta"]
    assert observed == pytest.approx(0.15)
    assert comparison["strict_qc_gain_ci95"] is True
    assert result["budgets"] == [1.5, 3.0, 4.5]
    hybrid_auc = next(row for row in result["auc_summary"] if row["method"] == "hybrid_8k")
    assert hybrid_auc["auc_mean_quality_cost_utility"] == pytest.approx(0.15)
