import json

import pytest

from scripts.build_sn7_controller_validation_table import build_table, render_markdown


def _row(episode, aoi, *, policy, utility, correct=True, target="ADD"):
    return {
        "source_episode": episode,
        "aoi_id": aoi,
        "split": "val",
        "policy": policy,
        "budget": 3.0,
        "target_edit": target,
        "predicted_edit": target if correct else "KEEP",
        "terminal_correct": correct,
        "false_edit": False,
        "missed_edit": not correct,
        "wrong_edit": False,
        "acquisitions": 1,
        "steps": 2,
        "spent_cost": 0.1,
        "tool_calls": 0,
        "tool_belief_l1_delta": 0.0,
        "quality_gain": utility + 0.1,
        "quality_cost_utility": utility,
        "episode_utility_v2_proxy_balanced": utility,
        "episode_utility_v2_proxy_safety": utility,
        "episode_utility_v2_proxy_cost_aware": utility,
        "selected_evidence_ids": [],
        "model_action_count": 1,
        "valid_action_count": 1,
        "fallback_count": 0,
        "events": [{}],
        "test_assets_read": False,
    }


def _trace(tmp_path, name, rows):
    path = tmp_path / f"{name}.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_controller_table_requires_aligned_support_and_reports_pairing(tmp_path):
    reference = _trace(
        tmp_path,
        "sft",
        [_row("e1", "a", policy="sft", utility=0.0, correct=False),
         _row("e2", "b", policy="sft", utility=0.0, correct=False)],
    )
    candidate = _trace(
        tmp_path,
        "react",
        [_row("e1", "a", policy="react", utility=0.2),
         _row("e2", "b", policy="react", utility=0.3)],
    )
    result = build_table(
        {"sft": reference, "react": candidate},
        reference="sft",
        candidate="react",
        expected_records=2,
        repetitions=50,
        seed=7,
    )
    assert result["test_assets_read"] is False
    assert result["paired_vs_reference"]["react"]["strict_utility_gain_ci95"] is True
    assert result["by_budget"]["3"]["record_count"] == 2
    assert (
        result["by_budget"]["3"]["paired_vs_reference"]["react"][
            "strict_utility_gain_ci95"
        ]
        is True
    )
    assert "Matched Budget Frontier" in render_markdown(result)
    assert "SN7 Common Controller Validation" in render_markdown(result)


def test_controller_table_rejects_mismatched_support(tmp_path):
    reference = _trace(
        tmp_path,
        "sft",
        [_row("e1", "a", policy="sft", utility=0.0)],
    )
    candidate = _trace(
        tmp_path,
        "react",
        [_row("e2", "a", policy="react", utility=0.2)],
    )
    with pytest.raises(ValueError, match="identical"):
        build_table(
            {"sft": reference, "react": candidate},
            reference="sft",
            candidate=None,
            expected_records=1,
            repetitions=20,
            seed=7,
        )


def test_controller_table_rejects_test_trace(tmp_path):
    row = _row("e1", "a", policy="sft", utility=0.0)
    row["split"] = "test"
    row["test_assets_read"] = True
    path = _trace(tmp_path, "test", [row])
    with pytest.raises(ValueError, match="invalid val trace"):
        build_table(
            {"sft": path},
            reference="sft",
            candidate=None,
            expected_records=1,
            repetitions=20,
            seed=7,
        )
