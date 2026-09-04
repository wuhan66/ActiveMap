import pytest

from scripts.build_tool_opportunity_strata import build_tool_opportunity_strata


def _rows(sequence_id: str, episode_id: str, target: str, baseline: str, paired: list[str]):
    return [
        {
            "sequence_id": sequence_id,
            "episode_id": episode_id,
            "step": step,
            "target": target,
            "baseline": baseline,
            "paired": prediction,
            "spent_cost": 0.18 * step,
        }
        for step, prediction in enumerate(paired, start=1)
    ]


def test_strata_separate_beneficial_boundary_and_harmful_tool_use():
    rows = [
        *_rows("positive", "task-positive", "ADD", "KEEP", ["ADD", "ADD", "ADD"]),
        *_rows("boundary", "task-boundary", "ADD", "ADD", ["ADD", "ADD", "ADD"]),
        *_rows("harmful", "task-harmful", "ADD", "ADD", ["KEEP", "KEEP", "KEEP"]),
    ]

    result = build_tool_opportunity_strata(rows)

    by_id = {row["sequence_id"]: row for row in result["records"]}
    assert by_id["positive"]["stratum"] == "positive"
    assert by_id["boundary"]["stratum"] == "near_boundary"
    assert by_id["harmful"]["stratum"] == "harmful"
    assert by_id["positive"]["best_tool_margin"] == pytest.approx(1.57)
    assert by_id["boundary"]["best_tool_margin"] == pytest.approx(-0.18)
    assert result["allowed_for_checkpoint_selection"] is False
    assert result["test_assets_read"] is False


def test_strata_reject_test_construction_and_duplicate_episodes():
    rows = [
        *_rows("one", "same-task", "ADD", "ADD", ["ADD", "ADD", "ADD"]),
        *_rows("two", "same-task", "KEEP", "KEEP", ["KEEP", "KEEP", "KEEP"]),
    ]

    with pytest.raises(ValueError, match="test data are forbidden"):
        build_tool_opportunity_strata(rows[:3], split="test")
    with pytest.raises(ValueError, match="duplicate episode_id"):
        build_tool_opportunity_strata(rows)
