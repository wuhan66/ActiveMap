import pytest

from scripts.merge_policy_relative_vlm_branch_shards import interleave_shards


def _row(example_id: str) -> dict:
    return {"example_id": example_id}


def test_interleave_shards_restores_source_order():
    rows = interleave_shards(
        [[_row("e0"), _row("e2"), _row("e4")], [_row("e1"), _row("e3")]],
        5,
    )

    assert [row["example_id"] for row in rows] == ["e0", "e1", "e2", "e3", "e4"]


def test_interleave_shards_rejects_incomplete_coverage():
    with pytest.raises(ValueError, match="cover"):
        interleave_shards([[_row("e0")], []], 2)
