import pytest

from scripts.evaluate_semantic_vlm_rollouts import index_pairs


def test_index_pairs_preserves_pre_post_pair_order():
    rows = [
        {"example_id": "b", "stage": "POST_TOOL"},
        {"example_id": "a", "stage": "PRE_TOOL"},
        {"example_id": "b", "stage": "PRE_TOOL"},
        {"example_id": "a", "stage": "POST_TOOL"},
    ]
    pairs = index_pairs(rows)
    assert [(pre["example_id"], post["example_id"]) for pre, post in pairs] == [
        ("b", "b"),
        ("a", "a"),
    ]


def test_index_pairs_rejects_missing_post_state():
    with pytest.raises(ValueError, match="incomplete"):
        index_pairs([{"example_id": "a", "stage": "PRE_TOOL"}])
