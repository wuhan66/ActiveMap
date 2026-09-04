import pytest

from scripts.diagnose_sequential_selector_likelihood import (
    binary_auc,
    candidate_actions,
    common_prefix_length,
)


def _row(evidence_id: str = "evidence-1") -> dict:
    return {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "system"}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": '{"controller_stage":"SELECT","evidence_id":"'
                        + evidence_id
                        + '"}',
                    }
                ],
            },
        ]
    }


def test_candidate_actions_are_strict_and_include_evidence_only_for_acquire():
    actions = candidate_actions(_row())

    assert actions["STOP"] == '{"stage":"SELECT","selection":"STOP"}'
    assert actions["ACQUIRE"] == (
        '{"stage":"SELECT","selection":"ACQUIRE","evidence_id":"evidence-1"}'
    )


def test_binary_auc_handles_wins_and_ties():
    assert binary_auc([True, False], [1.0, 0.0]) == 1.0
    assert binary_auc([True, False], [0.0, 1.0]) == 0.0
    assert binary_auc([True, False], [1.0, 1.0]) == 0.5


def test_binary_auc_requires_both_classes():
    with pytest.raises(ValueError, match="both selector classes"):
        binary_auc([True], [1.0])


def test_common_prefix_length_returns_first_divergent_token():
    assert common_prefix_length([1, 2, 3], [1, 2, 4, 5]) == 2
    with pytest.raises(ValueError, match="divergent decision token"):
        common_prefix_length([1, 2], [1, 2, 3])
