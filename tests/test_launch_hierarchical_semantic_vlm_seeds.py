from argparse import Namespace

import pytest

from scripts.launch_hierarchical_semantic_vlm_seeds import (
    assignments,
    validate_rollout_contract,
)


def test_hierarchical_seed_assignments_respect_two_gpu_round_robin():
    assert assignments([16, 19, 22], [5, 7]) == [(16, 5), (19, 7), (22, 5)]


def test_rollout_contract_requires_complete_pre_post_pairs():
    args = Namespace(
        expected_rollout_records=828,
        expected_rollout_pre=414,
        expected_rollout_post=414,
    )
    valid = {
        "records": 828,
        "stage_counts": {"PRE_TOOL": 414, "POST_TOOL": 414},
        "invalid_action_records": 0,
    }
    validate_rollout_contract(args, valid)

    invalid = {**valid, "stage_counts": {"PRE_TOOL": 414, "POST_TOOL": 413}}
    with pytest.raises(ValueError, match="rollout.POST_TOOL"):
        validate_rollout_contract(args, invalid)
