import torch

from activemap.agent.vlm_rl import (
    contextual_policy_loss,
    sample_rl_training_rows,
    select_hard_actions,
    sequence_log_probabilities,
)


def test_contextual_policy_loss_rewards_better_action_probability():
    utilities = torch.tensor([0.0, 1.0])
    reference = torch.tensor([0.0, 0.0])
    bad_loss, _ = contextual_policy_loss(
        torch.tensor([2.0, 0.0]), reference, utilities,
        kl_beta=0.0, entropy_weight=0.0, temperature=1.0,
    )
    good_loss, diagnostics = contextual_policy_loss(
        torch.tensor([0.0, 2.0]), reference, utilities,
        kl_beta=0.0, entropy_weight=0.0, temperature=1.0,
    )
    assert good_loss < bad_loss
    assert diagnostics["best_action_probability"] > 0.5


def test_hard_action_subset_retains_stop_and_optimum():
    row = {
        "target_action_key": "ACQUIRE:e3",
        "actions": [
            {"key": "STOP", "utility": 0.1},
            {"key": "ACQUIRE:e1", "utility": 0.2},
            {"key": "ACQUIRE:e2", "utility": 0.3},
            {"key": "ACQUIRE:e3", "utility": 0.9},
        ],
    }
    selected = select_hard_actions(row, 3)
    assert {item["key"] for item in selected} == {"STOP", "ACQUIRE:e2", "ACQUIRE:e3"}


def test_pairwise_loss_prefers_acquire_over_stop():
    utilities = torch.tensor([0.0, 1.0])
    reference = torch.zeros(2)
    bad_loss, _ = contextual_policy_loss(
        torch.tensor([1.0, 0.0]), reference, utilities,
        kl_beta=0.0, entropy_weight=0.0, temperature=1.0,
        pairwise_weight=1.0, stop_index=0, target_index=1,
    )
    good_loss, diagnostics = contextual_policy_loss(
        torch.tensor([0.0, 1.0]), reference, utilities,
        kl_beta=0.0, entropy_weight=0.0, temperature=1.0,
        pairwise_weight=1.0, stop_index=0, target_index=1,
    )
    assert good_loss < bad_loss
    assert diagnostics["pairwise_loss"] < 1.0


def test_length_normalized_action_score_removes_token_count_bias():
    logits = torch.zeros(2, 4, 3)
    labels = torch.tensor([[-100, 0, -100, -100], [-100, 0, 0, 0]])
    summed = sequence_log_probabilities(logits, labels)
    normalized = sequence_log_probabilities(logits, labels, normalize_by_length=True)
    assert summed[1] < summed[0]
    assert torch.allclose(normalized[0], normalized[1])


def test_balanced_sampling_targets_requested_acquire_fraction():
    rows = [
        *[{"target_action_key": "STOP", "task_id": f"s{i}"} for i in range(10)],
        *[{"target_action_key": "ACQUIRE:e", "task_id": f"a{i}"} for i in range(2)],
    ]
    sampled = sample_rl_training_rows(
        rows, limit=10, acquire_fraction=0.5, seed=7
    )
    assert len(sampled) == 10
    assert sum(row["target_action_key"].startswith("ACQUIRE:") for row in sampled) == 5
