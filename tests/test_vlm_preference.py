import json

import pytest
import torch

from activemap.agent.vlm_preference import (
    VisualPreferenceCollator,
    audit_preference_splits,
    combine_preference_weights,
    combined_preference_batch,
    dpo_loss,
    load_vlm_preference_rows,
    preference_family_weights,
    preference_safety_weights,
    preference_weighted_loss,
    sequence_log_probabilities,
)


def preference_row():
    return {
        "schema_version": "active-catalog-vlm-preference-v1",
        "example_id": "example",
        "task_id": "task-train",
        "rejected_action_key": "STOP",
        "preference_family": "acquire_over_stop",
        "split": "train",
        "prompt": [
            {"role": "system", "content": []},
            {"role": "user", "content": []},
        ],
        "chosen": {"role": "assistant", "content": []},
        "rejected": {"role": "assistant", "content": []},
        "chosen_utility": 1.0,
        "rejected_utility": 0.0,
        "model_visible_utility": False,
        "test_assets_read": False,
    }


def test_preference_loader_rejects_nonpositive_margin(tmp_path):
    path = tmp_path / "rows.jsonl"
    row = preference_row()
    row["chosen_utility"] = 0.0
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="positive"):
        load_vlm_preference_rows(path)


def test_sequence_log_probabilities_use_only_assistant_labels():
    logits = torch.tensor(
        [[[4.0, 0.0], [0.0, 4.0], [4.0, 0.0], [0.0, 4.0]]]
    )
    labels = torch.tensor([[-100, -100, 0, 1]])
    value = sequence_log_probabilities(logits, labels)
    expected = torch.log_softmax(logits[:, :-1], -1)[0, 1, 0] + torch.log_softmax(
        logits[:, :-1], -1
    )[0, 2, 1]
    assert torch.allclose(value[0], expected)


def test_dpo_loss_rewards_policy_margin_above_reference():
    improved = dpo_loss(
        torch.tensor([2.0]),
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        torch.tensor([0.0]),
        beta=0.1,
    )
    degraded = dpo_loss(
        torch.tensor([0.0]),
        torch.tensor([2.0]),
        torch.tensor([1.0]),
        torch.tensor([0.0]),
        beta=0.1,
    )
    assert improved.item() < degraded.item()


def test_preference_collator_combines_chosen_then_rejected():
    def encoded(token):
        return {
            "input_ids": torch.tensor([token, token + 1]),
            "attention_mask": torch.tensor([1, 1]),
            "labels": torch.tensor([-100, token + 1]),
        }

    batch = VisualPreferenceCollator(pad_token_id=0)(
        [
            {"chosen": encoded(10), "rejected": encoded(20)},
            {"chosen": encoded(11), "rejected": encoded(21)},
        ]
    )
    combined = combined_preference_batch(batch)
    assert combined["input_ids"][:, 0].tolist() == [10, 11, 20, 21]


def test_preference_collator_preserves_loss_weights():
    encoded = {
        "input_ids": torch.tensor([1, 2]),
        "attention_mask": torch.tensor([1, 1]),
        "labels": torch.tensor([-100, 2]),
    }
    batch = VisualPreferenceCollator(pad_token_id=0)(
        [
            {
                "chosen": encoded,
                "rejected": encoded,
                "loss_weight": torch.tensor(1.5),
            }
        ]
    )
    assert batch["pair_loss_weight"].tolist() == [1.5]


def test_inverse_frequency_weights_equalize_family_mass():
    rows = [
        {"preference_family": "stop_over_acquire"},
        {"preference_family": "stop_over_acquire"},
        {"preference_family": "acquire_over_stop"},
    ]
    weights, by_family = preference_family_weights(
        rows, mode="inverse_frequency"
    )
    assert weights is not None
    assert sum(weights[:2]) == pytest.approx(weights[2])
    assert sum(weights) / len(weights) == pytest.approx(1.0)
    assert by_family["acquire_over_stop"] > by_family["stop_over_acquire"]


def test_tempered_family_weights_interpolate_without_changing_mean():
    rows = [
        {"preference_family": "majority"},
        {"preference_family": "majority"},
        {"preference_family": "majority"},
        {"preference_family": "majority"},
        {"preference_family": "minority"},
    ]
    weights, by_family = preference_family_weights(
        rows, mode="inverse_frequency", power=0.5
    )
    assert weights is not None
    assert sum(weights) / len(weights) == pytest.approx(1.0)
    assert 1.0 < by_family["minority"] < 2.5
    assert 0.625 < by_family["majority"] < 1.0


def test_family_balance_power_is_bounded():
    rows = [{"preference_family": "only"}]
    with pytest.raises(ValueError, match="between zero and one"):
        preference_family_weights(rows, mode="inverse_frequency", power=1.1)


def test_preference_weight_survives_batch_size_one():
    loss = torch.tensor([2.0])
    assert preference_weighted_loss(loss).item() == pytest.approx(2.0)
    assert preference_weighted_loss(loss, torch.tensor([0.5])).item() == pytest.approx(
        1.0
    )


def test_safety_weights_prioritize_resolving_false_edits():
    rows = [
        {
            "chosen_action_key": "ACQUIRE",
            "chosen_false_edit_rate": 0.0,
            "rejected_false_edit_rate": 1.0,
        },
        {
            "chosen_action_key": "STOP",
            "chosen_false_edit_rate": 1.0,
            "rejected_false_edit_rate": 1.0,
        },
        {
            "chosen_action_key": "STOP",
            "chosen_false_edit_rate": 0.0,
            "rejected_false_edit_rate": 0.0,
        },
    ]
    weights, summary = preference_safety_weights(
        rows, safe_acquire_boost=2.0, unsafe_stop_scale=0.25
    )
    assert weights is not None
    assert sum(weights) / len(weights) == pytest.approx(1.0)
    assert weights[0] > weights[2] > weights[1]
    assert summary["boosted_pairs"] == 1
    assert summary["downweighted_unsafe_stop_pairs"] == 1


def test_safety_weight_arguments_are_bounded():
    with pytest.raises(ValueError, match="nonnegative"):
        preference_safety_weights([], safe_acquire_boost=-1.0)
    with pytest.raises(ValueError, match=r"in \(0, 1\]"):
        preference_safety_weights([], unsafe_stop_scale=0.0)


def test_combined_preference_weights_preserve_global_mean():
    weights = combine_preference_weights([0.5, 1.5], [2.0, 1.0])
    assert weights is not None
    assert sum(weights) / len(weights) == pytest.approx(1.0)
    assert weights[1] > weights[0]


def test_split_audit_rejects_model_visible_target_metadata():
    train = preference_row()
    train["prompt"][1]["content"] = [
        {"type": "text", "text": json.dumps({"target_evidence_id": "hidden"})}
    ]
    val = preference_row()
    val.update(
        {
            "split": "val",
            "example_id": "val-example",
            "task_id": "task-val",
        }
    )
    with pytest.raises(ValueError, match="target metadata"):
        audit_preference_splits([train], [val])


def test_split_audit_rejects_task_overlap():
    train = preference_row()
    val = preference_row()
    val.update({"split": "val", "example_id": "val-example"})
    with pytest.raises(ValueError, match="task overlap"):
        audit_preference_splits([train], [val])
