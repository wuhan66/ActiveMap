from argparse import Namespace
from pathlib import Path

import pytest

from scripts.run_hierarchical_vlm_posttrain_pipeline import (
    hierarchical_command,
    validate_hidden_smoke,
)


def _args() -> Namespace:
    return Namespace(
        python="python",
        model=Path("model"),
        training_root=Path("train"),
        train_sft=Path("train.jsonl"),
        val_sft=Path("val.jsonl"),
        rollout_jsonl=Path("rollout.jsonl"),
        evaluation_root=Path("eval"),
        feature_batch_size=2,
        pooling="last_mean",
        gate_selection_objective="proxy_utility",
        gate_fit_weighting="utility_magnitude",
        gpu=[5],
        seed=[20260716],
        expected_train_records=3330,
        expected_val_records=484,
        expected_train_use_tool=786,
        expected_val_use_tool=70,
        expected_rollout_records=828,
        expected_rollout_pre=414,
        expected_rollout_post=414,
    )


def test_hierarchical_command_forwards_frozen_contract():
    command = hierarchical_command(_args())
    assert command[command.index("--pooling") + 1] == "last_mean"
    assert command[command.index("--gate-selection-objective") + 1] == "proxy_utility"
    assert command[command.index("--gate-fit-weighting") + 1] == "utility_magnitude"
    assert command[command.index("--expected-train-records") + 1] == "3330"
    assert command[command.index("--expected-rollout-post") + 1] == "414"


def test_hidden_state_smoke_checks_visual_prompt_contract():
    summary = {
        "sample_count": 2,
        "pooling": "last_mean",
        "assistant_tokens_seen": False,
        "test_assets_read": False,
        "feature_dim": 8192,
    }
    validate_hidden_smoke(summary, "last_mean")
    summary["assistant_tokens_seen"] = True
    with pytest.raises(ValueError, match="assistant_tokens_seen"):
        validate_hidden_smoke(summary, "last_mean")
