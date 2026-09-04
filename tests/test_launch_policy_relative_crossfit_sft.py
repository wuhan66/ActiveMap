from argparse import Namespace
from pathlib import Path

from scripts.launch_policy_relative_crossfit_sft import fold_command


def test_fold_command_forwards_manifest_counts_and_fixed_seed():
    args = Namespace(
        python="python",
        model=Path("model"),
        folds_root=Path("folds"),
        seed=17,
        epochs=3.0,
        learning_rate=1e-4,
        batch_size=1,
        gradient_accumulation=16,
        eval_steps=50,
        save_steps=50,
        early_stopping_patience=3,
    )
    spec = {
        "fold": 1,
        "files": {
            "train_sft": {"records": 2000, "pre_tool_positive_records": 450},
            "inner_val_sft": {"records": 188, "pre_tool_positive_records": 17},
        },
    }
    command = fold_command(args, spec, 5, Path("output"))

    assert command[command.index("--gpu") + 1] == "5"
    assert command[command.index("--seed") + 1] == "17"
    assert command[command.index("--expected-train-records") + 1] == "2000"
    assert command[command.index("--expected-val-use-tool") + 1] == "17"
    assert str(Path("folds") / "fold1" / "train_sft.jsonl") in command
