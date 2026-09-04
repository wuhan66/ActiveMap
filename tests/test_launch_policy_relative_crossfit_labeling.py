from argparse import Namespace
from pathlib import Path

from scripts.launch_policy_relative_crossfit_labeling import branch_command, feature_command


def _args() -> Namespace:
    return Namespace(
        python="python",
        model=Path("model"),
        seed=11,
        feature_batch_size=2,
        pooling="last_mean",
    )


def test_labeling_commands_preserve_fold_adapter_and_resume_contract():
    args = _args()
    branch = branch_command(
        args,
        2,
        Path("adapter"),
        Path("rollout.jsonl"),
        Path("branches"),
        855,
        resume=True,
    )
    feature = feature_command(
        args,
        Path("adapter"),
        Path("rollout.jsonl"),
        Path("features"),
    )

    assert branch[branch.index("--expected-pairs") + 1] == "855"
    assert branch[branch.index("--seed") + 1] == "11"
    assert "--resume" in branch
    assert feature[feature.index("--pooling") + 1] == "last_mean"
    assert feature[feature.index("--batch-size") + 1] == "2"
