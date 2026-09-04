from argparse import Namespace
from pathlib import Path

from scripts.run_policy_relative_crossfit_pipeline import (
    assembly_command,
    evaluation_command,
    gate_command,
    labeling_command,
)


def _args(tmp_path: Path) -> Namespace:
    return Namespace(
        python="python",
        model=Path("model"),
        folds_root=Path("folds"),
        training_root=Path("training"),
        val_feature_root=Path("val-features"),
        val_branch_root=Path("val-branches"),
        output_root=tmp_path / "output",
        gpu=[4, 5],
        seed=17,
    )


def test_pipeline_commands_freeze_policy_relative_protocol(tmp_path):
    args = _args(tmp_path)
    labeling = labeling_command(args)
    assembly = assembly_command(args)
    gate = gate_command(args)
    evaluation = evaluation_command(args)

    assert labeling[labeling.index("--pooling") + 1] == "last_mean"
    assert labeling.count("--gpu") == 2
    assert str(args.val_branch_root) in assembly
    assert gate[gate.index("--selection-objective") + 1] == "proxy_utility"
    assert gate[gate.index("--fit-weighting") + 1] == "utility_magnitude"
    assert str(args.val_branch_root) in evaluation
