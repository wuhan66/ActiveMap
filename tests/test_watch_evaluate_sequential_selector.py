from argparse import Namespace
from pathlib import Path

from scripts.watch_evaluate_sequential_selector import evaluation_command


def test_evaluation_command_freezes_component_protocol():
    args = Namespace(
        python="python",
        model=Path("model"),
        val_jsonl=Path("val.jsonl"),
        output_root=Path("output"),
        seed=20260716,
    )

    command = evaluation_command(args, Path("adapter"))

    assert command[command.index("--expected-records") + 1] == "414"
    assert command[command.index("--bootstrap-repetitions") + 1] == "2000"
    assert "adapter" in command
