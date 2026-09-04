from argparse import Namespace
from pathlib import Path

from scripts.launch_policy_relative_onpolicy_cache import shard_command


def test_shard_command_freezes_total_pair_count_and_assignment():
    args = Namespace(
        python="python",
        model=Path("model"),
        adapter=Path("adapter"),
        rollout_jsonl=Path("rollout.jsonl"),
        output_root=Path("output"),
        gpu=[4, 5],
        seed=7,
        expected_pairs=12,
        resume=False,
    )

    command = shard_command(args, 1)

    assert command[command.index("--num-shards") + 1] == "2"
    assert command[command.index("--shard-index") + 1] == "1"
    assert command[command.index("--expected-pairs") + 1] == "12"
