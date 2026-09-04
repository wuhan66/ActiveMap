from argparse import Namespace
from pathlib import Path

from scripts.launch_active_catalog_writeback import evaluator_command


def test_writeback_command_preserves_smoke_limit_and_root_map():
    args = Namespace(
        python="python",
        checkpoint=Path("checkpoint.pt"),
        episodes=Path("episodes.jsonl"),
        rollouts=Path("rollouts.jsonl"),
        image_size=512,
        threshold=0.5,
        simplify_tolerance=0.0,
        min_delta_component_pixels=0,
        protocol_name="sn7-active-catalog-vector-writeback-v1",
        asset_root_map=["/old=/new"],
        limit=2,
    )
    command = evaluator_command(args, Path("out"))
    assert command[1] == "scripts/evaluate_agent_map_writeback.py"
    assert command[command.index("--delta-margin") + 1] == "0.0"
    assert command[command.index("--confidence-floor") + 1] == "0.0"
    assert command[-4:] == ["--asset-root-map", "/old=/new", "--limit", "2"]


def test_writeback_command_forwards_frozen_test_authorization():
    args = Namespace(
        python="python",
        checkpoint=Path("checkpoint.pt"),
        episodes=Path("episodes.jsonl"),
        rollouts=Path("rollouts.jsonl"),
        image_size=512,
        threshold=0.5,
        simplify_tolerance=0.0,
        min_delta_component_pixels=0,
        protocol_name="sn7-frozen-test-v1",
        asset_root_map=[],
        limit=None,
        split="test",
        frozen_test=True,
    )
    command = evaluator_command(args, Path("out"))
    assert command[command.index("--split") + 1] == "test"
    assert "--frozen-test" in command
