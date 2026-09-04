import hashlib
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.run_sn7_learned_defer_extension import (
    POLICY,
    _comparison_command,
    _evaluation_command,
    validate_evaluation_summary,
)


def _args(tmp_path):
    return Namespace(
        python="python",
        val_states=tmp_path / "states.jsonl",
        val_episodes=tmp_path / "episodes.jsonl",
        generic_selector=tmp_path / "generic.pt",
        device="cuda:0",
        max_candidates=16,
        tool_out_size=256,
        seed=20260730,
        asset_root_map=["/old=/new"],
        per_seed_bootstrap_repetitions=1,
        bootstrap_seed=7,
    )


def _controller(tmp_path):
    return {
        key: {"path": tmp_path / f"{key}.bin", "sha256": key}
        for key in ("tool_belief", "post_tool_adapter", "tool_gate", "tool_gate_summary")
    }


def test_evaluation_command_is_protocol_matched_and_uses_checkpoint_margin(tmp_path):
    command = _evaluation_command(_args(tmp_path), _controller(tmp_path), output=tmp_path / "out")
    assert command[command.index("--split") + 1] == "val"
    assert command[command.index("--policy") + 1] == POLICY
    assert command[command.index("--tool-mode") + 1] == "selective"
    assert command[command.index("--belief-mode") + 1] == "recurrent"
    assert command[command.index("--max-acquisitions") + 1] == "1"
    assert "--learned-selector-stop-margin" not in command


def test_summary_requires_hypothesis_agnostic_checkpoint_margin(tmp_path):
    checkpoint = tmp_path / "generic.pt"
    checkpoint.write_bytes(b"checkpoint")
    selector = {
        "condition_on_hypothesis": False,
        "checkpoint_stop_margin": 0.5,
        "effective_stop_margin": 0.5,
        "stop_margin_source": "checkpoint",
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    summary = {
        "split": "val",
        "protocol": {
            "test_assets_read": False,
            "evaluated_policies": [POLICY],
            "same_max_acquisitions": 1,
            "tool_mode": "selective",
            "belief_mode": "recurrent",
            "learned_selectors": {POLICY: selector},
        },
    }
    validate_evaluation_summary(summary, checkpoint)
    selector["condition_on_hypothesis"] = True
    with pytest.raises(ValueError, match="hypothesis-agnostic"):
        validate_evaluation_summary(summary, checkpoint)


def test_comparison_orientation_is_activemap_minus_learned_defer(tmp_path):
    command = _comparison_command(
        _args(tmp_path),
        active=Path("active.jsonl"),
        learned_defer=Path("defer.jsonl"),
        output=tmp_path / "comparison.json",
        manifest=tmp_path / "manifest.json",
    )
    assert command[command.index("--candidate") + 1] == "activemap"
    assert f"{POLICY}=defer.jsonl" in command
    assert "--manifest" in command
