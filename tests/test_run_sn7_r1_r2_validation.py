from argparse import Namespace
from pathlib import Path

from scripts.run_sn7_r1_r2_validation import (
    _baseline_command,
    _comparison_command,
    _rate_matched_command,
)


def _args(tmp_path):
    return Namespace(
        python="python",
        train_states=tmp_path / "train_states.jsonl",
        train_episodes=tmp_path / "train_episodes.jsonl",
        val_states=tmp_path / "val_states.jsonl",
        val_episodes=tmp_path / "val_episodes.jsonl",
        device="cuda:1",
        max_candidates=12,
        seed=7,
        tool_out_size=128,
        asset_root_map=["/old=/new"],
        bootstrap_repetitions=99,
        per_seed_bootstrap_repetitions=1,
        bootstrap_seed=13,
    )


def _controller(tmp_path):
    return {
        "selector": {"path": tmp_path / "selector.pt", "sha256": "a", "stop_margin": 0.2},
        "tool_gate": {"path": tmp_path / "gate.joblib", "sha256": "b"},
        "tool_gate_summary": {"path": tmp_path / "summary.json", "sha256": "c"},
        "tool_belief": {"path": tmp_path / "belief.pt", "sha256": "d"},
        "post_tool_adapter": {"path": tmp_path / "adapter.pt", "sha256": "e"},
    }


def test_active_and_forced_commands_preserve_full_controller_components(tmp_path):
    args = _args(tmp_path)
    controller = _controller(tmp_path)
    active = _baseline_command(
        args, controller, output=tmp_path / "active", policy="activemap", tool_mode="selective"
    )
    forced = _baseline_command(
        args, controller, output=tmp_path / "forced", policy="activemap", tool_mode="forced"
    )
    for command in (active, forced):
        assert command[command.index("--split") + 1] == "val"
        assert command[command.index("--max-acquisitions") + 1] == "1"
        assert command[command.index("--learned-selector-stop-margin") + 1].startswith("activemap=")
        assert "--tool-belief-checkpoint" in command
        assert "--post-tool-action-adapter" in command
    assert active[active.index("--tool-mode") + 1] == "selective"
    assert "--tool-gate" in active
    assert forced[forced.index("--tool-mode") + 1] == "forced"
    assert "--tool-gate" not in forced


def test_stop_command_is_validation_only_and_has_no_controller_dependency(tmp_path):
    command = _baseline_command(
        _args(tmp_path),
        _controller(tmp_path),
        output=tmp_path / "stop",
        policy="always_stop",
        tool_mode="none",
    )
    assert command[command.index("--split") + 1] == "val"
    assert command[command.index("--policy") + 1] == "always_stop"
    assert "--learned-selector" not in command
    assert "--tool-mode" not in command


def test_r2_comparison_is_manifest_bound_and_includes_candidate(tmp_path):
    args = _args(tmp_path)
    command = _comparison_command(
        args,
        output=tmp_path / "comparison.json",
        records={"activemap": Path("active.jsonl"), "always_stop": Path("stop.jsonl")},
        manifest=tmp_path / "manifest.json",
        bootstrap_repetitions=1,
    )
    assert command[command.index("--candidate") + 1] == "activemap"
    assert command[command.index("--manifest") + 1].endswith("manifest.json")
    assert command.count("--records") == 2


def test_rate_matched_uses_hashed_train_receipt_and_trace(tmp_path):
    command = _rate_matched_command(
        _args(tmp_path),
        output=tmp_path / "rate",
        receipt=tmp_path / "receipt.json",
        trace=tmp_path / "train.jsonl",
        bootstrap_repetitions=1,
    )
    assert command[command.index("--target-rate-receipt") + 1].endswith("receipt.json")
    assert command[command.index("--target-train-trace") + 1].endswith("train.jsonl")
    assert "--val-manifest" not in command
