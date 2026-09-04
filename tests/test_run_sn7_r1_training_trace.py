import hashlib
import json
from argparse import Namespace

import pytest
import yaml

from scripts.run_sn7_r1_training_trace import (
    evaluator_command,
    load_frozen_controller,
    validate_trace,
)


def _write(path, contents="asset"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _registry(tmp_path):
    storage = tmp_path / "storage"
    values = {}
    for name, filename in {
        "selector": "selector.pt",
        "tool_gate": "gate.joblib",
        "tool_belief": "belief.pt",
        "post_tool_adapter": "adapter.pt",
    }.items():
        path = storage / "runs" / filename
        values[name] = {"path": "${STORAGE_ROOT}/runs/" + filename, "sha256": _write(path)}
    values["selector"]["stop_margin"] = 0.25
    _write(storage / "runs" / "summary.json", "{}")
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "test_assets_read": False,
                "seed_artifacts": {"7": values},
            }
        ),
        encoding="utf-8",
    )
    return registry, storage


def test_r1_launcher_command_uses_full_controller_train_only_and_single_acquire(tmp_path):
    registry, storage = _registry(tmp_path)
    controller = load_frozen_controller(
        registry, 7, storage_root=storage, project_root=tmp_path
    )
    args = Namespace(
        python="python",
        states=tmp_path / "states.jsonl",
        episodes=tmp_path / "episodes.jsonl",
        device="cuda:1",
        max_candidates=12,
        bootstrap_repetitions=0,
        seed=7,
        tool_out_size=128,
        asset_root_map=["/old=/new"],
    )
    command = evaluator_command(args, controller, tmp_path / "output")
    assert command[1] == "scripts/evaluate_active_catalog_closed_loop_baselines.py"
    assert command[command.index("--split") + 1] == "train"
    assert command[command.index("--max-acquisitions") + 1] == "1"
    assert command[command.index("--policy") + 1] == "activemap"
    assert command[command.index("--tool-mode") + 1] == "selective"
    assert command[command.index("--belief-mode") + 1] == "recurrent"
    assert command[command.index("--learned-selector") + 1].startswith("activemap=")
    assert "--tool-gate" in command
    assert "--tool-belief-checkpoint" in command
    assert "--post-tool-action-adapter" in command


def test_r1_registry_rejects_unregistered_component_hash(tmp_path):
    registry, storage = _registry(tmp_path)
    (storage / "runs" / "selector.pt").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_frozen_controller(registry, 7, storage_root=storage, project_root=tmp_path)


def test_r1_launcher_trace_validator_rejects_noncompliant_rows(tmp_path):
    trace = tmp_path / "trace.jsonl"
    valid = {
        "split": "train",
        "policy": "activemap",
        "test_assets_read": False,
        "acquisitions": 1,
    }
    trace.write_text(json.dumps(valid) + "\n", encoding="utf-8")
    validate_trace(trace)

    for field, value in (("split", "val"), ("policy", "random"), ("acquisitions", 2)):
        invalid = {**valid, field: value}
        trace.write_text(json.dumps(invalid) + "\n", encoding="utf-8")
        with pytest.raises(ValueError):
            validate_trace(trace)
