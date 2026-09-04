import importlib.util
from pathlib import Path

import yaml

SCRIPT = Path(__file__).parents[1] / "scripts" / "derive_online_candidate_registry.py"
SPEC = importlib.util.spec_from_file_location("derive_candidate_registry", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_derive_records_explicit_candidate_method_and_purpose(tmp_path: Path) -> None:
    base_registry = tmp_path / "base.yaml"
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"candidate checkpoint")
    base_registry.write_text(
        yaml.safe_dump(
            {
                "test_assets_read": False,
                "dataset": "sn7",
                "protocol": {"image_size": 512},
                "seed_artifacts": {"7": {"selector": {"path": "selector.pt"}}},
                "online_state_contract": {"version": "v1"},
            }
        ),
        encoding="utf-8",
    )

    registry = MODULE.derive(
        base_registry,
        checkpoint,
        controller_seed=7,
        method="carried_replay_updater_headroom",
        purpose="train_internal_real_carried_replay_raw_headroom_gate",
    )

    assert registry["method"] == "carried_replay_updater_headroom"
    assert registry["provenance"]["candidate_updater"]["purpose"] == (
        "train_internal_real_carried_replay_raw_headroom_gate"
    )
    assert registry["shared_artifacts"][0]["path"] == str(checkpoint.resolve())
