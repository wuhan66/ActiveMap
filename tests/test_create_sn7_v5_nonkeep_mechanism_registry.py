from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.create_sn7_v5_nonkeep_mechanism_registry import SEEDS, build_registry


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _setup(tmp_path: Path) -> dict[str, object]:
    authorization = tmp_path / "authorization.json"
    authorization.write_text(
        json.dumps(
            {
                "schema_version": "sn7-v5b-three-seed-headroom-authorization-v1",
                "authorization": "matched_nonkeep_factorial_writeback",
                "test_assets_read": False,
                "records": [
                    {
                        "seed": seed,
                        "passed": True,
                        "test_assets_read": False,
                        "state_file_sha256": hashlib.sha256(f"val-{seed}".encode()).hexdigest(),
                    }
                    for seed in SEEDS
                ],
            }
        ),
        encoding="utf-8",
    )
    generic = Path("configs/selector/sn7_v5_nonkeep_generic_utility_v1_server.yaml")
    policy_relative = Path("configs/selector/sn7_v5_nonkeep_policy_relative_utility_v1_server.yaml")
    states = []
    checkpoints = []
    for seed in SEEDS:
        states.append(
            (
                seed,
                _write(
                    tmp_path / f"state-{seed}.jsonl",
                    json.dumps(
                        {
                            "split": "train",
                            "test_assets_read": False,
                            "source_episode": f"episode-{seed}",
                        }
                    )
                    + "\n",
                ),
            )
        )
        checkpoints.append((seed, _write(tmp_path / f"updater-{seed}.pt", f"checkpoint-{seed}")))
    return {
        "authorization": authorization,
        "generic": generic,
        "policy_relative": policy_relative,
        "states": states,
        "checkpoints": checkpoints,
    }


def test_registry_locks_matched_templates_and_all_three_seeds(tmp_path: Path) -> None:
    setup = _setup(tmp_path)
    result = build_registry(
        authorization_path=setup["authorization"],  # type: ignore[arg-type]
        generic_template=setup["generic"],  # type: ignore[arg-type]
        policy_relative_template=setup["policy_relative"],  # type: ignore[arg-type]
        train_states=setup["states"],  # type: ignore[arg-type]
        updater_checkpoints=setup["checkpoints"],  # type: ignore[arg-type]
        bootstrap_seed=7,
    )

    assert result["test_assets_read"] is False
    assert result["selector_templates"]["capacity_matched"] is True
    assert result["common_safe_commit"]["policy_blind"] is True
    assert set(result["train_states"]) == {str(seed) for seed in SEEDS}


def test_registry_rejects_nonmatching_capacity_templates(tmp_path: Path) -> None:
    setup = _setup(tmp_path)
    mutated = tmp_path / "policy_relative_mutated.yaml"
    text = Path(setup["policy_relative"]).read_text(encoding="utf-8")  # type: ignore[arg-type]
    _write(mutated, text.replace("hidden_dim: 128", "hidden_dim: 192"))

    with pytest.raises(ValueError, match="differ outside"):
        build_registry(
            authorization_path=setup["authorization"],  # type: ignore[arg-type]
            generic_template=setup["generic"],  # type: ignore[arg-type]
            policy_relative_template=mutated,
            train_states=setup["states"],  # type: ignore[arg-type]
            updater_checkpoints=setup["checkpoints"],  # type: ignore[arg-type]
            bootstrap_seed=7,
        )
