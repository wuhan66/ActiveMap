from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.prepare_sn7_v5_matched_resume import SEEDS
from scripts.prepare_sn7_v5_matched_writeback_resume import validate_writeback_resume


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _state(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"split":"train"}\n', encoding="utf-8")


def _internal(root: Path, seed: int) -> None:
    state_root = root / "selector_states"
    train = state_root / f"seed{seed}_train.jsonl"
    directory = state_root / f"seed{seed}_internal"
    outputs = {}
    for name, split in (("fit", "train"), ("tune", "val"), ("fit_tune", "train")):
        path = directory / f"{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"split": split}) + "\n", encoding="utf-8")
        outputs[name] = {"sha256": _sha(path)}
    _json(
        directory / "summary.json",
        {
            "schema_version": "sn7-v5-selector-train-internal-split-v1",
            "formal_validation_assets_read": False,
            "test_assets_read": False,
            "source": {"original_split": "train", "sha256": _sha(train)},
            "split": {"seed": seed, "source_episode_overlap": 0},
            "outputs": outputs,
        },
    )
    _json(state_root / f"seed{seed}_audit.json", {"splits": {"train": 1, "val": 1}, "test_assets_read": False})


def _receipt(root: Path, seed: int, split: str, selector: Path, state_hash: str, auth_hash: str) -> None:
    directory = root / "rollouts" / f"seed{seed}_{split}"
    payload = {
        "schema_version": "sn7-v5-matched-rollout-receipt-v1",
        "split": split,
        "test_assets_read": False,
        "authorization_sha256": auth_hash,
        "selector_checkpoint_sha256": _sha(selector),
        "states_sha256": state_hash,
    }
    for policy in ("direct", "selected", "forced"):
        path = directory / f"{policy}_rollouts.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"row":1}\n', encoding="utf-8")
        payload[f"{policy}_rollouts"] = str(path.resolve())
        payload[f"{policy}_rollouts_sha256"] = _sha(path)
    _json(directory / "receipt.json", payload)


def _complete_rollout_root(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    _json(root / "queue_status.json", {"status": "resuming", "test_assets_read": False})
    validation_hashes = {seed: hashlib.sha256(f"val-{seed}".encode()).hexdigest() for seed in SEEDS}
    authorization_path = root / "authorization" / "three_seed_headroom_authorization.json"
    _json(
        authorization_path,
        {
            "test_assets_read": False,
            "records": [
                {
                    "seed": seed,
                    "passed": True,
                    "test_assets_read": False,
                    "state_file_sha256": validation_hashes[seed],
                }
                for seed in SEEDS
            ],
        },
    )
    for seed in SEEDS:
        _state(root / "selector_states" / f"seed{seed}_train.jsonl")
        _internal(root, seed)
        selector = root / "selectors" / f"seed{seed}" / "best.pt"
        selector.parent.mkdir(parents=True, exist_ok=True)
        selector.write_bytes(f"selector-{seed}".encode())
        _json(selector.parent / "metrics.json", {"checkpoint": "best"})
        _receipt(root, seed, "train", selector, _sha(root / "selector_states" / f"seed{seed}_train.jsonl"), _sha(authorization_path))
        _receipt(root, seed, "val", selector, validation_hashes[seed], _sha(authorization_path))
    return root


def test_writeback_resume_accepts_hashed_pre_writeback_artifacts(tmp_path: Path) -> None:
    result = validate_writeback_resume(_complete_rollout_root(tmp_path))

    assert result["test_assets_read"] is False
    assert set(result["seeds"]) == {str(seed) for seed in SEEDS}


def test_writeback_resume_allows_empty_failed_writeback_directories(tmp_path: Path) -> None:
    root = _complete_rollout_root(tmp_path)
    (root / "writebacks" / "seed20260817" / "train" / "direct_commit").mkdir(
        parents=True
    )

    result = validate_writeback_resume(root)

    assert result["test_assets_read"] is False


def test_writeback_resume_rejects_partial_writeback(tmp_path: Path) -> None:
    root = _complete_rollout_root(tmp_path)
    partial = root / "writebacks" / "seed20260817" / "train" / "direct_commit" / "progress.json"
    partial.parent.mkdir(parents=True)
    partial.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="partial later-phase artifact"):
        validate_writeback_resume(root)
