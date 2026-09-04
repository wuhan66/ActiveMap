from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.prepare_sn7_v5_matched_resume import SEEDS, validate_resume


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _state(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"split":"train"}\n', encoding="utf-8")


def _internal(root: Path, seed: int, *, with_audit: bool = True) -> None:
    state_root = root / "selector_states"
    source = state_root / f"seed{seed}_train.jsonl"
    directory = state_root / f"seed{seed}_internal"
    outputs = {}
    for name, split in (("fit", "train"), ("tune", "val"), ("fit_tune", "train")):
        artifact = directory / f"{name}.jsonl"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(json.dumps({"split": split}) + "\n", encoding="utf-8")
        outputs[name] = {"sha256": _sha(artifact)}
    _json(
        directory / "summary.json",
        {
            "schema_version": "sn7-v5-selector-train-internal-split-v1",
            "formal_validation_assets_read": False,
            "test_assets_read": False,
            "source": {"original_split": "train", "sha256": _sha(source)},
            "split": {"seed": seed, "source_episode_overlap": 0},
            "outputs": outputs,
        },
    )
    if with_audit:
        _json(
            state_root / f"seed{seed}_audit.json",
            {"splits": {"train": 1, "val": 1}, "test_assets_read": False},
        )


def _run_root(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    _json(root / "queue_status.json", {"status": "starting", "test_assets_read": False})
    _json(
        root / "authorization" / "three_seed_headroom_authorization.json",
        {
            "test_assets_read": False,
            "records": [
                {"seed": seed, "passed": True, "test_assets_read": False} for seed in SEEDS
            ],
        },
    )
    for seed in SEEDS:
        _state(root / "selector_states" / f"seed{seed}_train.jsonl")
    return root


def test_resume_preflight_reuses_only_hashed_train_artifacts(tmp_path: Path) -> None:
    root = _run_root(tmp_path)
    _internal(root, 20260817)

    result = validate_resume(root)

    assert result["test_assets_read"] is False
    assert result["missing_internal_split_seeds"] == [20260818, 20260819]
    assert set(result["validated_existing_internal_splits"]) == {"20260817"}


def test_resume_preflight_rejects_later_phase_artifact(tmp_path: Path) -> None:
    root = _run_root(tmp_path)
    _internal(root, 20260817)
    checkpoint = root / "selectors" / "seed20260817" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"partial")

    with pytest.raises(ValueError, match="resume phase is ambiguous"):
        validate_resume(root)
