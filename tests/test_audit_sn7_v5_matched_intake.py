from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.audit_sn7_v5_matched_intake import audit


SEEDS = (20260817, 20260818, 20260819)
POLICIES = (
    "direct_commit",
    "direct_safe_commit",
    "selected_commit",
    "selected_safe_commit",
    "forced_safe_commit",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _rows(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "task_id": f"task-{operation.lower()}",
            "aoi_id": "aoi-a",
            "budget": 1.5,
            "target": f"COMMIT:{operation}",
            "split": "val",
            "test_assets_read": False,
        }
        for operation in ("ADD", "DELETE", "RESHAPE")
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _rollout_receipt(root: Path, seed: int, split: str, state_hash: str, selector_hash: str, auth_hash: str) -> None:
    directory = root / "rollouts" / f"seed{seed}_{split}"
    paths = {}
    for policy in ("direct", "selected", "forced"):
        path = directory / f"{policy}_rollouts.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"row": 1}\n', encoding="utf-8")
        paths[policy] = path
    _json(
        directory / "receipt.json",
        {
            "split": split,
            "test_assets_read": False,
            "authorization_sha256": auth_hash,
            "selector_checkpoint_sha256": selector_hash,
            "states_sha256": state_hash,
            **{
                f"{policy}_rollouts": str(path.resolve())
                for policy, path in paths.items()
            },
            **{
                f"{policy}_rollouts_sha256": _sha(path)
                for policy, path in paths.items()
            },
        },
    )


def _build_complete_run(root: Path) -> None:
    validation_hashes = {seed: hashlib.sha256(f"val-{seed}".encode()).hexdigest() for seed in SEEDS}
    authorization = _json(
        root / "authorization" / "three_seed_headroom_authorization.json",
        {
            "schema_version": "sn7-v5b-three-seed-headroom-authorization-v1",
            "authorization": "matched_nonkeep_factorial_writeback",
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
    auth_hash = _sha(authorization)
    for seed in SEEDS:
        state_root = root / "selector_states"
        train_states = state_root / f"seed{seed}_train.jsonl"
        train_states.parent.mkdir(parents=True, exist_ok=True)
        train_states.write_text('{"split":"train"}\n', encoding="utf-8")
        internal_dir = state_root / f"seed{seed}_internal"
        outputs = {}
        for name in ("fit", "tune", "fit_tune"):
            path = internal_dir / f"{name}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"split":"train"}\n', encoding="utf-8")
            outputs[name] = {"path": str(path.resolve()), "sha256": _sha(path)}
        _json(
            internal_dir / "summary.json",
            {
                "schema_version": "sn7-v5-selector-train-internal-split-v1",
                "formal_validation_assets_read": False,
                "test_assets_read": False,
                "source": {"original_split": "train", "sha256": _sha(train_states)},
                "split": {"source_episode_overlap": 0},
                "outputs": outputs,
            },
        )
        selector_dir = root / "selectors" / f"seed{seed}"
        selector_dir.mkdir(parents=True, exist_ok=True)
        selector = selector_dir / "best.pt"
        selector.write_bytes(f"selector-{seed}".encode())
        _json(selector_dir / "metrics.json", {"calibrate_stop_margin": True})
        _rollout_receipt(root, seed, "train", _sha(train_states), _sha(selector), auth_hash)
        _rollout_receipt(root, seed, "val", validation_hashes[seed], _sha(selector), auth_hash)

        train_writebacks = root / "writebacks" / f"seed{seed}" / "train"
        calibration_inputs = {}
        for policy in ("direct", "selected"):
            path = train_writebacks / f"{policy}_commit" / "writeback.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"train": true}\n', encoding="utf-8")
            calibration_inputs[policy] = {"sha256": _sha(path)}
        _json(
            root / "train_calibration" / f"seed{seed}.json",
            {
                "schema_version": "sn7-v5-safe-commit-calibration-v1",
                "split": "train",
                "test_assets_read": False,
                "inputs": calibration_inputs,
            },
        )
        val_writebacks = root / "writebacks" / f"seed{seed}" / "val"
        for policy in (*POLICIES, "forced_commit"):
            _rows(val_writebacks / policy / "writeback.jsonl")

    _json(
        root / "three_seed_nonkeep_factorial_with_forced_summary.json",
        {
            "schema_version": "sn7-v5-matched-nonkeep-factorial-v2",
            "split": "val",
            "test_assets_read": False,
            "model_seeds": list(SEEDS),
            "policies": list(POLICIES),
            "operation_slices": {
                operation: {
                    factor: {"row_count_per_seed": {str(seed): 1 for seed in SEEDS}}
                    for factor in ("selection_factor", "safe_commit_factor")
                }
                for operation in ("ADD", "DELETE", "RESHAPE")
            },
            "promotion": {"forced_acquisition_cost_control": {"paired_delta": {}}},
        },
    )


def test_audit_accepts_complete_validation_only_five_policy_run(tmp_path: Path) -> None:
    _build_complete_run(tmp_path)

    result = audit(tmp_path)

    assert result["passed"] is True
    assert set(result["seeds"]) == {str(seed) for seed in SEEDS}
    assert result["test_assets_read"] is False


def test_audit_rejects_test_contaminated_aggregate(tmp_path: Path) -> None:
    _build_complete_run(tmp_path)
    summary_path = tmp_path / "three_seed_nonkeep_factorial_with_forced_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["test_assets_read"] = True
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="test_assets_read=false"):
        audit(tmp_path)
