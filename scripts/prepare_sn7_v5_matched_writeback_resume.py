#!/usr/bin/env python3
"""Fail-closed preflight for resuming SN7 V5 at raw writeback.

This phase is allowed only after all registered selectors and paired rollouts
are immutable and before any writeback row, calibration, aggregate, or intake
artifact exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.prepare_sn7_v5_matched_resume import (
    SEEDS,
    _require_false,
    load_json,
    sha256,
    validate_internal_split,
)


def _require_empty(path: Path) -> None:
    if path.exists() and (
        not path.is_dir() or any(child.is_file() for child in path.rglob("*"))
    ):
        raise ValueError(f"V5 writeback resume found partial later-phase artifact: {path}")


def _validate_rollout_receipt(
    root: Path,
    *,
    seed: int,
    split: str,
    selector: Path,
    authorization_hash: str,
) -> dict[str, Any]:
    receipt_path = root / "rollouts" / f"seed{seed}_{split}" / "receipt.json"
    receipt = load_json(receipt_path)
    if receipt.get("schema_version") != "sn7-v5-matched-rollout-receipt-v1":
        raise ValueError(f"unexpected V5 rollout schema: {receipt_path}")
    if receipt.get("split") != split:
        raise ValueError(f"V5 rollout split mismatch: {receipt_path}")
    _require_false(receipt, "test_assets_read", path=receipt_path)
    if receipt.get("authorization_sha256") != authorization_hash:
        raise ValueError(f"V5 rollout authorization hash mismatch: {receipt_path}")
    if receipt.get("selector_checkpoint_sha256") != sha256(selector):
        raise ValueError(f"V5 rollout selector hash mismatch: {receipt_path}")
    for policy in ("direct", "selected", "forced"):
        rollout_path = Path(str(receipt.get(f"{policy}_rollouts", "")))
        if not rollout_path.is_file() or receipt.get(f"{policy}_rollouts_sha256") != sha256(
            rollout_path
        ):
            raise ValueError(f"V5 {split} {policy} rollout hash mismatch: {receipt_path}")
    return receipt


def validate_writeback_resume(root: Path) -> dict[str, Any]:
    status_path = root / "queue_status.json"
    status = load_json(status_path)
    _require_false(status, "test_assets_read", path=status_path)
    if status.get("status") not in {"resuming", "resuming_writeback"}:
        raise ValueError(f"V5 writeback resume is not allowed from status={status.get('status')!r}")

    authorization_path = root / "authorization" / "three_seed_headroom_authorization.json"
    authorization = load_json(authorization_path)
    _require_false(authorization, "test_assets_read", path=authorization_path)
    records = authorization.get("records")
    if not isinstance(records, list):
        raise ValueError("V5 authorization lacks seed records")
    records_by_seed = {int(record.get("seed", -1)): record for record in records}
    if set(records_by_seed) != set(SEEDS):
        raise ValueError("V5 authorization does not match the registered seed set")
    for seed, record in records_by_seed.items():
        if record.get("passed") is not True or record.get("test_assets_read") is not False:
            raise ValueError(f"V5 authorization is not passing and test-isolated for seed {seed}")
    authorization_hash = sha256(authorization_path)

    seed_receipts = {}
    for seed in SEEDS:
        split_receipt = validate_internal_split(root, seed)
        if split_receipt is None:
            raise ValueError(f"V5 writeback resume lacks internal split for seed {seed}")
        selector = root / "selectors" / f"seed{seed}" / "best.pt"
        metrics = root / "selectors" / f"seed{seed}" / "metrics.json"
        if not selector.is_file() or not metrics.is_file():
            raise FileNotFoundError(f"V5 writeback resume lacks selector artifact for seed {seed}")
        train = _validate_rollout_receipt(
            root,
            seed=seed,
            split="train",
            selector=selector,
            authorization_hash=authorization_hash,
        )
        validation = _validate_rollout_receipt(
            root,
            seed=seed,
            split="val",
            selector=selector,
            authorization_hash=authorization_hash,
        )
        if validation.get("states_sha256") != records_by_seed[seed].get("state_file_sha256"):
            raise ValueError(f"V5 validation rollout state is not authorized for seed {seed}")
        seed_receipts[str(seed)] = {
            "internal_split": split_receipt,
            "selector_checkpoint_sha256": sha256(selector),
            "selector_metrics_sha256": sha256(metrics),
            "train_rollout_receipt_sha256": sha256(
                root / "rollouts" / f"seed{seed}_train" / "receipt.json"
            ),
            "validation_rollout_receipt_sha256": sha256(
                root / "rollouts" / f"seed{seed}_val" / "receipt.json"
            ),
            "train_state_sha256": train.get("states_sha256"),
            "validation_state_sha256": validation.get("states_sha256"),
        }

    for seed in SEEDS:
        _require_empty(root / "writebacks" / f"seed{seed}" / "train")
        _require_empty(root / "writebacks" / f"seed{seed}" / "val")
    _require_empty(root / "train_calibration")
    for artifact in (
        "three_seed_nonkeep_factorial_summary.json",
        "three_seed_nonkeep_factorial_with_forced_summary.json",
        "v5_matched_intake_audit.json",
    ):
        if (root / artifact).exists():
            raise ValueError(f"V5 writeback resume found result artifact: {artifact}")

    return {
        "schema_version": "sn7-v5-matched-writeback-resume-preflight-v1",
        "run_root": str(root.resolve()),
        "authorization_sha256": authorization_hash,
        "seeds": seed_receipts,
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite V5 writeback resume receipt: {args.output}")
    result = validate_writeback_resume(args.run_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
