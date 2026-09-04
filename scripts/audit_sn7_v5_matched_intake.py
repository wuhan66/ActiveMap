#!/usr/bin/env python3
"""Fail-closed intake audit for the registered V5 matched writeback branch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SEEDS = (20260817, 20260818, 20260819)
FACTORIAL_POLICIES = (
    "direct_commit",
    "direct_safe_commit",
    "selected_commit",
    "selected_safe_commit",
)
FORCED_POLICY = "forced_safe_commit"
RAW_POLICIES = ("direct_commit", "selected_commit", "forced_commit")
SAFE_POLICIES = ("direct_safe_commit", "selected_safe_commit", FORCED_POLICY)
REQUIRED_POLICIES = FACTORIAL_POLICIES + (FORCED_POLICY,)
OPERATIONS = ("ADD", "DELETE", "RESHAPE")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def assert_false(payload: dict[str, Any], key: str, source: Path) -> None:
    require(payload.get(key) is False, f"{source} must record {key}=false")


def audit_authorization(root: Path) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    path = root / "authorization" / "three_seed_headroom_authorization.json"
    payload = read_json(path)
    require(
        payload.get("schema_version") == "sn7-v5b-three-seed-headroom-authorization-v1",
        "unexpected V5 authorization schema",
    )
    require(payload.get("authorization") == "matched_nonkeep_factorial_writeback", "V5 is not authorized")
    assert_false(payload, "test_assets_read", path)
    records = payload.get("records")
    require(isinstance(records, list), "V5 authorization lacks records")
    by_seed: dict[int, dict[str, Any]] = {}
    for record in records:
        require(isinstance(record, dict), "V5 authorization record is not an object")
        seed = int(record.get("seed", -1))
        require(seed in SEEDS and seed not in by_seed, "invalid or duplicate V5 authorization seed")
        require(record.get("passed") is True, f"V5 authorization did not pass for seed {seed}")
        assert_false(record, "test_assets_read", path)
        state_hash = record.get("state_file_sha256")
        require(isinstance(state_hash, str) and len(state_hash) == 64, f"seed {seed} lacks validation state hash")
        by_seed[seed] = record
    require(tuple(sorted(by_seed)) == SEEDS, "V5 authorization lacks a registered seed")
    return by_seed, {"path": str(path.resolve()), "sha256": sha256(path)}


def read_writeback_keys(path: Path) -> tuple[set[tuple[str, float]], dict[tuple[str, float], tuple[str, str]], set[str]]:
    keys: set[tuple[str, float]] = set()
    metadata: dict[tuple[str, float], tuple[str, str]] = {}
    operations: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            require(isinstance(row, dict), f"{path}:{line_number} is not an object")
            require(row.get("split") == "val", f"{path}:{line_number} is not validation data")
            require(row.get("test_assets_read") is False, f"{path}:{line_number} is not test-isolated")
            task_id, budget = str(row.get("task_id", "")), float(row.get("budget", 0.0))
            require(task_id and budget > 0.0, f"{path}:{line_number} lacks task identity")
            key = (task_id, budget)
            require(key not in keys, f"{path}:{line_number} duplicates a task-budget row")
            target = str(row.get("target", ""))
            require(target.startswith("COMMIT:") or target == "REJECT", f"{path}:{line_number} has invalid target")
            keys.add(key)
            metadata[key] = (str(row.get("aoi_id", "")), target)
            if target.startswith("COMMIT:"):
                operations.add(target.removeprefix("COMMIT:"))
    require(keys, f"{path} contains no writeback rows")
    return keys, metadata, operations


def audit_seed(root: Path, seed: int, authorization: dict[str, Any]) -> dict[str, Any]:
    state_root = root / "selector_states"
    train_states = state_root / f"seed{seed}_train.jsonl"
    internal_summary_path = state_root / f"seed{seed}_internal" / "summary.json"
    selector_root = root / "selectors" / f"seed{seed}"
    selector = selector_root / "best.pt"
    selector_metrics_path = selector_root / "metrics.json"
    require(train_states.is_file() and train_states.stat().st_size > 0, f"missing train states for {seed}")
    internal = read_json(internal_summary_path)
    require(internal.get("schema_version") == "sn7-v5-selector-train-internal-split-v1", "bad internal split schema")
    assert_false(internal, "formal_validation_assets_read", internal_summary_path)
    assert_false(internal, "test_assets_read", internal_summary_path)
    source = internal.get("source", {})
    require(isinstance(source, dict) and source.get("original_split") == "train", "internal split source is not train")
    require(source.get("sha256") == sha256(train_states), "internal split is not bound to train states")
    split = internal.get("split", {})
    require(isinstance(split, dict) and split.get("source_episode_overlap") == 0, "internal sources overlap")
    outputs = internal.get("outputs", {})
    require(isinstance(outputs, dict), "internal split lacks outputs")
    for name in ("fit", "tune", "fit_tune"):
        item = outputs.get(name, {})
        require(isinstance(item, dict), f"internal split lacks {name}")
        path = Path(str(item.get("path", "")))
        require(path.is_file() and item.get("sha256") == sha256(path), f"internal {name} hash mismatch")

    require(selector.is_file(), f"missing selector checkpoint for {seed}")
    selector_metrics = read_json(selector_metrics_path)
    require(selector_metrics.get("calibrate_stop_margin") is True, f"selector {seed} lacks train-only STOP calibration")

    rollout_root = root / "rollouts"
    rollout_receipts: dict[str, dict[str, Any]] = {}
    for split_name in ("train", "val"):
        receipt_path = rollout_root / f"seed{seed}_{split_name}" / "receipt.json"
        receipt = read_json(receipt_path)
        require(receipt.get("split") == split_name, f"rollout receipt split mismatch for {seed}")
        assert_false(receipt, "test_assets_read", receipt_path)
        require(receipt.get("authorization_sha256") == sha256(root / "authorization" / "three_seed_headroom_authorization.json"), "rollout authorization hash mismatch")
        require(receipt.get("selector_checkpoint_sha256") == sha256(selector), "rollout selector hash mismatch")
        for policy in ("direct", "selected", "forced"):
            path_key, hash_key = f"{policy}_rollouts", f"{policy}_rollouts_sha256"
            rollout_path = Path(str(receipt.get(path_key, "")))
            require(rollout_path.is_file() and receipt.get(hash_key) == sha256(rollout_path), f"{split_name} {policy} rollout hash mismatch")
        rollout_receipts[split_name] = receipt
    require(rollout_receipts["train"].get("states_sha256") == sha256(train_states), "train rollout state hash mismatch")
    require(rollout_receipts["val"].get("states_sha256") == authorization.get("state_file_sha256"), "validation rollout state is not authorized")

    calibration_path = root / "train_calibration" / f"seed{seed}.json"
    calibration = read_json(calibration_path)
    require(calibration.get("schema_version") == "sn7-v5-safe-commit-calibration-v1", "bad Safe Commit schema")
    require(calibration.get("split") == "train", "Safe Commit calibration is not train-only")
    assert_false(calibration, "test_assets_read", calibration_path)
    calibration_inputs = calibration.get("inputs", {})
    require(isinstance(calibration_inputs, dict) and set(calibration_inputs) == {"direct", "selected"}, "bad Safe Commit inputs")
    for policy in ("direct", "selected"):
        raw_path = root / "writebacks" / f"seed{seed}" / "train" / f"{policy}_commit" / "writeback.jsonl"
        require(calibration_inputs[policy].get("sha256") == sha256(raw_path), f"Safe Commit {policy} input hash mismatch")

    writeback_root = root / "writebacks" / f"seed{seed}" / "val"
    policy_paths = {policy: writeback_root / policy / "writeback.jsonl" for policy in REQUIRED_POLICIES}
    raw_paths = {policy: writeback_root / policy / "writeback.jsonl" for policy in RAW_POLICIES}
    for path in [*policy_paths.values(), *raw_paths.values()]:
        require(path.is_file(), f"missing V5 writeback: {path}")
    expected_keys: set[tuple[str, float]] | None = None
    expected_metadata: dict[tuple[str, float], tuple[str, str]] | None = None
    expected_operations: set[str] | None = None
    for policy, path in policy_paths.items():
        keys, metadata, operations = read_writeback_keys(path)
        if expected_keys is None:
            expected_keys, expected_metadata, expected_operations = keys, metadata, operations
        else:
            require(keys == expected_keys and metadata == expected_metadata, f"{policy} lacks paired validation support")
            require(operations == expected_operations, f"{policy} has mismatched operation support")
    require(expected_operations is not None and set(OPERATIONS).issubset(expected_operations), f"seed {seed} lacks an editable operation")
    return {
        "train_states_sha256": sha256(train_states),
        "selector_checkpoint_sha256": sha256(selector),
        "selector_stop_margin_calibrated": True,
        "safe_commit_calibration_sha256": sha256(calibration_path),
        "validation_row_count": len(expected_keys or ()),
        "validation_operations": sorted(expected_operations or ()),
        "writeback_sha256": {policy: sha256(path) for policy, path in policy_paths.items()},
        "raw_writeback_sha256": {policy: sha256(path) for policy, path in raw_paths.items()},
    }


def audit(root: Path) -> dict[str, Any]:
    authorization, authorization_receipt = audit_authorization(root)
    summary_path = root / "three_seed_nonkeep_factorial_with_forced_summary.json"
    summary = read_json(summary_path)
    require(summary.get("schema_version") == "sn7-v5-matched-nonkeep-factorial-v2", "bad V5 aggregate schema")
    require(summary.get("split") == "val", "V5 aggregate is not validation-only")
    assert_false(summary, "test_assets_read", summary_path)
    require(tuple(summary.get("model_seeds", ())) == SEEDS, "aggregate lacks registered seeds")
    require(tuple(summary.get("policies", ())) == REQUIRED_POLICIES, "aggregate lacks forced cost control")
    seed_audits = {str(seed): audit_seed(root, seed, authorization[seed]) for seed in SEEDS}
    operation_slices = summary.get("operation_slices", {})
    require(isinstance(operation_slices, dict) and set(operation_slices) == set(OPERATIONS), "aggregate lacks operation slices")
    for operation in OPERATIONS:
        for factor in ("selection_factor", "safe_commit_factor"):
            values = operation_slices[operation].get(factor, {})
            counts = values.get("row_count_per_seed", {}) if isinstance(values, dict) else {}
            require(all(int(counts.get(str(seed), counts.get(seed, 0))) > 0 for seed in SEEDS), f"{operation} has empty {factor} support")
    promotion = summary.get("promotion", {})
    require(isinstance(promotion, dict) and isinstance(promotion.get("forced_acquisition_cost_control"), dict), "forced cost control is not aggregated")
    return {
        "schema_version": "sn7-v5-matched-intake-audit-v1",
        "passed": True,
        "split": "val",
        "test_assets_read": False,
        "run_root": str(root.resolve()),
        "authorization": authorization_receipt,
        "aggregate": {"path": str(summary_path.resolve()), "sha256": sha256(summary_path)},
        "seeds": seed_audits,
        "promotion": promotion,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite intake audit: {args.output}")
    result = audit(args.run_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
