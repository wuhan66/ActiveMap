#!/usr/bin/env python3
"""Finalize the test-free SN7 ChangeMamba validation evidence chain."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def jsonl_count(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"invalid JSONL object at {path}:{line_number}")
            count += 1
    if count == 0:
        raise ValueError(f"empty JSONL artifact: {path}")
    return count


def require_test_free(value: dict[str, Any], path: Path) -> None:
    flag = value.get("test_assets_read")
    if flag is None and isinstance(value.get("protocol"), dict):
        flag = value["protocol"].get("test_assets_read")
    if flag is not False:
        raise ValueError(f"artifact is not explicitly test-free: {path}")


def finalize(
    run_root: Path,
    bundle_root: Path,
    *,
    seeds: list[int],
) -> dict[str, Any]:
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("finalization requires distinct replicated seeds")
    artifacts: list[Path] = []
    validation_count = None
    train_count = None
    checkpoint_hashes = set()
    for seed in seeds:
        run = run_root / f"full_weight5_seed{seed}_v1"
        training_path = run / "summary.json"
        training = read_json(training_path)
        require_test_free(training, training_path)
        if int(training["seed"]) != seed or int(training["best_epoch"]) <= 0:
            raise ValueError(f"invalid training identity: {training_path}")

        train_audit_path = run / "train_audit" / "summary.json"
        val_audit_path = run / "val_audit" / "summary.json"
        safe_path = run / "safe_commit" / "summary.json"
        train_audit = read_json(train_audit_path)
        val_audit = read_json(val_audit_path)
        safe = read_json(safe_path)
        for value, path in (
            (train_audit, train_audit_path),
            (val_audit, val_audit_path),
            (safe, safe_path),
        ):
            require_test_free(value, path)
        if train_audit["split"] != "train" or val_audit["split"] != "val":
            raise ValueError(f"train/validation split mismatch: {run}")
        if train_audit.get("masks_saved") is not False:
            raise ValueError(f"train calibration masks should not be saved: {run}")
        if val_audit.get("masks_saved") is not True:
            raise ValueError(f"validation masks must be available: {run}")

        current_train_count = int(train_audit["sample_count"])
        current_validation_count = int(val_audit["sample_count"])
        if train_count is None:
            train_count = current_train_count
            validation_count = current_validation_count
        elif (
            current_train_count != train_count
            or current_validation_count != validation_count
        ):
            raise ValueError("sample counts differ across seeds")
        if safe["validation"]["sample_count"] != current_validation_count:
            raise ValueError(f"safe-commit validation count mismatch: {run}")
        if safe["calibration"]["train_sample_count"] != current_train_count:
            raise ValueError(f"safe-commit train count mismatch: {run}")

        checkpoint_hashes.add(str(val_audit["checkpoint_sha256"]))
        for directory, expected_count in (
            (run / "train_audit", current_train_count),
            (run / "val_audit", current_validation_count),
            (run / "safe_commit", current_validation_count),
        ):
            per_sample = directory / "per_sample.jsonl"
            if jsonl_count(per_sample) != expected_count:
                raise ValueError(f"per-sample count mismatch: {per_sample}")
            artifacts.append(per_sample)
        for predictions in (
            run / "val_audit" / "predictions.jsonl",
            run / "safe_commit" / "predictions.jsonl",
        ):
            if jsonl_count(predictions) != current_validation_count:
                raise ValueError(f"prediction count mismatch: {predictions}")
            artifacts.append(predictions)
        artifacts.extend(
            (training_path, train_audit_path, val_audit_path, safe_path)
        )
    if len(checkpoint_hashes) != len(seeds):
        raise ValueError("best checkpoint hashes must be distinct across seeds")

    aggregate_paths = (
        run_root / "full_weight5_three_seed_20260726.json",
        run_root / "full_weight5_three_seed_aoi_bootstrap_20260726.json",
        run_root / "full_weight5_three_seed_safe_commit_20260726.json",
        run_root / "full_weight5_three_seed_safe_commit_aoi_bootstrap_20260726.json",
    )
    for path in aggregate_paths:
        value = read_json(path)
        require_test_free(value, path)
        if int(value["run_count"]) != len(seeds):
            raise ValueError(f"aggregate run count mismatch: {path}")
        artifacts.append(path)

    bundle_specs = (
        (
            bundle_root / "changemamba_always_commit_val.json",
            "always_commit",
        ),
        (
            bundle_root / "changemamba_safe_commit_val.json",
            "safe_commit",
        ),
    )
    for path, variant in bundle_specs:
        bundle = read_json(path)
        if (
            bundle.get("experiment_id") != "sn7_changemamba_commit_policy"
            or bundle.get("variant") != variant
            or bundle.get("split") != "val"
            or list(map(int, bundle.get("seeds", []))) != sorted(seeds)
        ):
            raise ValueError(f"paper bundle protocol mismatch: {path}")
        if int(bundle["unit_count"]) < 2:
            raise ValueError(f"paper bundle lacks AOI support: {path}")
        artifacts.append(path)

    return {
        "schema_version": "sn7-changemamba-validation-finalization-v1",
        "status": "complete",
        "seeds": seeds,
        "train_sample_count_per_seed": train_count,
        "validation_sample_count_per_seed": validation_count,
        "checkpoint_sha256": sorted(checkpoint_hashes),
        "test_assets_read": False,
        "artifact_count": len(artifacts),
        "artifacts": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in artifacts
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("bundle_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, action="append", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = finalize(args.run_root, args.bundle_root, seeds=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
