#!/usr/bin/env python3
"""Verify immutable data gates before promoting SN7 controller experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def require_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{name}: expected {expected!r}, got {actual!r}")


def verify_token_audit(
    report_path: Path,
    expected_train: int,
    expected_val: int,
    max_length: int,
) -> dict[str, Any]:
    report = load_json(report_path)
    require_equal("schema_version", report.get("schema_version"), "visual-sft-token-audit-v1")
    require_equal("passed", report.get("passed"), True)
    require_equal("test_assets_read", report.get("test_assets_read"), False)
    require_equal("max_length", report.get("max_length"), max_length)
    require_equal("over_max_length_records", report.get("over_max_length_records"), 0)
    require_equal("zero_supervised_label_records", report.get("zero_supervised_label_records"), 0)
    splits = report.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("token audit is missing split summaries")
    require_equal("train.records", splits.get("train", {}).get("records"), expected_train)
    require_equal("val.records", splits.get("val", {}).get("records"), expected_val)
    for split in ("train", "val"):
        summary = splits.get(split, {})
        require_equal(f"{split}.over_max_length_count", summary.get("over_max_length_count"), 0)
        require_equal(f"{split}.zero_supervised_labels", summary.get("zero_supervised_labels"), [])
        actions = summary.get("actions", {})
        if int(actions.get("ACQUIRE", 0)) <= 0 or int(actions.get("STOP", 0)) <= 0:
            raise ValueError(f"{split} must contain both ACQUIRE and STOP supervision")
    return {
        "gate": "token_audit",
        "passed": True,
        "report": str(report_path.resolve()),
        "train_records": expected_train,
        "val_records": expected_val,
        "test_assets_read": False,
    }


def verify_smoke(run_root: Path, seed: int) -> dict[str, Any]:
    manifest = load_json(run_root / "launch_manifest.json")
    require_equal("manifest.test_assets_read", manifest.get("test_assets_read"), False)
    schedule = manifest.get("schedule")
    if not isinstance(schedule, list) or len(schedule) != 1:
        raise ValueError(f"smoke must contain exactly one scheduled seed: {schedule!r}")
    require_equal("manifest.schedule.seed", schedule[0].get("seed"), seed)
    audit = manifest.get("data_audit", {})
    for split in ("train", "val"):
        require_equal(f"{split}.records", audit.get(split, {}).get("records"), 2)
        actions = audit.get(split, {}).get("action_counts", {})
        require_equal(f"{split}.ACQUIRE", actions.get("ACQUIRE"), 1)
        require_equal(f"{split}.STOP", actions.get("STOP"), 1)
        require_equal(f"{split}.invalid", audit.get(split, {}).get("invalid_action_records"), 0)
    seed_root = run_root / f"seed{seed}"
    state = load_json(seed_root / "run_state.json")
    result = load_json(seed_root / "process_result.json")
    require_equal("run_state.status", state.get("status"), "completed")
    require_equal("run_state.returncode", state.get("returncode"), 0)
    require_equal("process_result.returncode", result.get("returncode"), 0)
    adapter = seed_root / "final" / "adapter_config.json"
    if not adapter.is_file():
        raise FileNotFoundError(adapter)
    history = seed_root / "history.jsonl"
    if not history.is_file() or not history.read_text(encoding="utf-8").strip():
        raise ValueError(f"missing non-empty training history: {history}")
    return {
        "gate": "smoke",
        "passed": True,
        "run_root": str(run_root.resolve()),
        "seed": seed,
        "peak_memory_used_mib": result.get("peak_memory_used_mib"),
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="gate", required=True)
    token = subparsers.add_parser("token")
    token.add_argument("report", type=Path)
    token.add_argument("--expected-train", type=int, required=True)
    token.add_argument("--expected-val", type=int, required=True)
    token.add_argument("--max-length", type=int, default=2048)
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("run_root", type=Path)
    smoke.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.gate == "token":
        result = verify_token_audit(
            args.report, args.expected_train, args.expected_val, args.max_length
        )
    else:
        result = verify_smoke(args.run_root, args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
