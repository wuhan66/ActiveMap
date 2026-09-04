#!/usr/bin/env python3
"""Finalize V6 forced-control Safe Commit and its five-cell aggregation.

The V6 queue calibrates Safe Commit only on paired ``direct`` and ``selected``
training writebacks. This post-queue finalizer applies that already frozen gate
to the held-out ``forced`` cost-control arm, then aggregates the complete
three-seed factorial comparison. It never recalibrates a threshold or accesses
test assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

# The detached watcher does not inherit the queue's working directory or
# PYTHONPATH. Resolve both local package roots before importing project code.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    import_root_text = str(import_root)
    if import_root_text not in sys.path:
        sys.path.insert(0, import_root_text)

from scripts.aggregate_sn7_v5_factorial_writebacks import aggregate  # noqa: E402
from scripts.calibrate_sn7_v5_safe_commit import (  # noqa: E402
    apply_safe_commit,
    load_rows,
    sha256,
    summarize,
)

SEEDS = (20260817, 20260818, 20260819)
PROTOCOL_SCHEMA = "sn7-v6-evidence-value-writeback-v1"
CALIBRATION_SCHEMA = "sn7-v5-safe-commit-calibration-v1"


def load_protocol(root: Path) -> dict[str, Any]:
    path = root / "protocol.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != PROTOCOL_SCHEMA:
        raise ValueError("unexpected V6 writeback protocol schema")
    if payload.get("test_assets_read") is not False:
        raise ValueError("V6 writeback protocol must forbid test access")
    records = payload.get("seeds")
    if not isinstance(records, list) or {int(record["seed"]) for record in records} != set(SEEDS):
        raise ValueError("V6 writeback protocol lacks the registered seeds")
    for record in records:
        checkpoint = Path(str(record.get("checkpoint", "")))
        if not checkpoint.is_file():
            raise FileNotFoundError(f"registered checkpoint is missing: {checkpoint}")
        actual_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if actual_sha256 != record.get("checkpoint_sha256"):
            raise ValueError(f"registered checkpoint changed: {checkpoint}")
    return payload


def load_gate(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != CALIBRATION_SCHEMA
        or payload.get("split") != "train"
        or payload.get("test_assets_read") is not False
    ):
        raise ValueError(f"{path}: expected a train-only V5-compatible calibration")
    gate = payload.get("gate")
    if not isinstance(gate, dict):
        raise ValueError(f"{path}: calibration has no gate")
    for name in (
        "confidence_threshold",
        "replay_iou_threshold",
        "require_topology",
    ):
        if name not in gate:
            raise ValueError(f"{path}: calibration gate lacks {name}")
    return gate


def write_forced_safe_commit(
    raw_path: Path,
    calibration_path: Path,
    output_dir: Path,
) -> Path:
    gate = load_gate(calibration_path)
    raw = load_rows(raw_path, expected_split="val")
    safe_rows = [
        apply_safe_commit(
            row,
            confidence_threshold=float(gate["confidence_threshold"]),
            replay_iou_threshold=float(gate["replay_iou_threshold"]),
            require_topology=bool(gate["require_topology"]),
        )
        for _, row in sorted(raw.items())
    ]
    output_dir.mkdir(parents=True)
    output = output_dir / "writeback.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for row in safe_rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    raw_metrics = summarize(list(raw.values()))
    safe_metrics = summarize(safe_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "sn7-v6-evidence-value-forced-safe-commit-v1",
                "policy": "forced_safe_commit",
                "split": "val",
                "raw_writebacks": str(raw_path.resolve()),
                "raw_writebacks_sha256": sha256(raw_path),
                "calibration": str(calibration_path.resolve()),
                "calibration_sha256": sha256(calibration_path),
                "gate": gate,
                "raw_metrics": raw_metrics,
                "safe_commit_metrics": safe_metrics,
                "rejected_writeback_count": sum(
                    not row["safe_commit_accepted"] for row in safe_rows
                ),
                "row_count": len(safe_rows),
                "test_assets_read": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("v6_writeback_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260829)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite V6 finalization: {args.output_dir}")
    if not (args.v6_writeback_root / "queue_status.json").is_file():
        raise FileNotFoundError("V6 queue is not complete")
    status = json.loads(
        (args.v6_writeback_root / "queue_status.json").read_text(encoding="utf-8")
    )
    if status.get("status") != "complete" or status.get("test_assets_read") is not False:
        raise ValueError("V6 queue status is not a completed validation-only run")
    load_protocol(args.v6_writeback_root)
    records: dict[int, dict[str, Path]] = {}
    for seed in SEEDS:
        raw_root = args.v6_writeback_root / "raw_writebacks" / f"seed{seed}" / "val"
        safe_root = args.v6_writeback_root / "safe_writebacks" / f"seed{seed}"
        forced_safe = write_forced_safe_commit(
            raw_root / "forced" / "writeback.jsonl",
            args.v6_writeback_root / "train_calibration" / f"seed{seed}.json",
            args.output_dir / "forced_safe_writebacks" / f"seed{seed}",
        )
        records[seed] = {
            "direct_commit": raw_root / "direct" / "writeback.jsonl",
            "direct_safe_commit": safe_root / "direct" / "writeback.jsonl",
            "selected_commit": raw_root / "selected" / "writeback.jsonl",
            "selected_safe_commit": safe_root / "selected" / "writeback.jsonl",
            "forced_safe_commit": forced_safe,
        }
    aggregate_result = aggregate(
        records,
        repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.bootstrap_seed,
    )
    aggregate_result.update(
        {
            "schema_version": "sn7-v6-evidence-value-matched-factorial-v1",
            "source_factorial_implementation": "sn7-v5-matched-nonkeep-factorial-v2",
            "v6_protocol": str((args.v6_writeback_root / "protocol.json").resolve()),
            "v6_protocol_sha256": sha256(args.v6_writeback_root / "protocol.json"),
            "test_assets_read": False,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "three_seed_factorial_with_forced_summary.json").write_text(
        json.dumps(aggregate_result, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "finalization_receipt.json").write_text(
        json.dumps(
            {
                "schema_version": "sn7-v6-evidence-value-finalization-v1",
                "v6_writeback_root": str(args.v6_writeback_root.resolve()),
                "v6_protocol": str((args.v6_writeback_root / "protocol.json").resolve()),
                "v6_protocol_sha256": sha256(args.v6_writeback_root / "protocol.json"),
                "forced_safe_commit_only_uses_train_fitted_gates": True,
                "test_assets_read": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregate_result["promotion"], indent=2))


if __name__ == "__main__":
    main()
