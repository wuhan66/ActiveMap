#!/usr/bin/env python3
"""Fail-closed preflight for resuming the registered SN7 V5 queue.

The queue may resume only before selector training.  It can reuse immutable
train-state and internal-split artifacts but never rebuilds them, changes the
authorization, or opens validation/test results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SEEDS = (20260817, 20260818, 20260819)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required V5 resume artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"V5 resume artifact is not a JSON object: {path}")
    return payload


def _require_false(payload: dict[str, Any], field: str, *, path: Path) -> None:
    if payload.get(field) is not False:
        raise ValueError(f"{path} must record {field}=false")


def _validate_train_state(path: Path) -> int:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"train selector state is missing or empty: {path}")
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "train":
                raise ValueError(f"{path}:{line_number} is not an original train state")
            rows += 1
    if rows == 0:
        raise ValueError(f"train selector state has no rows: {path}")
    return rows


def validate_internal_split(run_root: Path, seed: int) -> dict[str, Any] | None:
    state_root = run_root / "selector_states"
    internal_dir = state_root / f"seed{seed}_internal"
    if not internal_dir.exists():
        return None
    summary_path = internal_dir / "summary.json"
    summary = load_json(summary_path)
    if summary.get("schema_version") != "sn7-v5-selector-train-internal-split-v1":
        raise ValueError(f"unexpected internal split schema: {summary_path}")
    _require_false(summary, "formal_validation_assets_read", path=summary_path)
    _require_false(summary, "test_assets_read", path=summary_path)
    if int(summary.get("split", {}).get("seed", -1)) != seed:
        raise ValueError(f"internal split seed mismatch: {summary_path}")
    if summary.get("split", {}).get("source_episode_overlap") != 0:
        raise ValueError(f"internal split has source-episode leakage: {summary_path}")
    train_states = state_root / f"seed{seed}_train.jsonl"
    source = summary.get("source", {})
    if source.get("original_split") != "train" or source.get("sha256") != sha256(train_states):
        raise ValueError(f"internal split source hash mismatch: {summary_path}")
    outputs = summary.get("outputs", {})
    for name in ("fit", "tune", "fit_tune"):
        output = outputs.get(name)
        artifact = internal_dir / f"{name}.jsonl"
        if not isinstance(output, dict) or output.get("sha256") != sha256(artifact):
            raise ValueError(f"internal split output hash mismatch: {artifact}")
    audit_path = state_root / f"seed{seed}_audit.json"
    audit = load_json(audit_path)
    if audit.get("test_assets_read") is not False:
        raise ValueError(f"internal state audit is not test-isolated: {audit_path}")
    if audit.get("splits", {}).get("train", 0) <= 0 or audit.get("splits", {}).get("val", 0) <= 0:
        raise ValueError(f"internal state audit lacks train/tune support: {audit_path}")
    return {"summary_sha256": sha256(summary_path), "state_audit_sha256": sha256(audit_path)}


def validate_resume(run_root: Path) -> dict[str, Any]:
    status_path = run_root / "queue_status.json"
    status = load_json(status_path)
    _require_false(status, "test_assets_read", path=status_path)
    if status.get("status") not in {"starting", "resuming"}:
        raise ValueError(f"V5 queue is not resumable from status={status.get('status')!r}")

    authorization_path = run_root / "authorization" / "three_seed_headroom_authorization.json"
    authorization = load_json(authorization_path)
    _require_false(authorization, "test_assets_read", path=authorization_path)
    records = authorization.get("records")
    if not isinstance(records, list) or {int(item.get("seed", -1)) for item in records} != set(SEEDS):
        raise ValueError("V5 authorization must contain exactly the registered three seeds")
    for record in records:
        if record.get("passed") is not True or record.get("test_assets_read") is not False:
            raise ValueError("V5 authorization contains a non-passing or non-isolated seed")

    state_root = run_root / "selector_states"
    state_rows = {}
    internal = {}
    for seed in SEEDS:
        state_rows[str(seed)] = _validate_train_state(state_root / f"seed{seed}_train.jsonl")
        value = validate_internal_split(run_root, seed)
        if value is not None:
            internal[str(seed)] = value

    # This resume implementation starts from selector training. A model
    # artifact would indicate an ambiguous partial later phase and must be
    # investigated rather than silently reused.
    for seed in SEEDS:
        if (run_root / "selectors" / f"seed{seed}" / "best.pt").exists():
            raise ValueError("selector checkpoint exists; V5 resume phase is ambiguous")
    for forbidden in ("rollouts", "writebacks", "train_calibration"):
        path = run_root / forbidden
        if path.exists() and any(path.iterdir()):
            raise ValueError(f"V5 resume found later-phase artifact: {path}")
    for forbidden in (
        "three_seed_nonkeep_factorial_summary.json",
        "three_seed_nonkeep_factorial_with_forced_summary.json",
        "v5_matched_intake_audit.json",
    ):
        if (run_root / forbidden).exists():
            raise ValueError(f"V5 resume found completed result artifact: {forbidden}")

    return {
        "schema_version": "sn7-v5-matched-resume-preflight-v1",
        "run_root": str(run_root.resolve()),
        "authorization_sha256": sha256(authorization_path),
        "state_rows": state_rows,
        "validated_existing_internal_splits": internal,
        "missing_internal_split_seeds": [seed for seed in SEEDS if str(seed) not in internal],
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite V5 resume receipt: {args.output}")
    result = validate_resume(args.run_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
