#!/usr/bin/env python3
"""Calibrate sparse heuristic controls on train states and evaluate them on val.

The runner is intentionally validation-only. It evaluates downstream evidence
selection under the same frozen candidate interpretation catalog and terminal
environment as the existing closed-loop controls; it does not claim raw-image
compute savings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from activemap.agent.environment import rollout_agent_policy
from activemap.policy.rate_matched import (
    RATE_MATCHED_MODES,
    RateMatchedMode,
    RateMatchedSingleAcquirePolicy,
    achieved_rate,
    fit_rate_calibration,
)
from scripts.evaluate_active_catalog_closed_loop import (
    grouped_bootstrap,
    load_initial_samples,
    metrics,
    trajectory_row,
)
from scripts.evaluate_active_catalog_closed_loop_baselines import (
    authorize_split,
    make_environment,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _support_sha256(samples: list[Any]) -> str:
    identities = [
        (
            str(sample.metadata["source_episode"]),
            float(sample.metadata["budget"]),
            str(sample.metadata["aoi_id"]),
        )
        for sample in samples
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("state support has duplicate episode-budget identities")
    payload = "\n".join(
        f"{episode}\t{budget:.8f}\t{aoi}" for episode, budget, aoi in sorted(identities)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_mode(value: str) -> RateMatchedMode:
    if value not in RATE_MATCHED_MODES:
        raise argparse.ArgumentTypeError(
            f"mode must be one of: {', '.join(RATE_MATCHED_MODES)}"
        )
    return value  # type: ignore[return-value]


def _load_validation_manifest(path: Path) -> tuple[set[tuple[str, float]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split") != "val" or payload.get("test_assets_read") is not False:
        raise ValueError("validation manifest has an invalid split contract")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("validation manifest has no records")
    keys = {
        (str(row["source_episode"]), float(row["budget"]))
        for row in records
        if isinstance(row, dict)
    }
    if len(keys) != len(records):
        raise ValueError("validation manifest has duplicate or invalid records")
    return keys, payload


def _load_train_rate_receipt(path: Path) -> tuple[float, dict[str, Any]]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema_version") != "activemap-train-call-rate-receipt-v1"
        or receipt.get("split") != "train"
        or receipt.get("test_assets_read") is not False
    ):
        raise ValueError("target-rate receipt has an invalid training-only contract")
    source = receipt.get("trace")
    if not isinstance(source, dict) or not source.get("sha256"):
        raise ValueError("target-rate receipt lacks a hashed training trace")
    if not isinstance(receipt.get("support_sha256"), str):
        raise ValueError("target-rate receipt lacks a hashed training support")
    rate = float(receipt.get("target_call_rate", -1.0))
    if not 0.0 <= rate <= 1.0:
        raise ValueError("target-rate receipt has an invalid one-step call rate")
    return rate, receipt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_states", type=Path)
    parser.add_argument("val_states", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--target-rate-receipt",
        required=True,
        type=Path,
        help="Immutable receipt derived only from a fixed training controller trace.",
    )
    parser.add_argument(
        "--target-train-trace",
        required=True,
        type=Path,
        help="Copied training trace whose SHA-256 must match --target-rate-receipt.",
    )
    parser.add_argument(
        "--val-manifest",
        type=Path,
        help="Frozen validation support manifest, e.g. the EDIT-only support.",
    )
    parser.add_argument("--mode", type=_parse_mode, action="append", default=[])
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260812)
    parser.add_argument("--call-rate-tolerance", type=float, default=0.0025)
    parser.add_argument("--max-candidates", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if authorize_split("val", False):  # pragma: no cover - explicit provenance guard
        raise RuntimeError("validation runner unexpectedly authorized test access")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if not 0.0 <= args.call_rate_tolerance <= 1.0:
        raise ValueError("--call-rate-tolerance must be between zero and one")
    if args.bootstrap_repetitions <= 0 or args.max_candidates <= 0:
        raise ValueError("bootstrap repetitions and max candidates must be positive")

    target_rate, rate_receipt = _load_train_rate_receipt(args.target_rate_receipt)
    if not args.target_train_trace.is_file():
        raise FileNotFoundError(f"target training trace is missing: {args.target_train_trace}")
    if _sha256(args.target_train_trace) != rate_receipt["trace"]["sha256"]:
        raise ValueError("target training trace does not match the target-rate receipt")
    modes = tuple(dict.fromkeys(args.mode or RATE_MATCHED_MODES))
    train = load_initial_samples(args.train_states, split="train")
    validation = load_initial_samples(args.val_states, split="val")
    if _support_sha256(train) != rate_receipt["support_sha256"]:
        raise ValueError("training states do not exactly match the target-rate receipt support")
    manifest_receipt = None
    if args.val_manifest is not None:
        allowed, manifest = _load_validation_manifest(args.val_manifest)
        validation = [
            sample
            for sample in validation
            if (str(sample.metadata["source_episode"]), float(sample.metadata["budget"])) in allowed
        ]
        if len(validation) != len(allowed):
            raise ValueError("validation states do not exactly cover the manifest support")
        manifest_receipt = {
            "path": str(args.val_manifest.resolve()),
            "sha256": _sha256(args.val_manifest),
            "record_count": len(allowed),
            "selection_contract": manifest["selection_contract"],
        }

    args.output_dir.mkdir(parents=True)
    results: dict[str, Any] = {}
    for mode in modes:
        calibration = fit_rate_calibration(train, mode, target_rate)
        train_rate = achieved_rate(train, calibration)
        rows: list[dict[str, Any]] = []
        for sample in validation:
            environment = make_environment(sample, args.max_candidates)
            policy = RateMatchedSingleAcquirePolicy(sample, calibration)
            trajectory = rollout_agent_policy(
                environment,
                policy,
                max_acquisitions=1,
                max_tool_calls=0,
                max_steps=2,
            )
            rows.append(
                trajectory_row(
                    sample,
                    environment,
                    trajectory,
                    policy_name=f"rate_matched_{mode}",
                )
            )
        trace = args.output_dir / f"rate_matched_{mode}.jsonl"
        trace.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
        observed = metrics(rows)
        validation_rate = float(observed["mean_acquisitions"])
        results[mode] = {
            "calibration": calibration.as_dict(),
            "train_call_rate": train_rate,
            "validation_call_rate": validation_rate,
            "target_call_rate": target_rate,
            "absolute_validation_rate_error": abs(validation_rate - target_rate),
            "within_call_rate_tolerance": abs(validation_rate - target_rate)
            <= args.call_rate_tolerance,
            "metrics": observed,
            "aoi_bootstrap": grouped_bootstrap(
                rows, args.bootstrap_repetitions, args.bootstrap_seed
            ),
            "trace": str(trace.resolve()),
            "trace_sha256": _sha256(trace),
        }
    summary = {
        "schema_version": "activemap-rate-matched-heuristics-v1",
        "split": "val",
        "test_assets_read": False,
        "target_call_rate": target_rate,
        "target_source": {
            "type": "predeclared_train_derived_rate",
            "receipt_path": str(args.target_rate_receipt.resolve()),
            "receipt_sha256": _sha256(args.target_rate_receipt),
            "trace_sha256": rate_receipt["trace"]["sha256"],
            "trace_path": str(args.target_train_trace.resolve()),
            "support_sha256": rate_receipt["support_sha256"],
        },
        "protocol": {
            "shared_frozen_candidate_frontend": True,
            "downstream_operational_cost_only": True,
            "max_acquisitions": 1,
            "calibration_split": "train",
            "evaluation_split": "val",
            "call_rate_tolerance": args.call_rate_tolerance,
            "bootstrap_group": "aoi_id",
            "bootstrap_repetitions": args.bootstrap_repetitions,
        },
        "inputs": {
            "train_states": {
                "path": str(args.train_states.resolve()),
                "sha256": _sha256(args.train_states),
            },
            "val_states": {
                "path": str(args.val_states.resolve()),
                "sha256": _sha256(args.val_states),
            },
            "validation_manifest": manifest_receipt,
        },
        "policies": results,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
