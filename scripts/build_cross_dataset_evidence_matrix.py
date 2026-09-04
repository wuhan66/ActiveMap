#!/usr/bin/env python3
"""Build a conservative, non-pooled paper evidence matrix across datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def _load(path: Path | None) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    if path is None:
        return None, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload, {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def build_matrix(
    protocol_path: Path,
    *,
    sn7_promotion_path: Path | None = None,
    muno21_report_path: Path | None = None,
) -> dict[str, Any]:
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "activemap-cross-dataset-protocol-v1":
        raise ValueError("unexpected cross-dataset protocol schema")
    if protocol.get("aggregation", {}).get("pool_samples_across_datasets") is not False:
        raise ValueError("cross-dataset sample pooling must be forbidden")

    sn7, sn7_source = _load(sn7_promotion_path)
    muno21, muno21_source = _load(muno21_report_path)
    sn7_expected = protocol["evidence"]["sn7"]
    muno_expected = protocol["evidence"]["muno21"]

    if sn7 is not None and sn7.get("schema_version") != sn7_expected["expected_schema"]:
        raise ValueError("unexpected SN7 evidence schema")
    if muno21 is not None:
        if muno21.get("schema_version") != muno_expected["expected_schema"]:
            raise ValueError("unexpected MUNO21 evidence schema")
        if muno21.get("comparison_id") != muno_expected["comparison_id"]:
            raise ValueError("MUNO21 evidence is not the predeclared strongest-baseline comparison")

    sn7_seed_ok = False
    if sn7 is not None:
        checks = sn7.get("checks", {})
        sn7_seed_ok = bool(checks.get("required_seed_count")) and int(
            sn7.get("minimum_seed_count", 0)
        ) >= int(sn7_expected["minimum_seeds"])
    sn7_passed = bool(sn7 and sn7.get("promote") and sn7_seed_ok)
    sn7_frozen = bool(
        sn7
        and sn7.get("test_assets_read") is True
        and sn7.get("formalized_from_completed_ledger") is True
        and sn7.get("frozen_test_ledger_sha256")
    )

    muno_seed_count = 0
    if muno21 is not None:
        baseline_seeds = set(map(str, muno21.get("baseline", {}).get("seeds", [])))
        candidate_seeds = set(map(str, muno21.get("candidate", {}).get("seeds", [])))
        if baseline_seeds == candidate_seeds:
            muno_seed_count = len(baseline_seeds)
    muno_passed = bool(
        muno21
        and muno21.get("all_required_gates_passed")
        and muno21.get("split") == "test"
        and muno21.get("test_assets_read") is True
        and muno_seed_count >= int(muno_expected["minimum_seeds"])
    )

    dataset_evidence = {
        "sn7": {
            "role": protocol["datasets"]["sn7"]["role"],
            "status": "passed" if sn7_passed else ("failed" if sn7 else "not_run"),
            "evidence_level": "frozen_test" if sn7_frozen else "validation_only",
            "independent_gate_passed": sn7_passed,
            "source": sn7_source,
        },
        "muno21": {
            "role": protocol["datasets"]["muno21"]["role"],
            "status": "passed" if muno_passed else ("failed" if muno21 else "not_run"),
            "evidence_level": "frozen_test" if muno21 and muno21.get("test_assets_read") else "not_run",
            "independent_gate_passed": muno_passed,
            "source": muno21_source,
        },
        "inria": {
            "role": protocol["datasets"]["inria"]["role"],
            "status": "auxiliary_only",
            "evidence_level": "pretraining_transfer_ablation",
            "independent_gate_passed": None,
            "source": None,
        },
    }
    cross_geometry = sn7_passed and sn7_frozen and muno_passed
    return {
        "schema_version": "activemap-cross-dataset-evidence-matrix-v1",
        "dataset_evidence": dataset_evidence,
        "gates": {
            "sn7_independent_gain": sn7_passed,
            "sn7_frozen_test": sn7_frozen,
            "muno21_independent_gain": muno_passed,
            "no_pooled_primary_metric": True,
            "cross_geometry_generalization_supported": cross_geometry,
        },
        "claim_status": (
            "cross_geometry_frozen_test_supported"
            if cross_geometry
            else "insufficient_for_cross_geometry_claim"
        ),
        "pooled_score": None,
        "claim_boundary": (
            "Cross-geometry Agent generalization requires independent frozen-test gains on "
            "SN7 and MUNO21. Inria never counts as an Agent benchmark."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sn7-promotion", type=Path)
    parser.add_argument("--muno21-report", type=Path)
    args = parser.parse_args()
    report = build_matrix(
        args.protocol,
        sn7_promotion_path=args.sn7_promotion,
        muno21_report_path=args.muno21_report,
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
