#!/usr/bin/env python3
"""Audit dataset roles and transfer boundaries for paper claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


EXPECTED = {
    "sn7": ("primary_temporal_editable_map_update", "real_temporal", True),
    "muno21": ("external_temporal_geometry_generalization", "real_temporal", True),
    "inria": ("auxiliary_boundary_pretraining_only", "real_single_timestamp", False),
}


def audit(protocol_path: Path, repository_root: Path) -> dict[str, Any]:
    payload = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {
        "schema": payload.get("schema_version") == "activemap-cross-dataset-protocol-v1",
        "exact_dataset_set": set(payload.get("datasets", {})) == set(EXPECTED),
    }
    datasets = payload.get("datasets", {})
    for name, (role, supervision, agent_evaluation) in EXPECTED.items():
        row = datasets.get(name, {})
        checks[f"{name}_role"] = row.get("role") == role
        checks[f"{name}_supervision"] = row.get("supervision") == supervision
        checks[f"{name}_agent_boundary"] = row.get("agent_evaluation") is agent_evaluation

    aggregation = payload.get("aggregation", {})
    checks.update(
        {
            "sample_pooling_forbidden": aggregation.get("pool_samples_across_datasets") is False,
            "per_dataset_reporting_required": aggregation.get("report_per_dataset_metrics") is True,
            "macro_average_not_primary": aggregation.get("allow_macro_average_as_primary") is False,
            "cross_geometry_requires_both_temporal_datasets": set(
                aggregation.get("cross_geometry_claim_requires", [])
            )
            == {"sn7", "muno21"},
        }
    )

    transfer_rows = payload.get("transfer_ablations", [])
    expected_edges = {("inria", "sn7", "segmentation"), ("sn7", "muno21", "encoder")}
    observed_edges: set[tuple[str, str, str]] = set()
    transfer_audits = []
    for row in transfer_rows:
        config_path = repository_root / str(row.get("config", ""))
        config_exists = config_path.is_file()
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_exists else {}
        declared_scope = str(row.get("init_scope", ""))
        configured_scope = str(config.get("training", {}).get("init_scope", ""))
        observed_edges.add((str(row.get("source")), str(row.get("target")), declared_scope))
        transfer_audits.append(
            {
                "source": row.get("source"),
                "target": row.get("target"),
                "config": str(config_path),
                "config_exists": config_exists,
                "init_scope_matches": config_exists and configured_scope == declared_scope,
                "has_init_checkpoint": bool(config.get("training", {}).get("init_checkpoint")),
            }
        )
    checks["exact_transfer_edges"] = observed_edges == expected_edges
    checks["transfer_configs_valid"] = bool(transfer_audits) and all(
        row["config_exists"] and row["init_scope_matches"] and row["has_init_checkpoint"]
        for row in transfer_audits
    )
    checks["claim_boundaries_declared"] = bool(payload.get("allowed_claims")) and bool(
        payload.get("forbidden_claims")
    )
    return {
        "schema_version": "activemap-cross-dataset-protocol-audit-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "transfer_ablations": transfer_audits,
        "claim_boundary": (
            "SN7 and MUNO21 are independently evaluated temporal map-update datasets; "
            "Inria is boundary pretraining only and no cross-dataset samples are pooled."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    args = parser.parse_args()
    report = audit(args.protocol, args.repository_root.resolve())
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
