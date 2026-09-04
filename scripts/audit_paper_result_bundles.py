#!/usr/bin/env python3
"""Audit final paper-result bundles against every frozen registry cell."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

BUDGETED_FAMILIES = {"selector", "agent", "rl", "writeback"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cell_id(experiment: str, variant: str | None, budget: float | None) -> str:
    variant_part = f"/{variant}" if variant else ""
    budget_part = f"@{budget:g}" if budget is not None else ""
    return f"{experiment}{variant_part}{budget_part}"


def _expected_cells(registry: dict[str, Any]) -> list[dict[str, Any]]:
    budgets = list(map(float, registry["protocol"]["muno21_budgets"]))
    cells: list[dict[str, Any]] = []
    for experiment in registry["experiments"]:
        family = str(experiment["family"])
        metric_family = "agent" if family == "rl" else family
        required_metrics = list(
            map(
                str,
                experiment.get(
                    "required_metrics", registry["required_metrics"][metric_family]
                ),
            )
        )
        primary_metrics = list(
            map(
                str,
                experiment.get("primary_metrics", registry["primary_metrics"][family]),
            )
        )
        variants = list(map(str, experiment.get("methods", []))) or [None]
        cell_budgets: list[float | None] = budgets if family in BUDGETED_FAMILIES else [None]
        for variant in variants:
            for budget in cell_budgets:
                cells.append(
                    {
                        "id": _cell_id(experiment["id"], variant, budget),
                        "experiment_id": experiment["id"],
                        "variant": variant,
                        "budget": budget,
                        "family": family,
                        "required_metrics": required_metrics,
                        "primary_metrics": primary_metrics,
                        "cell_kind": "budget_point" if budget is not None else "summary",
                        "split": (
                            "test"
                            if experiment["test_policy"] == "frozen_once"
                            else "val"
                        ),
                        "seeds": [
                            str(seed)
                            for seed in experiment.get("seeds", ["deterministic"])
                        ],
                    }
                )
        curve_metrics = list(
            map(
                str,
                experiment.get(
                    "curve_metrics", registry.get("curve_metrics", {}).get(family, [])
                ),
            )
        )
        curve_primary = list(
            map(
                str,
                experiment.get(
                    "curve_primary_metrics",
                    registry.get("curve_primary_metrics", {}).get(family, curve_metrics),
                ),
            )
        )
        if family in BUDGETED_FAMILIES and curve_metrics:
            for variant in variants:
                cells.append(
                    {
                        "id": _cell_id(experiment["id"], variant, None),
                        "experiment_id": experiment["id"],
                        "variant": variant,
                        "budget": None,
                        "family": family,
                        "required_metrics": curve_metrics,
                        "primary_metrics": curve_primary,
                        "cell_kind": "curve_summary",
                        "split": (
                            "test"
                            if experiment["test_policy"] == "frozen_once"
                            else "val"
                        ),
                        "seeds": [
                            str(seed)
                            for seed in experiment.get("seeds", ["deterministic"])
                        ],
                    }
                )
    return cells


def _valid_metric_summary(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    try:
        mean = float(value["mean"])
        std = float(value["std"])
        low, high = map(float, value["ci95"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        all(math.isfinite(item) for item in (mean, std, low, high))
        and std >= 0
        and low <= mean <= high
    )


def audit_result_bundles(
    registry_path: Path,
    bundles_root: Path,
) -> dict[str, Any]:
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    registry_hash = _sha256(registry_path)
    expected = _expected_cells(registry)
    expected_by_id = {cell["id"]: cell for cell in expected}
    errors: list[str] = []
    bundles: dict[str, dict[str, Any]] = {}
    source_paths: dict[str, Path] = {}

    for path in sorted(bundles_root.rglob("*.json")) if bundles_root.is_dir() else []:
        try:
            # Accept legacy JSON artifacts written with a UTF-8 BOM while
            # keeping the emitted audit report and bundle hashes unchanged.
            bundle = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            errors.append(f"{path}: invalid JSON: {exc}")
            continue
        # Result directories also contain tables, manifests, and arrays that
        # are not registry cells. Only object-shaped paper bundles are valid
        # candidates for the schema check below.
        if not isinstance(bundle, dict):
            continue
        if bundle.get("schema_version") != "activemap-paper-result-v1":
            continue
        cell_id = _cell_id(
            str(bundle.get("experiment_id", "")),
            bundle.get("variant"),
            float(bundle["budget"]) if bundle.get("budget") is not None else None,
        )
        if cell_id in bundles:
            errors.append(f"duplicate result cell {cell_id}: {source_paths[cell_id]} and {path}")
            continue
        bundles[cell_id] = bundle
        source_paths[cell_id] = path

    unknown_cells = sorted(set(bundles) - set(expected_by_id))
    errors.extend(f"unknown result cell: {cell}" for cell in unknown_cells)
    complete_cells: list[str] = []
    cell_errors: dict[str, list[str]] = {}
    test_ledgers: set[str] = set()

    bootstrap_unit = str(registry["protocol"]["bootstrap_unit"])
    bootstrap_replicates = int(registry["protocol"]["bootstrap_replicates"])
    confidence_level = float(registry["protocol"]["confidence_level"])
    for cell_id, cell in expected_by_id.items():
        bundle = bundles.get(cell_id)
        if bundle is None:
            continue
        issues: list[str] = []
        if bundle.get("registry_sha256") != registry_hash:
            issues.append("registry_sha256_mismatch")
        if bundle.get("split") != cell["split"]:
            issues.append(f"split_mismatch:{bundle.get('split')}!={cell['split']}")
        actual_seeds = sorted(map(str, bundle.get("seeds", [])))
        if actual_seeds != sorted(cell["seeds"]):
            issues.append(f"seed_mismatch:{actual_seeds}!={sorted(cell['seeds'])}")
        if int(bundle.get("sample_count", 0)) <= 0:
            issues.append("nonpositive_sample_count")
        if int(bundle.get("unit_count", 0)) <= 0:
            issues.append("nonpositive_unit_count")
        if bundle.get("bootstrap_unit") != bootstrap_unit:
            issues.append("bootstrap_unit_mismatch")
        if bundle.get("bootstrap_replicates") != bootstrap_replicates:
            issues.append("bootstrap_replicates_mismatch")
        if bundle.get("confidence_level") != confidence_level:
            issues.append("confidence_level_mismatch")
        sources = bundle.get("source_observations", [])
        if not isinstance(sources, list) or not sources:
            issues.append("missing_source_observations")
        else:
            for source in sources:
                source_path = Path(str(source.get("path", "")))
                if not source_path.is_file():
                    issues.append(f"missing_source_observation:{source_path}")
                elif _sha256(source_path) != source.get("sha256"):
                    issues.append(f"source_observation_sha256_mismatch:{source_path}")
        metrics = bundle.get("metrics", {})
        for metric in cell["required_metrics"]:
            if metric not in metrics:
                issues.append(f"missing_metric:{metric}")
            elif not _valid_metric_summary(metrics[metric]):
                issues.append(f"invalid_metric_summary:{metric}")
        for metric in cell["primary_metrics"]:
            if metric not in metrics or not _valid_metric_summary(metrics[metric]):
                issues.append(f"missing_primary_ci:{metric}")
        if cell["family"] == "writeback":
            metric_unit_counts = bundle.get("metric_unit_counts")
            if not isinstance(metric_unit_counts, dict):
                issues.append("missing_metric_unit_counts")
            else:
                total_units = int(bundle.get("unit_count", 0))
                for metric in cell["required_metrics"]:
                    try:
                        metric_units = int(metric_unit_counts[metric])
                        summary_units = int(metrics[metric]["unit_count"])
                    except (KeyError, TypeError, ValueError):
                        issues.append(f"invalid_metric_unit_count:{metric}")
                        continue
                    if not 2 <= metric_units <= total_units:
                        issues.append(f"invalid_metric_unit_count:{metric}")
                    if summary_units != metric_units:
                        issues.append(f"metric_unit_count_mismatch:{metric}")
        if cell["split"] == "test":
            ledger_path = Path(str(bundle.get("frozen_test_ledger", "")))
            if not ledger_path.is_file():
                issues.append("missing_frozen_test_ledger")
            else:
                ledger_hash = _sha256(ledger_path)
                if ledger_hash != bundle.get("frozen_test_ledger_sha256"):
                    issues.append("frozen_test_ledger_sha256_mismatch")
                try:
                    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    issues.append("invalid_frozen_test_ledger")
                else:
                    if ledger.get("status") != "complete" or ledger.get("returncode") != 0:
                        issues.append("frozen_test_ledger_not_complete")
                test_ledgers.add(str(ledger_path.resolve()))
        if issues:
            cell_errors[cell_id] = issues
        else:
            complete_cells.append(cell_id)

    if len(test_ledgers) > 1:
        errors.append(f"multiple frozen test ledgers referenced: {sorted(test_ledgers)}")
    missing_cells = sorted(set(expected_by_id) - set(bundles))
    return {
        "schema_version": "activemap-paper-result-audit-v1",
        "registry": str(registry_path),
        "registry_sha256": registry_hash,
        "contract_valid": not errors,
        "paper_tables_complete": not errors
        and not missing_cells
        and not cell_errors,
        "expected_cell_count": len(expected),
        "observed_cell_count": len(set(bundles) & set(expected_by_id)),
        "complete_cell_count": len(complete_cells),
        "missing_cells": missing_cells,
        "cell_errors": cell_errors,
        "errors": errors,
        "frozen_test_ledgers": sorted(test_ledgers),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("registry", type=Path)
    parser.add_argument("bundles_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = audit_result_bundles(args.registry, args.bundles_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["contract_valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
