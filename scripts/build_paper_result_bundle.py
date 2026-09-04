#!/usr/bin/env python3
"""Build one provenance-bound paper result bundle from grouped observations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scripts.audit_paper_result_bundles import _expected_cells


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(paths: list[Path]) -> list[tuple[Path, int, dict[str, Any]]]:
    rows: list[tuple[Path, int, dict[str, Any]]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    rows.append((path, line_number, json.loads(line)))
    if not rows:
        raise ValueError("observation inputs contain no rows")
    return rows


def _select_cell(
    registry: dict[str, Any],
    experiment_id: str,
    variant: str | None,
    budget: float | None,
) -> dict[str, Any]:
    matching = [
        cell
        for cell in _expected_cells(registry)
        if cell["experiment_id"] == experiment_id
        and cell["variant"] == variant
        and cell["budget"] == budget
    ]
    if len(matching) != 1:
        raise ValueError(
            "experiment/variant/budget does not identify exactly one registry cell"
        )
    return matching[0]


def build_result_bundle(
    registry_path: Path,
    observation_paths: list[Path],
    *,
    experiment_id: str,
    variant: str | None,
    budget: float | None,
    frozen_test_ledger: Path | None = None,
) -> dict[str, Any]:
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    cell = _select_cell(registry, experiment_id, variant, budget)
    expected_seeds = set(map(str, cell["seeds"]))
    required_metrics = set(map(str, cell["required_metrics"]))
    required_metrics.update(map(str, cell["primary_metrics"]))

    rows = _read_jsonl(observation_paths)
    seen_observations: set[tuple[str, str]] = set()
    values: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    row_counts_by_seed: dict[str, int] = defaultdict(int)
    for path, line_number, row in rows:
        location = f"{path}:{line_number}"
        if row.get("schema_version") != "activemap-paper-observation-v1":
            raise ValueError(f"{location}: invalid observation schema")
        if row.get("experiment_id") != experiment_id or row.get("variant") != variant:
            raise ValueError(f"{location}: experiment or variant mismatch")
        row_budget = float(row["budget"]) if row.get("budget") is not None else None
        if row_budget != budget or row.get("split") != cell["split"]:
            raise ValueError(f"{location}: budget or split mismatch")
        seed = str(row.get("seed"))
        if seed not in expected_seeds:
            raise ValueError(f"{location}: unexpected seed {seed}")
        observation_id = str(row.get("observation_id", ""))
        unit_id = str(row.get("unit_id", ""))
        if not observation_id or not unit_id:
            raise ValueError(f"{location}: observation_id and unit_id are required")
        unique_key = (seed, observation_id)
        if unique_key in seen_observations:
            raise ValueError(f"{location}: duplicate seed/observation_id {unique_key}")
        seen_observations.add(unique_key)
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"{location}: metrics must be an object")
        missing = sorted(required_metrics - set(metrics))
        if missing:
            raise ValueError(f"{location}: missing metrics {missing}")
        for metric in required_metrics:
            value = float(metrics[metric])
            if not math.isfinite(value):
                raise ValueError(f"{location}: non-finite metric {metric}")
            values[seed][unit_id][metric].append(value)
        row_counts_by_seed[seed] += 1

    observed_seeds = set(values)
    if observed_seeds != expected_seeds:
        raise ValueError(
            f"seed mismatch: observed={sorted(observed_seeds)} expected={sorted(expected_seeds)}"
        )
    unit_sets = {seed: set(seed_values) for seed, seed_values in values.items()}
    reference_units = unit_sets[next(iter(sorted(unit_sets)))]
    if any(units != reference_units for units in unit_sets.values()):
        raise ValueError("bootstrap unit ids must be identical across seeds")

    seed_order = sorted(expected_seeds)
    unit_order = sorted(reference_units)
    seed_unit_means: dict[str, dict[str, dict[str, float]]] = {}
    for seed in seed_order:
        seed_unit_means[seed] = {
            unit: {
                metric: float(np.mean(values[seed][unit][metric]))
                for metric in required_metrics
            }
            for unit in unit_order
        }

    replicates = int(registry["protocol"]["bootstrap_replicates"])
    confidence_level = float(registry["protocol"]["confidence_level"])
    bootstrap_seed = int(registry["protocol"].get("bootstrap_seed", 20260715))
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_indices = (
        rng.integers(0, len(unit_order), size=(replicates, len(unit_order)))
        if len(unit_order) > 1
        else np.zeros((replicates, 1), dtype=np.int64)
    )
    alpha = (1.0 - confidence_level) / 2.0
    summaries: dict[str, dict[str, Any]] = {}
    for metric in sorted(required_metrics):
        seed_means = np.asarray(
            [
                np.mean([seed_unit_means[seed][unit][metric] for unit in unit_order])
                for seed in seed_order
            ],
            dtype=np.float64,
        )
        paired_unit_means = np.asarray(
            [
                np.mean([seed_unit_means[seed][unit][metric] for seed in seed_order])
                for unit in unit_order
            ],
            dtype=np.float64,
        )
        bootstrapped = paired_unit_means[bootstrap_indices].mean(axis=1)
        if len(seed_means) > 1:
            standard_deviation = float(np.std(seed_means, ddof=1))
        elif len(paired_unit_means) > 1:
            standard_deviation = float(np.std(paired_unit_means, ddof=1))
        else:
            standard_deviation = 0.0
        summaries[metric] = {
            "mean": float(seed_means.mean()),
            "std": standard_deviation,
            "ci95": [
                float(np.quantile(bootstrapped, alpha)),
                float(np.quantile(bootstrapped, 1.0 - alpha)),
            ],
            "seed_means": {
                seed: float(seed_means[index]) for index, seed in enumerate(seed_order)
            },
        }

    bundle: dict[str, Any] = {
        "schema_version": "activemap-paper-result-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "variant": variant,
        "budget": budget,
        "family": cell["family"],
        "split": cell["split"],
        "seeds": seed_order,
        "sample_count": len(rows),
        "unit_count": len(unit_order),
        "rows_by_seed": dict(sorted(row_counts_by_seed.items())),
        "bootstrap_unit": registry["protocol"]["bootstrap_unit"],
        "bootstrap_replicates": replicates,
        "bootstrap_seed": bootstrap_seed,
        "confidence_level": confidence_level,
        "registry_sha256": _sha256(registry_path),
        "source_observations": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in observation_paths
        ],
        "metrics": summaries,
    }
    if cell["split"] == "test":
        if frozen_test_ledger is None:
            raise ValueError("test bundles require a frozen test ledger")
        ledger = json.loads(frozen_test_ledger.read_text(encoding="utf-8"))
        if ledger.get("status") != "complete" or ledger.get("returncode") != 0:
            raise ValueError("frozen test ledger is not complete")
        bundle["frozen_test_ledger"] = str(frozen_test_ledger.resolve())
        bundle["frozen_test_ledger_sha256"] = _sha256(frozen_test_ledger)
    elif frozen_test_ledger is not None:
        raise ValueError("validation bundles must not reference a frozen test ledger")
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("registry", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("observations", type=Path, nargs="+")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--variant")
    parser.add_argument("--budget", type=float)
    parser.add_argument("--frozen-test-ledger", type=Path)
    args = parser.parse_args()
    bundle = build_result_bundle(
        args.registry,
        args.observations,
        experiment_id=args.experiment_id,
        variant=args.variant,
        budget=args.budget,
        frozen_test_ledger=args.frozen_test_ledger,
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "sample_count": bundle["sample_count"]}))


if __name__ == "__main__":
    main()
