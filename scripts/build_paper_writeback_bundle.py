#!/usr/bin/env python3
"""Build one method-budget writeback bundle with metric-specific task support."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scripts.build_paper_result_bundle import _select_cell, _sha256
from scripts.build_paper_rollout_bundle import _parse_seed_paths

WRITEBACK_METRICS = {
    "final_map_iou",
    "change_polygon_iou",
    "topology_valid_rate",
    "replay_consistency",
}
OFFICIAL_METRICS = {
    "apls_improvement",
    "pixel_f1_improvement",
    "no_change_error_rate",
}
METRICS = WRITEBACK_METRICS | OFFICIAL_METRICS


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def _target_operation(row: dict[str, Any]) -> str | None:
    target = row.get("target")
    if target in {"KEEP", "ADD", "DELETE", "RESHAPE"}:
        return str(target)
    if isinstance(target, str) and target.startswith("COMMIT:"):
        return target.split(":", 1)[1]
    if target == "REJECT":
        return "KEEP"
    return None


def _change_polygon_iou(row: dict[str, Any]) -> float:
    add = float(row["added_polygon_iou"])
    remove = float(row["removed_polygon_iou"])
    operation = _target_operation(row)
    if operation == "ADD":
        return add
    if operation == "DELETE":
        return remove
    return (add + remove) / 2.0


def _index_writeback(path: Path, budget: float) -> dict[str, dict[str, float]]:
    indexed: dict[str, dict[str, float]] = {}
    for row in _read_jsonl(path):
        if not math.isclose(float(row["budget"]), budget, abs_tol=1e-8):
            continue
        task_id = str(row["task_id"])
        if task_id in indexed:
            raise ValueError(f"{path}: duplicate writeback task {task_id} at budget {budget}")
        values = {
            "final_map_iou": float(row["raster_iou"]),
            "change_polygon_iou": _change_polygon_iou(row),
            "topology_valid_rate": float(bool(row["vector_delta_topology_valid"])),
            "replay_consistency": float(row["vector_replay_iou"]),
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"{path}: non-finite writeback metric for {task_id}")
        indexed[task_id] = values
    if not indexed:
        raise ValueError(f"{path}: no writeback rows at budget {budget}")
    return indexed


def _index_official(path: Path, budget: float) -> dict[str, dict[str, float]]:
    indexed: dict[str, dict[str, float]] = defaultdict(dict)
    for row in _read_jsonl(path):
        if not math.isclose(float(row["budget"]), budget, abs_tol=1e-8):
            continue
        task_id = str(row["task_id"])
        for metric in OFFICIAL_METRICS:
            value = row.get(metric)
            if value is None:
                continue
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"{path}: non-finite {metric} for {task_id}")
            if metric in indexed[task_id]:
                raise ValueError(f"{path}: duplicate {metric} for task {task_id}")
            indexed[task_id][metric] = numeric
    return dict(indexed)


def _summarize_metric(
    values: dict[str, dict[str, float]],
    *,
    replicates: int,
    confidence_level: float,
    bootstrap_seed: int,
) -> dict[str, Any]:
    seed_order = sorted(values)
    unit_sets = {seed: set(seed_values) for seed, seed_values in values.items()}
    units = unit_sets[seed_order[0]]
    if any(current != units for current in unit_sets.values()):
        raise ValueError("metric task support must be identical across seeds")
    if len(units) < 2:
        raise ValueError("each writeback metric needs at least two paired tasks")
    unit_order = sorted(units)
    seed_means = np.asarray(
        [np.mean([values[seed][unit] for unit in unit_order]) for seed in seed_order]
    )
    paired = np.asarray(
        [np.mean([values[seed][unit] for seed in seed_order]) for unit in unit_order]
    )
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, len(unit_order), size=(replicates, len(unit_order)))
    bootstrapped = paired[indices].mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "mean": float(seed_means.mean()),
        "std": float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0,
        "ci95": [
            float(np.quantile(bootstrapped, alpha)),
            float(np.quantile(bootstrapped, 1.0 - alpha)),
        ],
        "seed_means": {
            seed: float(seed_means[index]) for index, seed in enumerate(seed_order)
        },
        "unit_count": len(unit_order),
    }


def _summarize_seed_aggregate(
    values: dict[str, dict[str, float]],
    *,
    replicates: int,
    confidence_level: float,
    bootstrap_seed: int,
) -> dict[str, Any]:
    seed_order = sorted(values)
    if any(set(seed_values) != {"__aggregate__"} for seed_values in values.values()):
        raise ValueError("official aggregate metric requires one __aggregate__ value per seed")
    seed_values = np.asarray(
        [values[seed]["__aggregate__"] for seed in seed_order], dtype=np.float64
    )
    if len(seed_values) < 2:
        raise ValueError("official aggregate metric requires at least two model seeds")
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, len(seed_values), size=(replicates, len(seed_values)))
    bootstrapped = seed_values[indices].mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "mean": float(seed_values.mean()),
        "std": float(np.std(seed_values, ddof=1)),
        "ci95": [
            float(np.quantile(bootstrapped, alpha)),
            float(np.quantile(bootstrapped, 1.0 - alpha)),
        ],
        "seed_means": {
            seed: float(seed_values[index]) for index, seed in enumerate(seed_order)
        },
        "unit_count": len(seed_order),
        "bootstrap_unit": "model_seed_official_aggregate",
    }


def build_writeback_bundle(
    registry_path: Path,
    seed_writeback_paths: dict[str, Path],
    seed_official_paths: dict[str, Path],
    *,
    variant: str,
    budget: float,
    frozen_test_ledger: Path | None = None,
) -> dict[str, Any]:
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    experiment_id = "executable_vector_writeback"
    cell = _select_cell(registry, experiment_id, variant, budget)
    expected_seeds = set(map(str, cell["seeds"]))
    if set(seed_writeback_paths) != expected_seeds or set(seed_official_paths) != expected_seeds:
        raise ValueError("writeback and official seed files must exactly match registry seeds")

    metric_values: dict[str, dict[str, dict[str, float]]] = {
        metric: {} for metric in METRICS
    }
    all_tasks: set[str] | None = None
    for seed in sorted(expected_seeds):
        writeback = _index_writeback(seed_writeback_paths[seed], budget)
        official = _index_official(seed_official_paths[seed], budget)
        tasks = set(writeback)
        if all_tasks is None:
            all_tasks = tasks
        elif tasks != all_tasks:
            raise ValueError("writeback task ids must be identical across seeds")
        unknown_official = set(official) - tasks - {"__aggregate__"}
        if unknown_official:
            example = sorted(unknown_official)[:3]
            raise ValueError(f"official metrics contain unknown tasks: {example}")
        for metric in METRICS:
            source = writeback if metric in WRITEBACK_METRICS else official
            metric_values[metric][seed] = {
                task: values[metric]
                for task, values in source.items()
                if metric in values
            }

    required_metrics = set(map(str, cell["required_metrics"]))
    required_metrics.update(map(str, cell["primary_metrics"]))
    if required_metrics != METRICS:
        raise ValueError("writeback builder metric contract differs from registry")
    replicates = int(registry["protocol"]["bootstrap_replicates"])
    confidence_level = float(registry["protocol"]["confidence_level"])
    bootstrap_seed = int(registry["protocol"].get("bootstrap_seed", 20260715))
    summaries = {}
    for metric in sorted(METRICS):
        summarize = (
            _summarize_seed_aggregate
            if metric == "no_change_error_rate"
            and all(
                set(values) == {"__aggregate__"}
                for values in metric_values[metric].values()
            )
            else _summarize_metric
        )
        summaries[metric] = summarize(
            metric_values[metric],
            replicates=replicates,
            confidence_level=confidence_level,
            bootstrap_seed=bootstrap_seed,
        )
    bundle: dict[str, Any] = {
        "schema_version": "activemap-paper-result-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "variant": variant,
        "budget": budget,
        "family": "writeback",
        "split": cell["split"],
        "seeds": sorted(expected_seeds),
        "sample_count": len(all_tasks or set()) * len(expected_seeds),
        "unit_count": len(all_tasks or set()),
        "rows_by_seed": {
            seed: len(all_tasks or set()) for seed in sorted(expected_seeds)
        },
        "metric_unit_counts": {
            metric: summary["unit_count"] for metric, summary in summaries.items()
        },
        "bootstrap_unit": registry["protocol"]["bootstrap_unit"],
        "bootstrap_replicates": replicates,
        "bootstrap_seed": bootstrap_seed,
        "confidence_level": confidence_level,
        "registry_sha256": _sha256(registry_path),
        "source_observations": [
            {"path": str(path.resolve()), "sha256": _sha256(path), "kind": kind}
            for kind, paths in (
                ("writeback", seed_writeback_paths),
                ("official_graph", seed_official_paths),
            )
            for path in paths.values()
        ],
        "metrics": summaries,
    }
    if cell["split"] == "test":
        if frozen_test_ledger is None:
            raise ValueError("test writeback bundles require a frozen test ledger")
        ledger = json.loads(frozen_test_ledger.read_text(encoding="utf-8"))
        if ledger.get("status") != "complete" or ledger.get("returncode") != 0:
            raise ValueError("frozen test ledger is not complete")
        bundle["frozen_test_ledger"] = str(frozen_test_ledger.resolve())
        bundle["frozen_test_ledger_sha256"] = _sha256(frozen_test_ledger)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("registry", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--budget", type=float, required=True)
    parser.add_argument("--seed-writeback", action="append", required=True)
    parser.add_argument("--seed-official", action="append", required=True)
    parser.add_argument("--frozen-test-ledger", type=Path)
    args = parser.parse_args()
    bundle = build_writeback_bundle(
        args.registry,
        _parse_seed_paths(args.seed_writeback),
        _parse_seed_paths(args.seed_official),
        variant=args.variant,
        budget=args.budget,
        frozen_test_ledger=args.frozen_test_ledger,
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
