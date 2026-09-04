#!/usr/bin/env python3
"""Build predeclared paired quality-cost-safety reports from rollout traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import yaml

from scripts.build_paper_rollout_bundle import _parse_seed_paths

POINT_METRICS = {
    "quality_cost_utility": "quality_cost_utility",
    "final_evidence_quality": "final_evidence_quality",
    "terminal_accuracy": "terminal_correct",
    "false_edit_rate": "false_edit",
    "missed_edit_rate": "missed_edit",
    "mean_cost": "spent_cost",
    "mean_acquisitions": "acquisitions",
    "mean_tool_calls": "tool_calls",
    "mean_tool_cost": "tool_cost",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, *, split: str) -> dict[tuple[str, float], dict[str, Any]]:
    rows: dict[tuple[str, float], dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split") != split:
            raise ValueError(f"{path}:{line_number}: rollout split mismatch")
        key = (str(row["task_id"]), float(row["budget"]))
        if key in rows:
            raise ValueError(f"{path}:{line_number}: duplicate task-budget key")
        rows[key] = row
    if not rows:
        raise ValueError(f"no rollout rows in {path}")
    return rows


def _experiment(registry: dict[str, Any], experiment_id: str) -> dict[str, Any]:
    matches = [row for row in registry["experiments"] if row["id"] == experiment_id]
    if len(matches) != 1:
        raise ValueError(f"unknown experiment {experiment_id}")
    return matches[0]


def _expected_seeds(experiment: dict[str, Any]) -> set[str]:
    return set(map(str, experiment.get("seeds", ["deterministic"])))


def _method_task_values(
    rows_by_seed: dict[str, dict[tuple[str, float], dict[str, Any]]],
    task_ids: list[str],
    budget: float,
    raw_metric: str,
) -> np.ndarray:
    return np.asarray(
        [
            np.mean(
                [float(rows[(task_id, budget)][raw_metric]) for rows in rows_by_seed.values()]
            )
            for task_id in task_ids
        ],
        dtype=np.float64,
    )


def _task_auc(
    rows_by_seed: dict[str, dict[tuple[str, float], dict[str, Any]]],
    task_ids: list[str],
    budgets: list[float],
) -> np.ndarray:
    width = budgets[-1] - budgets[0]
    if width <= 0:
        raise ValueError("quality-cost AUC requires at least two distinct budgets")
    values = []
    for task_id in task_ids:
        seed_aucs = []
        for rows in rows_by_seed.values():
            utilities = np.asarray(
                [float(rows[(task_id, budget)]["quality_cost_utility"]) for budget in budgets]
            )
            trapezoid = getattr(np, "trapezoid", np.trapz)
            seed_aucs.append(float(trapezoid(utilities, budgets) / width))
        values.append(float(np.mean(seed_aucs)))
    return np.asarray(values, dtype=np.float64)


def _summary(values: np.ndarray, indices: np.ndarray, confidence: float) -> dict[str, Any]:
    alpha = (1.0 - confidence) / 2.0
    draws = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [
            float(np.quantile(draws, alpha)),
            float(np.quantile(draws, 1.0 - alpha)),
        ],
    }


def _binary_summary(values: np.ndarray, confidence: float) -> dict[str, Any]:
    """Wilson interval over task-cluster rates, including zero-event uncertainty."""

    probability = float(values.mean())
    count = len(values)
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    denominator = 1.0 + z**2 / count
    center = (probability + z**2 / (2.0 * count)) / denominator
    radius = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / count
            + z**2 / (4.0 * count**2)
        )
        / denominator
    )
    return {
        "mean": probability,
        "ci95": [max(0.0, center - radius), min(1.0, center + radius)],
        "interval": "wilson_task_cluster",
    }


def _paired_summary(
    baseline: np.ndarray,
    candidate: np.ndarray,
    indices: np.ndarray,
    confidence: float,
    *,
    binary: bool = False,
) -> dict[str, Any]:
    if baseline.shape != candidate.shape:
        raise ValueError("paired metric arrays do not match")
    return {
        "baseline": (
            _binary_summary(baseline, confidence)
            if binary
            else _summary(baseline, indices, confidence)
        ),
        "candidate": (
            _binary_summary(candidate, confidence)
            if binary
            else _summary(candidate, indices, confidence)
        ),
        "delta_candidate_minus_baseline": _summary(
            candidate - baseline, indices, confidence
        ),
    }


def _complete_ledger(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or payload.get("returncode") != 0:
        raise ValueError("frozen test ledger is not complete")
    return payload


def build_report(
    registry_path: Path,
    baseline_paths: dict[str, Path],
    candidate_paths: dict[str, Path],
    *,
    comparison_id: str,
    frozen_test_ledger: Path | None = None,
) -> dict[str, Any]:
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    specifications = [
        row
        for row in registry.get("required_paired_comparisons", [])
        if row["id"] == comparison_id
    ]
    if len(specifications) != 1:
        raise ValueError(f"unknown paired comparison {comparison_id}")
    specification = specifications[0]
    baseline_experiment = _experiment(registry, specification["baseline_experiment"])
    candidate_experiment = _experiment(registry, specification["candidate_experiment"])
    if set(baseline_paths) != _expected_seeds(baseline_experiment):
        raise ValueError("baseline seed files do not match the registered experiment")
    if set(candidate_paths) != _expected_seeds(candidate_experiment):
        raise ValueError("candidate seed files do not match the registered experiment")
    split = "test" if specification.get("test_policy") == "frozen_once" else "val"
    if split == "test":
        if frozen_test_ledger is None:
            raise ValueError("test paired reports require a frozen test ledger")
        _complete_ledger(frozen_test_ledger)
    elif frozen_test_ledger is not None:
        raise ValueError("validation paired reports must not reference a test ledger")

    baseline = {
        seed: _load(path, split=split) for seed, path in sorted(baseline_paths.items())
    }
    candidate = {
        seed: _load(path, split=split) for seed, path in sorted(candidate_paths.items())
    }
    all_rows = list(baseline.values()) + list(candidate.values())
    supports = [set(rows) for rows in all_rows]
    support = supports[0]
    if not support or any(current != support for current in supports[1:]):
        raise ValueError("all seed rollouts must have identical task-budget support")
    first_rows = all_rows[0]
    for key in support:
        target = first_rows[key]["target"]
        if any(rows[key]["target"] != target for rows in all_rows[1:]):
            raise ValueError(f"paired rollout targets differ for {key}")
    task_ids = sorted({task_id for task_id, _ in support})
    budgets = sorted({budget for _, budget in support})
    if any(
        {budget for candidate_task, budget in support if candidate_task == task_id}
        != set(budgets)
        for task_id in task_ids
    ):
        raise ValueError("every task must contain all frozen budgets")

    protocol = registry["protocol"]
    replicates = int(protocol["bootstrap_replicates"])
    confidence = float(protocol["confidence_level"])
    rng = np.random.default_rng(int(protocol.get("bootstrap_seed", 20260715)))
    indices = rng.integers(0, len(task_ids), size=(replicates, len(task_ids)))
    by_budget: dict[str, Any] = {}
    for budget in budgets:
        metrics = {}
        for metric, raw_metric in POINT_METRICS.items():
            left = _method_task_values(baseline, task_ids, budget, raw_metric)
            right = _method_task_values(candidate, task_ids, budget, raw_metric)
            metrics[metric] = _paired_summary(
                left,
                right,
                indices,
                confidence,
                binary=metric
                in {"terminal_accuracy", "false_edit_rate", "missed_edit_rate"},
            )
        by_budget[f"{budget:g}"] = metrics
    auc = _paired_summary(
        _task_auc(baseline, task_ids, budgets),
        _task_auc(candidate, task_ids, budgets),
        indices,
        confidence,
    )

    primary_budget = float(protocol["primary_budget"]["muno21"])
    primary = by_budget[f"{primary_budget:g}"]
    margins = protocol["paired_comparison_margins"]
    utility_delta = primary["quality_cost_utility"]["delta_candidate_minus_baseline"]
    safety_delta = primary["false_edit_rate"]["delta_candidate_minus_baseline"]
    candidate_safety = primary["false_edit_rate"]["candidate"]
    cost_delta = primary["mean_cost"]["delta_candidate_minus_baseline"]
    auc_delta = auc["delta_candidate_minus_baseline"]
    gates = {
        "primary_utility_noninferior": utility_delta["ci95"][0]
        >= float(margins["min_primary_utility_delta"]),
        "quality_cost_auc_noninferior": auc_delta["ci95"][0]
        >= float(margins["min_quality_cost_auc_delta"]),
        "false_edit_delta_noninferior": safety_delta["ci95"][1]
        <= float(margins["max_false_edit_rate_delta"]),
        "absolute_false_edit_safe": candidate_safety["ci95"][1]
        <= float(protocol["safety_gates"]["max_false_edit_rate"]),
        "matched_cost": cost_delta["ci95"][1]
        <= float(margins["max_mean_cost_delta_for_matched_cost"]),
    }
    gates["quality_cost_safety_non_dominated"] = all(
        gates[name]
        for name in (
            "primary_utility_noninferior",
            "quality_cost_auc_noninferior",
            "false_edit_delta_noninferior",
            "absolute_false_edit_safe",
        )
    )

    sources = [
        {"side": side, "seed": seed, "path": str(path.resolve()), "sha256": _sha256(path)}
        for side, paths in (("baseline", baseline_paths), ("candidate", candidate_paths))
        for seed, path in sorted(paths.items())
    ]
    report: dict[str, Any] = {
        "schema_version": "activemap-paired-rollout-report-v1",
        "comparison_id": comparison_id,
        "claim": specification["claim"],
        "split": split,
        "baseline": {
            "experiment_id": specification["baseline_experiment"],
            "variant": specification.get("baseline_variant"),
            "seeds": sorted(baseline_paths),
        },
        "candidate": {
            "experiment_id": specification["candidate_experiment"],
            "variant": specification.get("candidate_variant"),
            "seeds": sorted(candidate_paths),
        },
        "task_count": len(task_ids),
        "budgets": budgets,
        "bootstrap": {
            "unit": "opaque_task_id_with_all_budgets",
            "replicates": replicates,
            "seed": int(protocol.get("bootstrap_seed", 20260715)),
            "confidence_level": confidence,
        },
        "by_budget": by_budget,
        "quality_cost_auc": auc,
        "gates": gates,
        "all_required_gates_passed": all(
            gates[name] for name in specification["required_gates"]
        ),
        "source_rollouts": sources,
        "registry_sha256": _sha256(registry_path),
        "test_assets_read": split == "test",
    }
    if frozen_test_ledger is not None:
        report["frozen_test_ledger"] = str(frozen_test_ledger.resolve())
        report["frozen_test_ledger_sha256"] = _sha256(frozen_test_ledger)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("registry", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--comparison-id", required=True)
    parser.add_argument("--baseline", action="append", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--frozen-test-ledger", type=Path)
    args = parser.parse_args()
    report = build_report(
        args.registry,
        _parse_seed_paths(args.baseline),
        _parse_seed_paths(args.candidate),
        comparison_id=args.comparison_id,
        frozen_test_ledger=args.frozen_test_ledger,
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
