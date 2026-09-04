#!/usr/bin/env python3
"""Finalize a gate-ranker target sweep without conflating tuning and replication."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from scripts.evaluate_active_catalog_selector import active_catalog_metrics
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from evaluate_active_catalog_selector import active_catalog_metrics


METRICS = (
    "selection_macro_f1",
    "selection_macro_f1_delta_vs_always_stop",
    "predicted_call_rate",
    "false_call_rate",
    "harmful_call_rate_all_states",
    "harmful_call_fraction_of_calls",
    "exact_evidence_recall",
    "random_exact_evidence_recall",
    "realized_utility_mean",
    "mean_regret",
    "mean_cost_per_state",
)
INVARIANT_FIELDS = (
    "aoi_id",
    "source_episode",
    "candidate_count",
    "target_selection",
    "target_evidence_id",
    "stop_utility",
    "oracle_utility",
)


@dataclass(frozen=True)
class RunSpec:
    label: str
    target: float
    seed: int
    trace: Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _load(spec: RunSpec) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not 0.0 < spec.target < 1.0:
        raise ValueError(f"acquire target must be in (0,1): {spec.label}")
    rows = [
        json.loads(line)
        for line in spec.trace.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not rows:
        raise ValueError(f"empty trace: {spec.trace}")
    ids = [str(row["example_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate example IDs: {spec.trace}")
    summary_path = spec.trace.parent / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema_version") != "active-catalog-visual-gate-ranker-evaluation-v1":
        raise ValueError(f"unexpected summary schema: {summary_path}")
    if summary.get("test_assets_read") is not False:
        raise ValueError(f"run is not validation-only: {summary_path}")
    recomputed = active_catalog_metrics(rows)
    for key in METRICS:
        if not math.isclose(
            float(recomputed[key]), float(summary["metrics"][key]), abs_tol=1e-12
        ):
            raise ValueError(f"summary metric mismatch for {spec.label}.{key}")
    return rows, summary


def _safety_checks(metrics: dict[str, float], valid_rate: float) -> dict[str, bool]:
    return {
        "valid_action_rate_1": valid_rate == 1.0,
        "nonzero_calls": metrics["predicted_call_rate"] > 0.0,
        "macro_f1_above_always_stop": (
            metrics["selection_macro_f1_delta_vs_always_stop"] > 0.0
        ),
        "false_call_rate_at_most_0_10": metrics["false_call_rate"] <= 0.10,
        "harmful_call_fraction_at_most_0_60": (
            metrics["harmful_call_fraction_of_calls"] <= 0.60
        ),
        "exact_recall_above_random": (
            metrics["exact_evidence_recall"]
            > metrics["random_exact_evidence_recall"]
        ),
    }


def _dominates(left: dict[str, float], right: dict[str, float]) -> bool:
    higher = ("selection_macro_f1", "exact_evidence_recall", "realized_utility_mean")
    lower = ("false_call_rate", "harmful_call_rate_all_states", "mean_cost_per_state")
    no_worse = all(left[key] >= right[key] for key in higher) and all(
        left[key] <= right[key] for key in lower
    )
    strictly_better = any(left[key] > right[key] for key in higher) or any(
        left[key] < right[key] for key in lower
    )
    return no_worse and strictly_better


def finalize(
    specs: list[RunSpec],
    *,
    tuning_seed: int,
    repetitions: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")
    labels = [spec.label for spec in specs]
    pairs = [(spec.target, spec.seed) for spec in specs]
    if len(labels) != len(set(labels)) or len(pairs) != len(set(pairs)):
        raise ValueError("run labels and target/seed pairs must be unique")
    if len({spec.target for spec in specs if spec.seed == tuning_seed}) < 2:
        raise ValueError("tuning seed must contain at least two acquire targets")

    loaded = {spec.label: _load(spec) for spec in specs}
    rows_by_label = {label: value[0] for label, value in loaded.items()}
    summaries = {label: value[1] for label, value in loaded.items()}
    reference_label = specs[0].label
    reference = {
        str(row["example_id"]): tuple(row.get(key) for key in INVARIANT_FIELDS)
        for row in rows_by_label[reference_label]
    }
    for label, rows in rows_by_label.items():
        protocol = {
            str(row["example_id"]): tuple(row.get(key) for key in INVARIANT_FIELDS)
            for row in rows
        }
        if protocol != reference or len(protocol) != len(rows):
            raise ValueError(f"run does not share frozen validation states: {label}")

    run_metrics = {
        label: active_catalog_metrics(rows) for label, rows in rows_by_label.items()
    }
    run_checks = {
        label: _safety_checks(
            run_metrics[label], float(summaries[label]["valid_action_rate"])
        )
        for label in labels
    }
    aoi_ids = sorted(
        {str(row["aoi_id"]) for row in rows_by_label[reference_label]}
    )
    if len(aoi_ids) < 2:
        raise ValueError("AOI bootstrap requires at least two AOIs")
    grouped = {
        label: {
            aoi: [row for row in rows if str(row["aoi_id"]) == aoi]
            for aoi in aoi_ids
        }
        for label, rows in rows_by_label.items()
    }
    by_target: dict[float, list[RunSpec]] = defaultdict(list)
    for spec in specs:
        by_target[spec.target].append(spec)
    run_draws = {label: {metric: [] for metric in METRICS} for label in labels}
    target_draws = {
        target: {metric: [] for metric in METRICS} for target in by_target
    }
    rng = random.Random(bootstrap_seed)
    for _ in range(repetitions):
        sampled_aois = rng.choices(aoi_ids, k=len(aoi_ids))
        sampled_metrics = {}
        for label in labels:
            metrics = active_catalog_metrics(
                [row for aoi in sampled_aois for row in grouped[label][aoi]]
            )
            sampled_metrics[label] = metrics
            for metric in METRICS:
                run_draws[label][metric].append(metrics[metric])
        for target, target_specs in by_target.items():
            sampled_specs = rng.choices(target_specs, k=len(target_specs))
            for metric in METRICS:
                target_draws[target][metric].append(
                    statistics.fmean(
                        sampled_metrics[spec.label][metric] for spec in sampled_specs
                    )
                )

    run_intervals = {
        label: {
            metric: {
                "observed": run_metrics[label][metric],
                "ci95_low": _quantile(values, 0.025),
                "ci95_high": _quantile(values, 0.975),
            }
            for metric, values in metrics.items()
        }
        for label, metrics in run_draws.items()
    }
    target_aggregates = {}
    for target, target_specs in sorted(by_target.items()):
        mean_metrics = {
            metric: statistics.fmean(
                run_metrics[spec.label][metric] for spec in target_specs
            )
            for metric in METRICS
        }
        target_aggregates[str(target)] = {
            "seed_count": len(target_specs),
            "seeds": sorted(spec.seed for spec in target_specs),
            "labels": sorted(spec.label for spec in target_specs),
            "mean_metrics": mean_metrics,
            "intervals": {
                metric: {
                    "observed": mean_metrics[metric],
                    "ci95_low": _quantile(values, 0.025),
                    "ci95_high": _quantile(values, 0.975),
                }
                for metric, values in target_draws[target].items()
            },
        }

    tuning_specs = [spec for spec in specs if spec.seed == tuning_seed]
    tuning_feasible = [
        spec for spec in tuning_specs if all(run_checks[spec.label].values())
    ]
    selection_pool = tuning_feasible or tuning_specs
    selected = max(
        selection_pool,
        key=lambda spec: (
            run_metrics[spec.label]["realized_utility_mean"],
            -run_metrics[spec.label]["harmful_call_rate_all_states"],
            -run_metrics[spec.label]["false_call_rate"],
            run_metrics[spec.label]["exact_evidence_recall"],
            run_metrics[spec.label]["selection_macro_f1"],
            -spec.target,
        ),
    )
    frontier = [
        spec.label
        for spec in tuning_specs
        if not any(
            other.label != spec.label
            and _dominates(run_metrics[other.label], run_metrics[spec.label])
            for other in tuning_specs
        )
    ]
    selected_aggregate = target_aggregates[str(selected.target)]
    selected_labels = selected_aggregate["labels"]
    promotion_checks = {
        "tuning_candidate_safety_feasible": all(run_checks[selected.label].values()),
        "three_or_more_independent_seeds": selected_aggregate["seed_count"] >= 3,
        "all_replications_safety_feasible": all(
            all(run_checks[label].values()) for label in selected_labels
        ),
        "utility_aoi_seed_ci_above_zero": selected_aggregate["intervals"][
            "realized_utility_mean"
        ]["ci95_low"]
        > 0.0,
        "macro_f1_delta_ci_above_zero": selected_aggregate["intervals"][
            "selection_macro_f1_delta_vs_always_stop"
        ]["ci95_low"]
        > 0.0,
        "test_assets_read_false": True,
    }
    return {
        "schema_version": "active-catalog-gate-ranker-round-v1",
        "protocol": {
            "split": "val",
            "tuning_seed": tuning_seed,
            "selection_rule": (
                "maximize proxy utility among safety-feasible tuning-seed targets; "
                "replication seeds never select the target"
            ),
            "bootstrap": {
                "repetitions": repetitions,
                "seed": bootstrap_seed,
                "unit": "shared AOI draws plus model-seed resampling",
            },
            "test_assets_read": False,
        },
        "sample_count_per_run": len(reference),
        "aoi_count": len(aoi_ids),
        "runs": {
            spec.label: {
                "acquire_target": spec.target,
                "seed": spec.seed,
                "trace": str(spec.trace.resolve()),
                "trace_sha256": _sha256(spec.trace),
                "metrics": run_metrics[spec.label],
                "intervals": run_intervals[spec.label],
                "safety_checks": run_checks[spec.label],
                "safety_feasible": all(run_checks[spec.label].values()),
            }
            for spec in specs
        },
        "target_aggregates": target_aggregates,
        "tuning_frontier": sorted(frontier),
        "selected_tuning_run": selected.label,
        "selected_acquire_target": selected.target,
        "promotion": {
            "checks": promotion_checks,
            "passed": all(promotion_checks.values()),
            "replication_labels": selected_labels,
        },
    }


def _parse_run(value: str) -> RunSpec:
    parts = value.split(",", 3)
    if len(parts) != 4:
        raise ValueError("runs must use LABEL,TARGET,SEED,TRACE")
    return RunSpec(parts[0], float(parts[1]), int(parts[2]), Path(parts[3]))


def export(report: dict[str, Any], output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    fields = [
        "label",
        "acquire_target",
        "seed",
        "safety_feasible",
        *METRICS,
    ]
    with (output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label, row in report["runs"].items():
            writer.writerow(
                {
                    "label": label,
                    "acquire_target": row["acquire_target"],
                    "seed": row["seed"],
                    "safety_feasible": row["safety_feasible"],
                    **{metric: row["metrics"][metric] for metric in METRICS},
                }
            )
    decision = {
        "schema_version": "active-catalog-gate-ranker-decision-v1",
        "selected_tuning_run": report["selected_tuning_run"],
        "selected_acquire_target": report["selected_acquire_target"],
        "tuning_frontier": report["tuning_frontier"],
        "promotion": report["promotion"],
        "test_assets_read": False,
    }
    (output_dir / "promotion_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--tuning-seed", type=int, default=20260720)
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260723)
    args = parser.parse_args()
    report = finalize(
        [_parse_run(value) for value in args.run],
        tuning_seed=args.tuning_seed,
        repetitions=args.repetitions,
        bootstrap_seed=args.bootstrap_seed,
    )
    export(report, args.output_dir)
    print(json.dumps(report["promotion"], indent=2))


if __name__ == "__main__":
    main()
