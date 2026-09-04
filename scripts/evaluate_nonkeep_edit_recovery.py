#!/usr/bin/env python3
"""Audit validation-only recovery of ADD, DELETE, and RESHAPE map edits.

This is intentionally a post-writeback auditor. It accepts only paired
validation trajectories produced by fixed policies; it neither fits a model nor
chooses thresholds. The result records all three editable-operation slices so
an aggregate cannot silently be driven by KEEP/no-change maintenance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


OPERATIONS = ("ADD", "DELETE", "RESHAPE")
METRICS = (
    "map_quality_after",
    "map_quality_gain",
    "balanced_utility",
    "false_edit_rate",
    "missed_edit_rate",
    "wrong_edit_rate",
    "terminal_accuracy",
    "commit_rate",
    "tool_call_rate",
    "mean_cost",
)
PROMOTION_METRICS = (
    "map_quality_gain",
    "false_edit_rate",
    "missed_edit_rate",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_record(value: str) -> tuple[str, str, Path]:
    head, separator, raw_path = value.partition("=")
    seed, divider, policy = head.partition(":")
    if not separator or not divider or not seed or not policy:
        raise argparse.ArgumentTypeError(
            "record must be SEED:POLICY=/path/to/writebacks.jsonl"
        )
    return seed, policy, Path(raw_path)


def operation(value: Any) -> str:
    text = str(value or "KEEP").upper().replace("-", "_").replace(" ", "")
    text = text.removeprefix("COMMIT:")
    if text in {"", "REJECT", "STOP", "KEEP"}:
        return "KEEP"
    if text in OPERATIONS:
        return text
    raise ValueError(f"unsupported edit operation: {value!r}")


def pick(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    raise KeyError(f"none of {names!r} found in writeback row")


def pick_optional(row: dict[str, Any], *names: str, default: Any = None) -> Any:
    try:
        return pick(row, *names)
    except KeyError:
        return default


def row_identity(row: dict[str, Any]) -> tuple[str, float]:
    return str(pick(row, "task_id", "sample_id")), float(pick(row, "budget"))


def target_operation(row: dict[str, Any]) -> str:
    target = pick_optional(row, "target", "target_edit", "edit_type")
    if target is None and isinstance(row.get("metadata"), dict):
        target = row["metadata"].get("gt_edit")
    return operation(target)


def executed_operation(row: dict[str, Any]) -> str:
    return operation(
        pick_optional(
            row,
            "effective_operation",
            "executed_operation",
            "prediction",
            "predicted_edit",
            default="KEEP",
        )
    )


def bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def canonical_row(row: dict[str, Any]) -> dict[str, Any]:
    target = target_operation(row)
    executed = executed_operation(row)
    derived_false = target == "KEEP" and executed != "KEEP"
    derived_missed = target != "KEEP" and executed == "KEEP"
    derived_wrong = target != "KEEP" and executed not in {"KEEP", target}
    tool_calls = pick_optional(row, "tool_calls", "acquisitions", default=None)
    selected = pick_optional(row, "selected_evidence_ids", default=None)
    return {
        "key": row_identity(row),
        "aoi_id": str(pick(row, "aoi_id")),
        "target": target,
        "executed": executed,
        "map_quality_before": float(
            pick(row, "map_quality_before", "prior_raster_iou")
        ),
        "map_quality_after": float(pick(row, "map_quality_after", "raster_iou")),
        "balanced_utility": float(
            pick(
                row,
                "episode_utility_v2_balanced",
                "episode_utility_v2_proxy_balanced",
                "balanced_utility",
            )
        ),
        "false_edit": bool_value(pick_optional(row, "false_edit", default=derived_false)),
        "missed_edit": bool_value(pick_optional(row, "missed_edit", default=derived_missed)),
        "wrong_edit": bool_value(pick_optional(row, "wrong_edit", default=derived_wrong)),
        "terminal_correct": bool_value(
            pick_optional(row, "terminal_correct", default=target == executed)
        ),
        "commit": bool_value(
            pick_optional(row, "writeback_changed", "commit", default=executed != "KEEP")
        ),
        "tool_call": bool_value(
            pick_optional(
                row,
                "semantic_tool_called",
                default=(float(tool_calls or 0) > 0 or bool(selected)),
            )
        ),
        "spent_cost": float(pick(row, "spent_cost")),
    }


def load_rows(path: Path) -> tuple[dict[tuple[str, float], dict[str, Any]], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: dict[tuple[str, float], dict[str, Any]] = {}
    total = 0
    filtered_keep = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            total += 1
            if raw.get("split") != "val" or raw.get("test_assets_read") is not False:
                raise ValueError(
                    f"validation-only provenance failure in {path}:{line_number}"
                )
            normalized = canonical_row(raw)
            if normalized["target"] == "KEEP":
                filtered_keep += 1
                continue
            key = normalized["key"]
            if key in rows:
                raise ValueError(f"duplicate task-budget identity in {path}: {key}")
            rows[key] = normalized
    if not rows:
        raise ValueError(f"no non-KEEP rows in {path}")
    return rows, {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "total_rows": total,
        "nonkeep_rows": len(rows),
        "filtered_keep_rows": filtered_keep,
        "split": "val",
        "test_assets_read": False,
    }


def values(row: dict[str, Any]) -> dict[str, float]:
    after = float(row["map_quality_after"])
    before = float(row["map_quality_before"])
    return {
        "map_quality_after": after,
        "map_quality_gain": after - before,
        "balanced_utility": float(row["balanced_utility"]),
        "false_edit_rate": float(bool(row["false_edit"])),
        "missed_edit_rate": float(bool(row["missed_edit"])),
        "wrong_edit_rate": float(bool(row["wrong_edit"])),
        "terminal_accuracy": float(bool(row["terminal_correct"])),
        "commit_rate": float(bool(row["commit"])),
        "tool_call_rate": float(bool(row["tool_call"])),
        "mean_cost": float(row["spent_cost"]),
    }


def mean_by_metric(rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    records = list(rows)
    if not records:
        raise ValueError("cannot summarize empty support")
    metric_values = [values(row) for row in records]
    return {
        metric: statistics.fmean(record[metric] for record in metric_values)
        for metric in METRICS
    }


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot estimate quantile of an empty sample")
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def validate_inputs(
    inputs: dict[str, dict[str, dict[tuple[str, float], dict[str, Any]]]],
    *,
    reference_policy: str,
    candidate_policy: str,
    expected_seed_count: int,
) -> tuple[list[str], list[tuple[str, float]], list[str]]:
    seeds = sorted(inputs)
    if len(seeds) != expected_seed_count:
        raise ValueError(
            f"expected exactly {expected_seed_count} seeds, received {len(seeds)}"
        )
    expected_policies = {reference_policy, candidate_policy}
    all_keys: set[tuple[str, float]] | None = None
    all_aois: set[str] | None = None
    for seed in seeds:
        policies = inputs[seed]
        if set(policies) != expected_policies:
            raise ValueError(f"seed {seed} must contain policies {sorted(expected_policies)}")
        reference = policies[reference_policy]
        candidate = policies[candidate_policy]
        if reference.keys() != candidate.keys():
            raise ValueError(f"policy task-budget support differs for seed {seed}")
        for key in reference:
            reference_row = reference[key]
            candidate_row = candidate[key]
            if (
                reference_row["aoi_id"] != candidate_row["aoi_id"]
                or reference_row["target"] != candidate_row["target"]
            ):
                raise ValueError(f"policy target provenance differs for seed {seed}, key {key}")
        seed_keys = set(reference)
        seed_aois = {row["aoi_id"] for row in reference.values()}
        if all_keys is None:
            all_keys = seed_keys
            all_aois = seed_aois
        elif all_keys != seed_keys or all_aois != seed_aois:
            raise ValueError("model seeds must share identical task-budget and AOI support")
    assert all_keys is not None and all_aois is not None
    return seeds, sorted(all_keys), sorted(all_aois)


def delta_rows(
    reference: dict[tuple[str, float], dict[str, Any]],
    candidate: dict[tuple[str, float], dict[str, Any]],
    operation_filter: str | None,
) -> list[tuple[str, dict[str, float]]]:
    result = []
    for key in sorted(reference):
        reference_row = reference[key]
        if operation_filter is not None and reference_row["target"] != operation_filter:
            continue
        candidate_row = candidate[key]
        left = values(candidate_row)
        right = values(reference_row)
        result.append(
            (
                reference_row["aoi_id"],
                {metric: left[metric] - right[metric] for metric in METRICS},
            )
        )
    if not result:
        raise ValueError(f"empty support for operation {operation_filter or 'ALL_NONKEEP'}")
    return result


def summarize_comparison(
    inputs: dict[str, dict[str, dict[tuple[str, float], dict[str, Any]]]],
    *,
    seeds: list[str],
    reference_policy: str,
    candidate_policy: str,
    aoi_ids: list[str],
    operation_filter: str | None,
    repetitions: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    per_seed_rows = {
        seed: delta_rows(
            inputs[seed][reference_policy],
            inputs[seed][candidate_policy],
            operation_filter,
        )
        for seed in seeds
    }
    supports = {seed: len(rows) for seed, rows in per_seed_rows.items()}
    observed = {
        metric: statistics.fmean(
            statistics.fmean(delta[metric] for _, delta in rows)
            for rows in per_seed_rows.values()
        )
        for metric in METRICS
    }
    grouped: dict[str, dict[str, dict[str, list[float]]]] = {}
    for seed, rows in per_seed_rows.items():
        grouped[seed] = {aoi: {metric: [0.0, 0.0] for metric in METRICS} for aoi in aoi_ids}
        for aoi, delta in rows:
            for metric, value in delta.items():
                grouped[seed][aoi][metric][0] += value
                grouped[seed][aoi][metric][1] += 1.0
    for seed in seeds:
        for aoi in aoi_ids:
            if any(grouped[seed][aoi][metric][1] == 0 for metric in METRICS):
                raise ValueError(f"missing {operation_filter or 'ALL_NONKEEP'} support for {seed}/{aoi}")
    rng = random.Random(bootstrap_seed)
    draws = {metric: [] for metric in METRICS}
    for _ in range(repetitions):
        sampled_aois = [rng.choice(aoi_ids) for _ in aoi_ids]
        sampled_seeds = [rng.choice(seeds) for _ in seeds]
        for metric in METRICS:
            seed_means = []
            for seed in sampled_seeds:
                total = sum(grouped[seed][aoi][metric][0] for aoi in sampled_aois)
                count = sum(grouped[seed][aoi][metric][1] for aoi in sampled_aois)
                seed_means.append(total / count)
            draws[metric].append(statistics.fmean(seed_means))
    intervals = {
        metric: {
            "observed_delta": observed[metric],
            "ci95_low": quantile(draws[metric], 0.025),
            "ci95_high": quantile(draws[metric], 0.975),
        }
        for metric in METRICS
    }
    per_seed = {}
    for seed in seeds:
        reference_rows = [
            row
            for row in inputs[seed][reference_policy].values()
            if operation_filter is None or row["target"] == operation_filter
        ]
        candidate_rows = [
            row
            for row in inputs[seed][candidate_policy].values()
            if operation_filter is None or row["target"] == operation_filter
        ]
        per_seed[seed] = {
            "support": len(reference_rows),
            "reference": mean_by_metric(reference_rows),
            "candidate": mean_by_metric(candidate_rows),
            "candidate_minus_reference": {
                metric: statistics.fmean(delta[metric] for _, delta in per_seed_rows[seed])
                for metric in METRICS
            },
        }
    return {
        "operation": operation_filter or "ALL_NONKEEP",
        "support_per_seed": supports,
        "per_seed": per_seed,
        "candidate_minus_reference": intervals,
    }


def promotion_gate(
    comparisons: dict[str, dict[str, Any]],
    *,
    seed_count: int,
    aoi_count: int,
    minimum_aois: int,
    minimum_rows_per_operation_per_seed: int,
) -> dict[str, Any]:
    overall = comparisons["ALL_NONKEEP"]["candidate_minus_reference"]
    support = {
        operation: min(comparisons[operation]["support_per_seed"].values())
        for operation in OPERATIONS
    }
    checks = {
        "exactly_three_seeds": seed_count == 3,
        "minimum_aoi_support": aoi_count >= minimum_aois,
        "minimum_per_operation_support": all(
            count >= minimum_rows_per_operation_per_seed for count in support.values()
        ),
        "quality_gain_ci_positive": overall["map_quality_gain"]["ci95_low"] > 0.0,
        "false_edit_not_worse": overall["false_edit_rate"]["ci95_high"] <= 0.0,
        "missed_edit_not_worse": overall["missed_edit_rate"]["ci95_high"] <= 0.0,
    }
    return {
        "passes_strict_posthoc_diagnostic_gate": all(checks.values()),
        "checks": checks,
        "minimum_rows_per_operation_per_seed": minimum_rows_per_operation_per_seed,
        "support_per_operation_minimum_across_seeds": support,
        "primary_metrics": list(PROMOTION_METRICS),
    }


def markdown(result: dict[str, Any]) -> str:
    lines = [
        "# SN7 Validation Non-KEEP Edit-Recovery Audit",
        "",
        "This result is validation-only and cannot change the sealed SN7 test claim.",
        "",
        "| Target operation | Support/seed | Quality gain delta [95% CI] | False-edit delta [95% CI] | Missed-edit delta [95% CI] |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for operation_name, comparison in result["comparisons"].items():
        values_by_metric = comparison["candidate_minus_reference"]
        support = min(comparison["support_per_seed"].values())
        def cell(metric: str) -> str:
            value = values_by_metric[metric]
            return "{:+.6f} [{:+.6f}, {:+.6f}]".format(
                value["observed_delta"], value["ci95_low"], value["ci95_high"]
            )
        lines.append(
            f"| {operation_name} | {support} | {cell('map_quality_gain')} | "
            f"{cell('false_edit_rate')} | {cell('missed_edit_rate')} |"
        )
    gate = result["promotion_gate"]
    lines.extend(
        [
            "",
            "## Strict Post-hoc Diagnostic Gate",
            "",
            f"Passes strict post-hoc diagnostic gate: `{gate['passes_strict_posthoc_diagnostic_gate']}`",
            "",
        ]
    )
    for name, passed in gate["checks"].items():
        lines.append(f"- `{name}`: `{passed}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--record", action="append", type=parse_record, required=True)
    parser.add_argument("--reference-policy", required=True)
    parser.add_argument("--candidate-policy", required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260816)
    parser.add_argument("--minimum-aois", type=int, default=4)
    parser.add_argument("--minimum-rows-per-operation-per-seed", type=int, default=20)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.bootstrap_repetitions <= 0 or args.minimum_aois <= 0:
        raise ValueError("bootstrap repetitions and minimum AOIs must be positive")
    inputs: dict[str, dict[str, dict[tuple[str, float], dict[str, Any]]]] = defaultdict(dict)
    sources: dict[str, dict[str, Any]] = defaultdict(dict)
    for seed, policy, path in args.record:
        if policy in inputs[seed]:
            raise ValueError(f"duplicate seed-policy record: {seed}:{policy}")
        rows, provenance = load_rows(path)
        inputs[seed][policy] = rows
        sources[seed][policy] = provenance
    seeds, keys, aoi_ids = validate_inputs(
        dict(inputs),
        reference_policy=args.reference_policy,
        candidate_policy=args.candidate_policy,
        expected_seed_count=3,
    )
    comparisons = {
        name: summarize_comparison(
            dict(inputs),
            seeds=seeds,
            reference_policy=args.reference_policy,
            candidate_policy=args.candidate_policy,
            aoi_ids=aoi_ids,
            operation_filter=None if name == "ALL_NONKEEP" else name,
            repetitions=args.bootstrap_repetitions,
            bootstrap_seed=args.bootstrap_seed + index,
        )
        for index, name in enumerate(("ALL_NONKEEP", *OPERATIONS))
    }
    result = {
        "schema_version": "sn7-nonkeep-edit-recovery-validation-v1",
        "split": "val",
        "test_assets_read": False,
        "analysis_role": "posthoc_validation_only",
        "reference_policy": args.reference_policy,
        "candidate_policy": args.candidate_policy,
        "operations": list(OPERATIONS),
        "model_seeds": seeds,
        "aoi_count": len(aoi_ids),
        "task_budget_count": len(keys),
        "bootstrap": {
            "unit": "hierarchical_model_seed_then_aoi_paired",
            "repetitions": args.bootstrap_repetitions,
            "seed": args.bootstrap_seed,
        },
        "sources": dict(sources),
        "comparisons": comparisons,
        "promotion_gate": promotion_gate(
            comparisons,
            seed_count=len(seeds),
            aoi_count=len(aoi_ids),
            minimum_aois=args.minimum_aois,
            minimum_rows_per_operation_per_seed=args.minimum_rows_per_operation_per_seed,
        ),
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "table.md").write_text(markdown(result), encoding="utf-8")
    with (args.output_dir / "per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["operation", "model_seed", "side", "support", *METRICS])
        for operation_name, comparison in comparisons.items():
            for seed, values_by_seed in comparison["per_seed"].items():
                for side in ("reference", "candidate", "candidate_minus_reference"):
                    writer.writerow(
                        [
                            operation_name,
                            seed,
                            side,
                            values_by_seed["support"],
                            *[values_by_seed[side][metric] for metric in METRICS],
                        ]
                    )
    print(json.dumps(result["promotion_gate"], indent=2))


if __name__ == "__main__":
    main()
