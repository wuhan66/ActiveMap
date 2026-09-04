#!/usr/bin/env python3
"""Fail-closed audit and paired table for the SN7 full-validation controller matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

import numpy as np

OPERATIONS = {"KEEP", "ADD", "DELETE", "RESHAPE"}
METRICS = (
    "terminal_accuracy",
    "false_edit_rate",
    "missed_edit_rate",
    "wrong_edit_rate",
    "mean_acquisitions",
    "mean_cost",
    "mean_tool_calls",
    "mean_quality_gain",
    "mean_quality_cost_utility",
    "balanced_utility",
    "safety_utility",
    "cost_aware_utility",
    "valid_action_rate",
    "fallback_episode_rate",
)


def _parse_run(value: str) -> tuple[str, str, Path]:
    identity, separator, raw_path = value.partition("=")
    method, colon, seed = identity.partition(":")
    if not separator or not colon or not method or not seed or not raw_path:
        raise argparse.ArgumentTypeError("run must use METHOD:SEED=PATH")
    return method, seed, Path(raw_path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty trace: {path}")
    return rows


def _resolve(path: Path) -> tuple[Path, Path | None, Path | None]:
    if path.is_file():
        return path, None, None
    candidates = (path / "evaluation" / "traces.jsonl", path / "traces.jsonl")
    traces = next((candidate for candidate in candidates if candidate.is_file()), None)
    if traces is None:
        raise FileNotFoundError(f"no traces.jsonl below {path}")
    summary = traces.parent / "summary.json"
    process = path / "process_result.json"
    return traces, summary if summary.is_file() else None, process if process.is_file() else None


def _row_values(row: dict[str, Any]) -> dict[str, float]:
    model_actions = int(row.get("model_action_count", 0))
    valid_actions = int(row.get("valid_action_count", 0))
    return {
        "terminal_accuracy": float(bool(row["terminal_correct"])),
        "false_edit_rate": float(bool(row["false_edit"])),
        "missed_edit_rate": float(bool(row["missed_edit"])),
        "wrong_edit_rate": float(bool(row["wrong_edit"])),
        "mean_acquisitions": float(row.get("acquisitions", 0)),
        "mean_cost": float(row.get("spent_cost", 0.0)),
        "mean_tool_calls": float(row.get("tool_calls", 0)),
        "mean_quality_gain": float(row.get("quality_gain", 0.0)),
        "mean_quality_cost_utility": float(row.get("quality_cost_utility", 0.0)),
        "balanced_utility": float(row.get("episode_utility_v2_proxy_balanced", 0.0)),
        "safety_utility": float(row.get("episode_utility_v2_proxy_safety", 0.0)),
        "cost_aware_utility": float(row.get("episode_utility_v2_proxy_cost_aware", 0.0)),
        "valid_action_rate": float(valid_actions / model_actions) if model_actions else 0.0,
        "fallback_episode_rate": float(int(row.get("fallback_count", 0)) > 0),
    }


def _metrics(rows: dict[str, dict[str, Any]]) -> dict[str, float]:
    values = [_row_values(row) for row in rows.values()]
    result = {metric: fmean(row[metric] for row in values) for metric in METRICS}
    model_actions = sum(int(row.get("model_action_count", 0)) for row in rows.values())
    valid_actions = sum(int(row.get("valid_action_count", 0)) for row in rows.values())
    result["valid_action_rate"] = valid_actions / model_actions if model_actions else 0.0
    return result


def _normalized_auc(points: list[tuple[float, float]]) -> float:
    ordered = sorted(points)
    if not ordered:
        raise ValueError("AUC requires budget points")
    if len(ordered) == 1:
        return float(ordered[0][1])
    area = sum(
        (right[0] - left[0]) * (left[1] + right[1]) / 2.0
        for left, right in zip(ordered, ordered[1:], strict=False)
    )
    return float(area / (ordered[-1][0] - ordered[0][0]))


def _support(rows: dict[str, dict[str, Any]]) -> dict[str, tuple[str, str, float]]:
    return {
        sample_id: (
            str(row["aoi_id"]),
            str(row["target_edit"]),
            float(row["budget"]),
        )
        for sample_id, row in rows.items()
    }


def _validate_rows(rows: list[dict[str, Any]], expected_count: int) -> dict[str, Any]:
    identities = [str(row["sample_id"]) for row in rows]
    checks = {
        "exact_count": len(rows) == expected_count,
        "unique_sample_ids": len(identities) == len(set(identities)),
        "validation_only": all(
            row.get("split") == "val" and row.get("test_assets_read") is False
            for row in rows
        ),
        "terminal_edits_valid": all(row.get("predicted_edit") in OPERATIONS for row in rows),
        "budget_safe": all(
            float(row.get("spent_cost", 0.0)) <= float(row.get("budget", 0.0)) + 1e-6
            and int(row.get("acquisitions", 0)) <= 2
            for row in rows
        ),
        "observable_state_has_no_target": all(
            "target_edit" not in (event.get("observable_state") or {})
            for row in rows
            for event in row.get("events", [])
        ),
        "finite_metrics": all(
            np.isfinite(value)
            for row in rows
            for value in _row_values(row).values()
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _paired_bootstrap(
    candidate_seeds: list[dict[str, dict[str, Any]]],
    reference: dict[str, dict[str, Any]],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    sample_ids = sorted(reference)
    groups: dict[str, list[str]] = defaultdict(list)
    for sample_id in sample_ids:
        groups[str(reference[sample_id]["aoi_id"])].append(sample_id)
    group_ids = sorted(groups)
    observed_by_seed = []
    group_sums = np.zeros(
        (len(candidate_seeds), len(group_ids), len(METRICS)), dtype=np.float64
    )
    group_counts = np.asarray([len(groups[group]) for group in group_ids], dtype=np.float64)
    for seed_index, candidate in enumerate(candidate_seeds):
        deltas = []
        for sample_id in sample_ids:
            left = _row_values(candidate[sample_id])
            right = _row_values(reference[sample_id])
            deltas.append({metric: left[metric] - right[metric] for metric in METRICS})
        observed_by_seed.append(
            {metric: fmean(row[metric] for row in deltas) for metric in METRICS}
        )
        for group_index, group in enumerate(group_ids):
            for sample_id in groups[group]:
                left = _row_values(candidate[sample_id])
                right = _row_values(reference[sample_id])
                for metric_index, metric in enumerate(METRICS):
                    group_sums[seed_index, group_index, metric_index] += (
                        left[metric] - right[metric]
                    )
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(group_ids), size=(repetitions, len(group_ids)))
    weights = np.zeros((repetitions, len(group_ids)), dtype=np.float64)
    for index, draw in enumerate(sampled):
        weights[index] = np.bincount(draw, minlength=len(group_ids))
    denominators = weights @ group_counts
    draws = np.stack(
        [
            (weights @ group_sums[index]) / denominators[:, None]
            for index in range(len(candidate_seeds))
        ]
    ).mean(axis=0)
    observed = {
        metric: fmean(seed_values[metric] for seed_values in observed_by_seed)
        for metric in METRICS
    }
    intervals = {
        metric: {
            "observed_delta": observed[metric],
            "ci95_low": float(np.quantile(draws[:, index], 0.025)),
            "ci95_high": float(np.quantile(draws[:, index], 0.975)),
        }
        for index, metric in enumerate(METRICS)
    }
    return {
        "seed_count": len(candidate_seeds),
        "aoi_count": len(group_ids),
        "repetitions": repetitions,
        "per_seed_delta": observed_by_seed,
        "intervals": intervals,
        "non_dominated_observed": (
            observed["mean_quality_cost_utility"] > 0.0
            and observed["terminal_accuracy"] >= 0.0
            and observed["false_edit_rate"] <= 0.0
        ),
        "strict_qc_gain_ci95": intervals["mean_quality_cost_utility"]["ci95_low"] > 0.0,
        "false_edit_noninferior_observed": observed["false_edit_rate"] <= 0.0,
    }


def audit(
    runs: list[tuple[str, str, Path]],
    *,
    reference_method: str,
    expected_count: int,
    repetitions: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    indexed: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(dict)
    provenance = []
    canonical_support = None
    all_checks_passed = True
    for method, model_seed, raw_path in runs:
        if model_seed in indexed[method]:
            raise ValueError(f"duplicate run: {method}:{model_seed}")
        trace_path, summary_path, process_path = _resolve(raw_path)
        rows_list = _read_jsonl(trace_path)
        validation = _validate_rows(rows_list, expected_count)
        all_checks_passed &= bool(validation["passed"])
        rows = {str(row["sample_id"]): row for row in rows_list}
        support = _support(rows)
        if canonical_support is None:
            canonical_support = support
        elif support != canonical_support:
            raise ValueError(f"support mismatch: {method}:{model_seed}")
        process_check = None
        if process_path is not None:
            process = _read_json(process_path)
            process_check = process.get("status") == "completed" and process.get("returncode") == 0
            all_checks_passed &= bool(process_check)
        summary_check = None
        if summary_path is not None:
            summary = _read_json(summary_path)
            summary_check = (
                summary.get("schema_version") == "active-catalog-closed-loop-evaluation-v1"
                and int(summary.get("sample_count", -1)) == expected_count
                and summary.get("split") == "val"
                and summary.get("test_assets_read") is False
            )
            all_checks_passed &= bool(summary_check)
        indexed[method][model_seed] = rows
        provenance.append(
            {
                "method": method,
                "seed": model_seed,
                "input": str(raw_path),
                "traces": str(trace_path),
                "trace_sha256": _sha256(trace_path),
                "row_checks": validation,
                "process_complete": process_check,
                "summary_valid": summary_check,
            }
        )
    if reference_method not in indexed or len(indexed[reference_method]) != 1:
        raise ValueError("reference method must contain exactly one run")
    reference = next(iter(indexed[reference_method].values()))
    per_run = []
    method_summary = []
    comparisons = {}
    comparisons_by_budget: dict[str, dict[str, Any]] = {}
    budget_per_run = []
    budget_summary = []
    auc_per_run = []
    auc_summary = []
    budgets = sorted({float(row["budget"]) for row in reference.values()})
    for method, seed_rows in sorted(indexed.items()):
        seed_metrics = []
        seed_budget_metrics: dict[str, dict[float, dict[str, float]]] = {}
        for model_seed, rows in sorted(seed_rows.items()):
            values = _metrics(rows)
            per_run.append({"method": method, "seed": model_seed, **values})
            seed_metrics.append(values)
            seed_budget_metrics[model_seed] = {}
            for budget in budgets:
                subset = {
                    sample_id: row
                    for sample_id, row in rows.items()
                    if float(row["budget"]) == budget
                }
                if not subset:
                    raise ValueError(f"missing budget {budget}: {method}:{model_seed}")
                budget_values = _metrics(subset)
                seed_budget_metrics[model_seed][budget] = budget_values
                budget_per_run.append(
                    {
                        "method": method,
                        "seed": model_seed,
                        "budget": budget,
                        "sample_count": len(subset),
                        **budget_values,
                    }
                )
            auc_values = {
                f"auc_{metric}": _normalized_auc(
                    [
                        (budget, seed_budget_metrics[model_seed][budget][metric])
                        for budget in budgets
                    ]
                )
                for metric in (
                    "mean_quality_gain",
                    "mean_quality_cost_utility",
                    "terminal_accuracy",
                    "false_edit_rate",
                    "mean_cost",
                )
            }
            auc_per_run.append({"method": method, "seed": model_seed, **auc_values})
        summary = {
            "method": method,
            "seed_count": len(seed_metrics),
            **{metric: fmean(row[metric] for row in seed_metrics) for metric in METRICS},
            **{
                f"std_{metric}": stdev(row[metric] for row in seed_metrics)
                if len(seed_metrics) > 1
                else 0.0
                for metric in METRICS
            },
        }
        method_summary.append(summary)
        for budget in budgets:
            selected = [values[budget] for values in seed_budget_metrics.values()]
            budget_summary.append(
                {
                    "method": method,
                    "seed_count": len(selected),
                    "budget": budget,
                    "sample_count_per_seed": sum(
                        float(row["budget"]) == budget for row in reference.values()
                    ),
                    **{
                        metric: fmean(row[metric] for row in selected)
                        for metric in METRICS
                    },
                }
            )
        selected_auc = [row for row in auc_per_run if row["method"] == method]
        auc_fields = [field for field in selected_auc[0] if field.startswith("auc_")]
        auc_summary.append(
            {
                "method": method,
                "seed_count": len(selected_auc),
                **{
                    field: fmean(float(row[field]) for row in selected_auc)
                    for field in auc_fields
                },
                **{
                    f"std_{field}": (
                        stdev(float(row[field]) for row in selected_auc)
                        if len(selected_auc) > 1
                        else 0.0
                    )
                    for field in auc_fields
                },
            }
        )
        if method != reference_method:
            comparisons[method] = _paired_bootstrap(
                list(seed_rows.values()),
                reference,
                repetitions=repetitions,
                seed=bootstrap_seed,
            )
            comparisons_by_budget[method] = {}
            for budget in budgets:
                candidates = [
                    {
                        sample_id: row
                        for sample_id, row in rows.items()
                        if float(row["budget"]) == budget
                    }
                    for rows in seed_rows.values()
                ]
                reference_budget = {
                    sample_id: row
                    for sample_id, row in reference.items()
                    if float(row["budget"]) == budget
                }
                comparisons_by_budget[method][str(budget)] = _paired_bootstrap(
                    candidates,
                    reference_budget,
                    repetitions=repetitions,
                    seed=bootstrap_seed,
                )
    return {
        "schema_version": "sn7-fullval-controller-matrix-audit-v1",
        "split": "val",
        "expected_count": expected_count,
        "reference_method": reference_method,
        "support_identical": True,
        "all_checks_passed": all_checks_passed,
        "per_run": per_run,
        "method_summary": method_summary,
        "comparisons_vs_reference": comparisons,
        "budgets": budgets,
        "budget_per_run": budget_per_run,
        "budget_summary": budget_summary,
        "auc_per_run": auc_per_run,
        "auc_summary": auc_summary,
        "comparisons_by_budget_vs_reference": comparisons_by_budget,
        "provenance": provenance,
        "test_assets_read": False,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# SN7 Full-Validation Controller Matrix",
        "",
        f"Support: {result['expected_count']} validation states; all checks: "
        f"{'PASS' if result['all_checks_passed'] else 'FAIL'}.",
        "",
        "| Method | Seeds | Accuracy | False edit | Missed edit | Cost | "
        "Quality gain | QC utility | Tool calls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["method_summary"]:
        lines.append(
            f"| {row['method']} | {row['seed_count']} | {row['terminal_accuracy']:.6f} | "
            f"{row['false_edit_rate']:.6f} | {row['missed_edit_rate']:.6f} | "
            f"{row['mean_cost']:.6f} | {row['mean_quality_gain']:.6f} | "
            f"{row['mean_quality_cost_utility']:.6f} | {row['mean_tool_calls']:.6f} |"
        )
    lines.extend(["", "## Paired deltas vs reference", ""])
    for method, comparison in result["comparisons_vs_reference"].items():
        qc = comparison["intervals"]["mean_quality_cost_utility"]
        false = comparison["intervals"]["false_edit_rate"]
        lines.append(
            f"- **{method}**: QC `{qc['observed_delta']:+.6f}` "
            f"95% CI `[{qc['ci95_low']:+.6f}, {qc['ci95_high']:+.6f}]`; "
            f"false edit `{false['observed_delta']:+.6f}`; "
            f"non-dominated `{comparison['non_dominated_observed']}`."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_budget_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# SN7 Budget Frontier",
        "",
        "| Method | Budget | Quality gain | Cost | QC utility | False edit | Accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["budget_summary"]:
        lines.append(
            f"| {row['method']} | {row['budget']:.1f} | "
            f"{row['mean_quality_gain']:.6f} | {row['mean_cost']:.6f} | "
            f"{row['mean_quality_cost_utility']:.6f} | "
            f"{row['false_edit_rate']:.6f} | {row['terminal_accuracy']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Normalized AUC",
            "",
            "| Method | Seeds | Quality AUC | QC AUC | Cost AUC | False-edit AUC |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["auc_summary"]:
        lines.append(
            f"| {row['method']} | {row['seed_count']} | "
            f"{row['auc_mean_quality_gain']:.6f} | "
            f"{row['auc_mean_quality_cost_utility']:.6f} | "
            f"{row['auc_mean_cost']:.6f} | {row['auc_false_edit_rate']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--run", action="append", type=_parse_run, required=True)
    parser.add_argument("--reference-method", default="old_vla")
    parser.add_argument("--expected-count", type=int, default=6369)
    parser.add_argument("--repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260801)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    result = audit(
        args.run,
        reference_method=args.reference_method,
        expected_count=args.expected_count,
        repetitions=args.repetitions,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(args.output_dir / "per_run.csv", result["per_run"])
    _write_csv(args.output_dir / "method_summary.csv", result["method_summary"])
    _write_csv(args.output_dir / "budget_per_run.csv", result["budget_per_run"])
    _write_csv(args.output_dir / "budget_summary.csv", result["budget_summary"])
    _write_csv(args.output_dir / "auc_per_run.csv", result["auc_per_run"])
    _write_csv(args.output_dir / "auc_summary.csv", result["auc_summary"])
    _write_markdown(args.output_dir / "table.md", result)
    _write_budget_markdown(args.output_dir / "budget_frontier.md", result)
    print(json.dumps({"output": str(args.output_dir), "passed": result["all_checks_passed"]}))
    if not result["all_checks_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
