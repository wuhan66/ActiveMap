#!/usr/bin/env python3
"""Paired AOI bootstrap for ChangeMamba always- versus safe-commit policies."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

METRICS = (
    "committed_map_iou",
    "map_iou_delta",
    "operation_accuracy",
    "false_edit_rate",
    "missed_edit_rate",
    "wrong_edit_rate",
    "accepted_commit_rate",
)
STAT_SIZE = 10


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty safe-commit audit: {path}")
    return rows


def metrics_from_acceptance(
    rows: list[dict[str, Any]], accepted: np.ndarray
) -> dict[str, float]:
    accepted = np.asarray(accepted, dtype=bool)
    if accepted.shape != (len(rows),):
        raise ValueError("acceptance vector must align with rows")
    predicted_edit = np.asarray(
        [
            row["predicted_edit"] if accepted[index] else "KEEP"
            for index, row in enumerate(rows)
        ]
    )
    target_edit = np.asarray([row["target_edit"] for row in rows])
    stable = target_edit == "KEEP"
    updates = ~stable
    if not stable.any() or not updates.any():
        raise ValueError("bootstrap sample requires stable and update examples")
    prior_iou = np.asarray([row["prior_map_iou"] for row in rows], dtype=float)
    updater_iou = np.asarray(
        [row["committed_map_iou"] for row in rows], dtype=float
    )
    final_iou = np.where(accepted, updater_iou, prior_iou)
    return {
        "committed_map_iou": float(final_iou.mean()),
        "map_iou_delta": float((final_iou - prior_iou).mean()),
        "operation_accuracy": float(np.mean(predicted_edit == target_edit)),
        "false_edit_rate": float(np.mean(predicted_edit[stable] != "KEEP")),
        "missed_edit_rate": float(np.mean(predicted_edit[updates] == "KEEP")),
        "wrong_edit_rate": float(
            np.mean(
                (predicted_edit[updates] != "KEEP")
                & (predicted_edit[updates] != target_edit[updates])
            )
        ),
        "accepted_commit_rate": float(accepted.mean()),
    }


def paired_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    candidate = np.asarray([row["predicted_edit"] != "KEEP" for row in rows])
    safe = np.asarray([row["safe_commit_accepted"] for row in rows])
    always_metrics = metrics_from_acceptance(rows, candidate)
    safe_metrics = metrics_from_acceptance(rows, safe)
    return {
        "always_commit": always_metrics,
        "safe_commit": safe_metrics,
        "safe_minus_always": {
            metric: safe_metrics[metric] - always_metrics[metric]
            for metric in METRICS
        },
    }


def sufficient_statistics(
    rows: list[dict[str, Any]], accepted: np.ndarray
) -> np.ndarray:
    accepted = np.asarray(accepted, dtype=bool)
    predicted_edit = np.asarray(
        [
            row["predicted_edit"] if accepted[index] else "KEEP"
            for index, row in enumerate(rows)
        ]
    )
    target_edit = np.asarray([row["target_edit"] for row in rows])
    stable = target_edit == "KEEP"
    updates = ~stable
    prior_iou = np.asarray([row["prior_map_iou"] for row in rows], dtype=float)
    updater_iou = np.asarray(
        [row["committed_map_iou"] for row in rows], dtype=float
    )
    final_iou = np.where(accepted, updater_iou, prior_iou)
    return np.asarray(
        (
            len(rows),
            stable.sum(),
            updates.sum(),
            final_iou.sum(),
            (final_iou - prior_iou).sum(),
            (predicted_edit == target_edit).sum(),
            (predicted_edit[stable] != "KEEP").sum(),
            (predicted_edit[updates] == "KEEP").sum(),
            (
                (predicted_edit[updates] != "KEEP")
                & (predicted_edit[updates] != target_edit[updates])
            ).sum(),
            accepted.sum(),
        ),
        dtype=np.float64,
    )


def metrics_from_statistics(statistics: np.ndarray) -> dict[str, float]:
    values = np.asarray(statistics, dtype=np.float64)
    if values.shape != (STAT_SIZE,) or values[0] <= 0:
        raise ValueError("invalid safe-commit sufficient statistics")
    if values[1] <= 0 or values[2] <= 0:
        raise ValueError("statistics require stable and update examples")
    return {
        "committed_map_iou": float(values[3] / values[0]),
        "map_iou_delta": float(values[4] / values[0]),
        "operation_accuracy": float(values[5] / values[0]),
        "false_edit_rate": float(values[6] / values[1]),
        "missed_edit_rate": float(values[7] / values[2]),
        "wrong_edit_rate": float(values[8] / values[2]),
        "accepted_commit_rate": float(values[9] / values[0]),
    }


def grouped_statistics(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, np.ndarray]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["aoi_id"])].append(row)
    result = {"always_commit": {}, "safe_commit": {}}
    for aoi_id, selected in groups.items():
        candidate = np.asarray(
            [row["predicted_edit"] != "KEEP" for row in selected]
        )
        safe = np.asarray([row["safe_commit_accepted"] for row in selected])
        result["always_commit"][aoi_id] = sufficient_statistics(
            selected, candidate
        )
        result["safe_commit"][aoi_id] = sufficient_statistics(selected, safe)
    return result


def paired_from_grouped(
    grouped: dict[str, dict[str, np.ndarray]],
    selected_aoi_ids: list[str],
) -> dict[str, dict[str, float]]:
    result = {}
    for section in ("always_commit", "safe_commit"):
        statistics = np.sum(
            [grouped[section][aoi_id] for aoi_id in selected_aoi_ids],
            axis=0,
        )
        result[section] = metrics_from_statistics(statistics)
    result["safe_minus_always"] = {
        metric: result["safe_commit"][metric] - result["always_commit"][metric]
        for metric in METRICS
    }
    return result


def cluster_bootstrap(
    rows: list[dict[str, Any]], *, draws: int, seed: int
) -> dict[str, Any]:
    if draws < 100:
        raise ValueError("at least 100 bootstrap draws are required")
    grouped = grouped_statistics(rows)
    aoi_ids = sorted(grouped["always_commit"])
    if len(aoi_ids) < 2:
        raise ValueError("AOI bootstrap requires at least two AOIs")
    observed = paired_from_grouped(grouped, aoi_ids)
    sampled = {section: {metric: [] for metric in METRICS} for section in observed}
    rng = np.random.default_rng(seed)
    accepted_draws = 0
    attempts = 0
    while accepted_draws < draws and attempts < draws * 20:
        attempts += 1
        selected = rng.choice(aoi_ids, size=len(aoi_ids), replace=True)
        try:
            values = paired_from_grouped(
                grouped, [str(aoi_id) for aoi_id in selected]
            )
        except ValueError:
            continue
        for section in sampled:
            for metric in METRICS:
                sampled[section][metric].append(values[section][metric])
        accepted_draws += 1
    if accepted_draws != draws:
        raise ValueError(f"only {accepted_draws}/{draws} valid bootstrap draws")
    return {
        section: {
            metric: {
                "observed": observed[section][metric],
                "ci_low": float(np.quantile(sampled[section][metric], 0.025)),
                "ci_high": float(np.quantile(sampled[section][metric], 0.975)),
            }
            for metric in METRICS
        }
        for section in observed
    }


def seed_mean_cluster_bootstrap(
    rows_by_run: list[list[dict[str, Any]]],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    grouped_runs = [grouped_statistics(rows) for rows in rows_by_run]
    aoi_ids = sorted(grouped_runs[0]["always_commit"])
    if any(
        sorted(grouped["always_commit"]) != aoi_ids
        for grouped in grouped_runs[1:]
    ):
        raise ValueError("seed-mean bootstrap requires identical AOI ids")
    observed_runs = [
        paired_from_grouped(grouped, aoi_ids) for grouped in grouped_runs
    ]
    observed = {
        section: {
            metric: float(
                np.mean([run[section][metric] for run in observed_runs])
            )
            for metric in METRICS
        }
        for section in observed_runs[0]
    }
    sampled = {section: {metric: [] for metric in METRICS} for section in observed}
    rng = np.random.default_rng(seed)
    accepted_draws = 0
    attempts = 0
    while accepted_draws < draws and attempts < draws * 20:
        attempts += 1
        selected = [
            str(aoi_id)
            for aoi_id in rng.choice(
                aoi_ids, size=len(aoi_ids), replace=True
            )
        ]
        try:
            run_values = [
                paired_from_grouped(grouped, selected)
                for grouped in grouped_runs
            ]
        except ValueError:
            continue
        for section in sampled:
            for metric in METRICS:
                sampled[section][metric].append(
                    float(
                        np.mean(
                            [
                                run[section][metric]
                                for run in run_values
                            ]
                        )
                    )
                )
        accepted_draws += 1
    if accepted_draws != draws:
        raise ValueError(
            f"only {accepted_draws}/{draws} valid seed-mean bootstrap draws"
        )
    return {
        section: {
            metric: {
                "observed": observed[section][metric],
                "ci_low": float(np.quantile(sampled[section][metric], 0.025)),
                "ci_high": float(np.quantile(sampled[section][metric], 0.975)),
            }
            for metric in METRICS
        }
        for section in observed
    }


def audit_runs(
    run_dirs: list[Path],
    *,
    draws: int,
    seed: int,
    expected_test_assets_read: bool = False,
) -> dict[str, Any]:
    if len(run_dirs) < 2:
        raise ValueError("at least two safe-commit runs are required")
    reports = []
    identity = None
    rows_by_run = []
    for index, run_dir in enumerate(run_dirs):
        summary = json.loads(
            (run_dir / "summary.json").read_text(encoding="utf-8")
        )
        if summary.get("test_assets_read") is not expected_test_assets_read:
            expected = "frozen-test" if expected_test_assets_read else "test-free"
            raise ValueError(f"safe-commit run is not {expected}: {run_dir}")
        rows = read_rows(run_dir / "per_sample.jsonl")
        current_identity = [
            (row["sample_id"], row["aoi_id"], row["target_edit"])
            for row in rows
        ]
        if identity is None:
            identity = current_identity
        elif current_identity != identity:
            raise ValueError(f"sample identity mismatch: {run_dir}")
        rows_by_run.append(rows)
        reports.append(
            {
                "run_dir": str(run_dir),
                "bootstrap": cluster_bootstrap(
                    rows, draws=draws, seed=seed + index
                ),
            }
        )
    aggregate = {}
    for section in ("always_commit", "safe_commit", "safe_minus_always"):
        aggregate[section] = {}
        for metric in METRICS:
            values = np.asarray(
                [
                    report["bootstrap"][section][metric]["observed"]
                    for report in reports
                ],
                dtype=np.float64,
            )
            aggregate[section][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
            }
    return {
        "schema_version": "sn7-changemamba-safe-commit-bootstrap-v1",
        "run_count": len(reports),
        "sample_count": len(identity or []),
        "draws": draws,
        "seed": seed,
        "runs": reports,
        "aggregate": aggregate,
        "seed_mean_paired_bootstrap": seed_mean_cluster_bootstrap(
            rows_by_run, draws=draws, seed=seed + len(run_dirs)
        ),
        "test_assets_read": expected_test_assets_read,
    }


def markdown(result: dict[str, Any]) -> str:
    paired = result["seed_mean_paired_bootstrap"]
    lines = [
        "| Metric | Always commit | Safe commit | Paired delta (95% CI) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric in METRICS:
        always = paired["always_commit"][metric]["observed"]
        safe = paired["safe_commit"][metric]["observed"]
        delta = paired["safe_minus_always"][metric]
        lines.append(
            f"| {metric} | {always:.6f} | {safe:.6f} | "
            f"{delta['observed']:+.6f} "
            f"[{delta['ci_low']:+.6f}, {delta['ci_high']:+.6f}] |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--frozen-test", action="store_true")
    args = parser.parse_args()
    if args.frozen_test:
        from activemap.frozen_test import assert_frozen_test_access

        assert_frozen_test_access()
    result = audit_runs(
        args.run_dirs,
        draws=args.draws,
        seed=args.seed,
        expected_test_assets_read=args.frozen_test,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    if args.output_markdown is not None:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
