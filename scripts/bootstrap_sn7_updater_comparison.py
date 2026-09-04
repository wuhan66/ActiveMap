#!/usr/bin/env python3
"""Shared-AOI paired bootstrap between two three-seed SN7 updaters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.bootstrap_sn7_changemamba_modalities import (
    METRICS,
    _grouped,
    _read_rows,
    _seed_mean,
)


def paired_bootstrap(
    baseline_dirs: list[Path],
    candidate_dirs: list[Path],
    *,
    baseline_name: str,
    candidate_name: str,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    if len(baseline_dirs) != 3 or len(candidate_dirs) != 3:
        raise ValueError("each updater requires exactly three audits")
    if draws < 100:
        raise ValueError("at least 100 bootstrap draws are required")
    rows_by_method: dict[str, list[list[dict[str, Any]]]] = {
        baseline_name: [],
        candidate_name: [],
    }
    reference_identity: list[tuple[str, str, str]] | None = None
    for method, directories in (
        (baseline_name, baseline_dirs),
        (candidate_name, candidate_dirs),
    ):
        for directory in directories:
            rows = _read_rows(directory)
            identity = [
                (row["sample_id"], row["aoi_id"], row["target_edit"]) for row in rows
            ]
            if reference_identity is None:
                reference_identity = identity
            elif identity != reference_identity:
                raise ValueError(f"sample identity mismatch: {directory}")
            rows_by_method[method].append(rows)
    grouped = {
        method: [_grouped(rows) for rows in runs]
        for method, runs in rows_by_method.items()
    }
    aoi_ids = sorted(grouped[baseline_name][0])
    if len(aoi_ids) < 2:
        raise ValueError("AOI bootstrap requires at least two AOIs")
    if any(
        sorted(run) != aoi_ids
        for runs in grouped.values()
        for run in runs
    ):
        raise ValueError("AOI support differs across methods or seeds")

    delta_name = f"{candidate_name}_minus_{baseline_name}"

    def evaluate(selected: list[str]) -> dict[str, dict[str, float]]:
        values = {
            method: _seed_mean(runs, selected)
            for method, runs in grouped.items()
        }
        values[delta_name] = {
            metric: values[candidate_name][metric] - values[baseline_name][metric]
            for metric in METRICS
        }
        return values

    observed = evaluate(aoi_ids)
    samples = {
        section: {metric: [] for metric in METRICS} for section in observed
    }
    rng = np.random.default_rng(seed)
    accepted = 0
    attempts = 0
    while accepted < draws and attempts < draws * 20:
        attempts += 1
        selected = [
            str(value)
            for value in rng.choice(aoi_ids, size=len(aoi_ids), replace=True)
        ]
        try:
            result = evaluate(selected)
        except ValueError:
            continue
        for section in samples:
            for metric in METRICS:
                samples[section][metric].append(result[section][metric])
        accepted += 1
    if accepted != draws:
        raise ValueError(f"only {accepted}/{draws} valid bootstrap draws")
    return {
        "schema_version": "sn7-updater-paired-aoi-bootstrap-v1",
        "baseline": baseline_name,
        "candidate": candidate_name,
        "draws": draws,
        "seed": seed,
        "sample_count": len(reference_identity or []),
        "aoi_count": len(aoi_ids),
        "test_assets_read": False,
        "results": {
            section: {
                metric: {
                    "observed": observed[section][metric],
                    "ci_low": float(np.quantile(samples[section][metric], 0.025)),
                    "ci_high": float(np.quantile(samples[section][metric], 0.975)),
                }
                for metric in METRICS
            }
            for section in observed
        },
    }


def _markdown(result: dict[str, Any]) -> str:
    rows = [
        "| Method / delta | Map-IoU delta (95% AOI CI) "
        "| False edit (95% AOI CI) | Missed edit (95% AOI CI) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for section, metrics in result["results"].items():
        cells = []
        for metric in ("map_iou_delta", "false_edit_rate", "missed_edit_rate"):
            value = metrics[metric]
            cells.append(
                f"{value['observed']:+.5f} "
                f"[{value['ci_low']:+.5f}, {value['ci_high']:+.5f}]"
            )
        rows.append(f"| {section} | {' | '.join(cells)} |")
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, nargs=3, required=True)
    parser.add_argument("--candidate", type=Path, nargs=3, required=True)
    parser.add_argument("--baseline-name", default="changemamba")
    parser.add_argument("--candidate-name", default="ban")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()
    result = paired_bootstrap(
        args.baseline,
        args.candidate,
        baseline_name=args.baseline_name,
        candidate_name=args.candidate_name,
        draws=args.draws,
        seed=args.seed,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.output_markdown is not None:
        args.output_markdown.write_text(_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
