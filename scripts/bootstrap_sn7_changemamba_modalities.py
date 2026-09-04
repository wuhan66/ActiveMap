#!/usr/bin/env python3
"""Shared-AOI paired bootstrap for ChangeMamba modality ablations."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

MODES = ("image_prior", "image_only", "prior_only")
METRICS = (
    "committed_map_iou",
    "map_iou_delta",
    "change_iou",
    "operation_accuracy",
    "false_edit_rate",
    "missed_edit_rate",
    "wrong_edit_rate",
)
STAT_SIZE = 10


def _read_rows(directory: Path) -> list[dict[str, Any]]:
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if summary.get("test_assets_read") is not False:
        raise ValueError(f"audit is not test-free: {directory}")
    rows = [
        json.loads(line)
        for line in (directory / "per_sample.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty audit: {directory}")
    return rows


def _statistics(rows: list[dict[str, Any]]) -> np.ndarray:
    stable = [row for row in rows if row["target_edit"] == "KEEP"]
    updates = [row for row in rows if row["target_edit"] != "KEEP"]
    return np.asarray(
        (
            len(rows),
            len(stable),
            len(updates),
            sum(float(row["committed_map_iou"]) for row in rows),
            sum(float(row["map_iou_delta"]) for row in rows),
            sum(float(row["change_iou"]) for row in rows),
            sum(float(row["operation_correct"]) for row in rows),
            sum(float(row["false_edit"]) for row in stable),
            sum(float(row["missed_edit"]) for row in updates),
            sum(float(row["wrong_edit"]) for row in updates),
        ),
        dtype=np.float64,
    )


def _metrics(statistics: np.ndarray) -> dict[str, float]:
    values = np.asarray(statistics, dtype=np.float64)
    if values.shape != (STAT_SIZE,) or min(values[:3]) <= 0:
        raise ValueError("statistics require samples, stable cases, and updates")
    return {
        "committed_map_iou": float(values[3] / values[0]),
        "map_iou_delta": float(values[4] / values[0]),
        "change_iou": float(values[5] / values[0]),
        "operation_accuracy": float(values[6] / values[0]),
        "false_edit_rate": float(values[7] / values[1]),
        "missed_edit_rate": float(values[8] / values[2]),
        "wrong_edit_rate": float(values[9] / values[2]),
    }


def _grouped(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["aoi_id"])].append(row)
    return {aoi_id: _statistics(selected) for aoi_id, selected in groups.items()}


def _seed_mean(
    grouped_runs: list[dict[str, np.ndarray]], selected: list[str]
) -> dict[str, float]:
    per_seed = [
        _metrics(np.sum([grouped[aoi_id] for aoi_id in selected], axis=0))
        for grouped in grouped_runs
    ]
    return {
        metric: float(np.mean([seed[metric] for seed in per_seed]))
        for metric in METRICS
    }


def paired_bootstrap(
    audit_dirs: dict[str, list[Path]], *, draws: int, seed: int
) -> dict[str, Any]:
    if draws < 100:
        raise ValueError("at least 100 bootstrap draws are required")
    if set(audit_dirs) != set(MODES):
        raise ValueError(f"expected modes {MODES}")
    rows_by_mode: dict[str, list[list[dict[str, Any]]]] = {}
    reference_identity: list[tuple[str, str, str]] | None = None
    for mode in MODES:
        if len(audit_dirs[mode]) != 3:
            raise ValueError(f"{mode} requires exactly three audits")
        rows_by_mode[mode] = []
        for directory in audit_dirs[mode]:
            rows = _read_rows(directory)
            identity = [
                (row["sample_id"], row["aoi_id"], row["target_edit"]) for row in rows
            ]
            if reference_identity is None:
                reference_identity = identity
            elif identity != reference_identity:
                raise ValueError(f"sample identity mismatch: {directory}")
            rows_by_mode[mode].append(rows)

    grouped = {
        mode: [_grouped(rows) for rows in runs]
        for mode, runs in rows_by_mode.items()
    }
    aoi_ids = sorted(grouped["image_prior"][0])
    if len(aoi_ids) < 2:
        raise ValueError("AOI bootstrap requires at least two AOIs")
    if any(
        sorted(run) != aoi_ids
        for mode in MODES
        for run in grouped[mode]
    ):
        raise ValueError("AOI support differs across modes or seeds")

    def evaluate(selected: list[str]) -> dict[str, dict[str, float]]:
        modes = {
            mode: _seed_mean(grouped[mode], selected)
            for mode in MODES
        }
        result = dict(modes)
        for ablation in ("image_only", "prior_only"):
            result[f"full_minus_{ablation}"] = {
                metric: (
                    modes["image_prior"][metric] - modes[ablation][metric]
                )
                for metric in METRICS
            }
        return result

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
        "schema_version": "sn7-changemamba-modality-aoi-bootstrap-v1",
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
        "| Comparison | Map-IoU delta (95% AOI CI) | False edit (95% AOI CI) | Missed edit (95% AOI CI) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for section in (
        "image_prior",
        "image_only",
        "prior_only",
        "full_minus_image_only",
        "full_minus_prior_only",
    ):
        metrics = result["results"][section]
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
    parser.add_argument("--image-prior", type=Path, nargs=3, required=True)
    parser.add_argument("--image-only", type=Path, nargs=3, required=True)
    parser.add_argument("--prior-only", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()
    result = paired_bootstrap(
        {
            "image_prior": args.image_prior,
            "image_only": args.image_only,
            "prior_only": args.prior_only,
        },
        draws=args.draws,
        seed=args.seed,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.output_markdown is not None:
        args.output_markdown.write_text(_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
