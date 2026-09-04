#!/usr/bin/env python3
"""AOI-cluster bootstrap for protocol-matched ChangeMamba audits."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.evaluate_sn7_changemamba import aggregate_rows

METRICS = (
    "committed_map_iou",
    "map_iou_delta",
    "change_iou",
    "operation_correct",
    "false_edit_rate",
    "missed_edit_rate",
    "wrong_edit_rate",
)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty audit: {path}")
    return rows


def cluster_bootstrap(
    rows: list[dict[str, Any]],
    *,
    draws: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    if draws < 100:
        raise ValueError("at least 100 bootstrap draws are required")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["aoi_id"]].append(row)
    aoi_ids = sorted(groups)
    if len(aoi_ids) < 2:
        raise ValueError("AOI bootstrap requires at least two AOIs")
    observed = aggregate_rows(rows)
    rng = np.random.default_rng(seed)
    samples = {metric: [] for metric in METRICS}
    accepted = 0
    attempts = 0
    max_attempts = draws * 20
    while accepted < draws and attempts < max_attempts:
        attempts += 1
        selected = rng.choice(aoi_ids, size=len(aoi_ids), replace=True)
        resampled = [
            row
            for aoi_id in selected
            for row in groups[str(aoi_id)]
        ]
        try:
            metrics = aggregate_rows(resampled)
        except ValueError:
            continue
        for metric in METRICS:
            samples[metric].append(metrics[metric])
        accepted += 1
    if accepted != draws:
        raise ValueError(
            f"only {accepted}/{draws} bootstrap draws had valid denominators"
        )
    return {
        metric: {
            "observed": observed[metric],
            "ci_low": float(np.quantile(samples[metric], 0.025)),
            "ci_high": float(np.quantile(samples[metric], 0.975)),
        }
        for metric in METRICS
    }


def audit_runs(
    audit_dirs: list[Path],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    if len(audit_dirs) < 2:
        raise ValueError("at least two audit directories are required")
    reports: list[dict[str, Any]] = []
    identity: list[tuple[str, str, str]] | None = None
    for index, audit_dir in enumerate(audit_dirs):
        summary = json.loads(
            (audit_dir / "summary.json").read_text(encoding="utf-8")
        )
        if summary.get("test_assets_read") is not False:
            raise ValueError(f"audit is not test-free: {audit_dir}")
        rows = _read_rows(audit_dir / "per_sample.jsonl")
        current_identity = [
            (row["sample_id"], row["aoi_id"], row["target_edit"])
            for row in rows
        ]
        if identity is None:
            identity = current_identity
        elif current_identity != identity:
            raise ValueError(f"sample identity mismatch: {audit_dir}")
        reports.append(
            {
                "audit_dir": str(audit_dir),
                "checkpoint_sha256": summary["checkpoint_sha256"],
                "bootstrap": cluster_bootstrap(
                    rows,
                    draws=draws,
                    seed=seed + index,
                ),
            }
        )

    aggregate: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        values = np.asarray(
            [report["bootstrap"][metric]["observed"] for report in reports],
            dtype=np.float64,
        )
        aggregate[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
        }
    return {
        "schema_version": "sn7-changemamba-aoi-bootstrap-v1",
        "run_count": len(reports),
        "sample_count": len(identity or []),
        "draws": draws,
        "seed": seed,
        "test_assets_read": False,
        "runs": reports,
        "aggregate": aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("audit_dirs", type=Path, nargs="+")
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()
    result = audit_runs(args.audit_dirs, draws=args.draws, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
