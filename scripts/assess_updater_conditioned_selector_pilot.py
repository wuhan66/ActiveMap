#!/usr/bin/env python3
"""Assess the C5 stale-versus-refreshed selector pilot with paired AOI bootstrap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


METRICS = (
    "quality_cost_utility",
    "terminal_score_before_cost",
    "final_raster_iou",
    "quality_gain",
    "false_edit",
    "missed_edit",
    "additional_cost",
    "called",
)


def _read(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        sample_id = str(row["sample_id"])
        if sample_id in rows:
            raise ValueError(f"duplicate sample_id in {path}: {sample_id}")
        missing = (set(METRICS) | {"aoi_id"}) - row.keys()
        if missing:
            raise ValueError(f"{path}: row {sample_id} lacks {sorted(missing)}")
        rows[sample_id] = row
    if not rows:
        raise ValueError(f"empty evaluation: {path}")
    return rows


def _means(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {metric: float(np.mean([float(row[metric]) for row in rows])) for metric in METRICS}


def _paired_bootstrap(
    stale: dict[str, dict[str, Any]], refreshed: dict[str, dict[str, Any]], *, draws: int, seed: int
) -> dict[str, list[float]]:
    if stale.keys() != refreshed.keys():
        raise ValueError("stale and refreshed evaluations do not share the same sample IDs")
    groups: dict[str, list[str]] = {}
    for sample_id, row in stale.items():
        aoi = str(row["aoi_id"])
        if aoi != str(refreshed[sample_id]["aoi_id"]):
            raise ValueError(f"AOI mismatch for {sample_id}")
        groups.setdefault(aoi, []).append(sample_id)
    names = sorted(groups)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {metric: [] for metric in METRICS}
    for _ in range(draws):
        ids = [sample_id for name in rng.choice(names, size=len(names), replace=True) for sample_id in groups[str(name)]]
        for metric in METRICS:
            samples[metric].append(
                float(np.mean([float(refreshed[item][metric]) - float(stale[item][metric]) for item in ids]))
            )
    return {
        metric: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
        for metric, values in samples.items()
    }


def _markdown(summary: dict[str, Any]) -> str:
    delta = summary["refreshed_minus_stale"]
    ci = summary["aoi_cluster_bootstrap_95_ci"]
    rows = [
        "# C5 Updater-Conditioned Selector Pilot",
        "",
        "The primary comparison is refreshed `f1_on_f1` minus stale `f0_on_f1` on the",
        "same immutable f1 validation cache. Positive utility/quality deltas are better;",
        "negative false-edit, missed-edit, and additional-cost deltas are better.",
        "",
        "| Metric | Refreshed - stale | AOI bootstrap 95% CI |",
        "| --- | ---: | --- |",
    ]
    for metric in METRICS:
        interval = ci[metric]
        rows.append(f"| {metric} | {delta[metric]:+.6f} | [{interval[0]:+.6f}, {interval[1]:+.6f}] |")
    rows.extend(
        [
            "",
            f"Promotion-ready pilot gate: `{summary['gate']['passed']}`.",
            "This is a validation-only one-seed screen; it is not a final efficacy claim.",
            "",
        ]
    )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matched", type=Path, help="f0_on_f0 per_sample.jsonl")
    parser.add_argument("stale", type=Path, help="f0_on_f1 per_sample.jsonl")
    parser.add_argument("refreshed", type=Path, help="f1_on_f1 per_sample.jsonl")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260809)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    matched = _read(args.matched)
    stale = _read(args.stale)
    refreshed = _read(args.refreshed)
    # ``matched`` is a protocol reference from the f0 capability cache. The
    # causal refreshed-versus-stale contrast is paired only within the f1 cache.
    # _paired_bootstrap below enforces exact sample and AOI parity for that pair.
    paired_ci = _paired_bootstrap(stale, refreshed, draws=args.bootstrap_draws, seed=args.bootstrap_seed)
    delta = {
        metric: float(np.mean([float(refreshed[key][metric]) - float(stale[key][metric]) for key in stale]))
        for metric in METRICS
    }
    gate = {
        "utility_ci_positive": paired_ci["quality_cost_utility"][0] > 0.0,
        "final_iou_ci_nonnegative": paired_ci["final_raster_iou"][0] >= 0.0,
        "false_edit_ci_nonpositive": paired_ci["false_edit"][1] <= 0.0,
        "missed_edit_ci_nonpositive": paired_ci["missed_edit"][1] <= 0.0,
    }
    gate["passed"] = all(gate.values())
    summary = {
        "schema_version": "updater-conditioned-selector-pilot-assessment-v1",
        "matched": str(args.matched.resolve()),
        "stale": str(args.stale.resolve()),
        "refreshed": str(args.refreshed.resolve()),
        "sample_count": len(stale),
        "matched_metrics": _means(list(matched.values())),
        "stale_metrics": _means(list(stale.values())),
        "refreshed_metrics": _means(list(refreshed.values())),
        "refreshed_minus_stale": delta,
        "bootstrap_draws": args.bootstrap_draws,
        "aoi_cluster_bootstrap_95_ci": paired_ci,
        "gate": gate,
        "test_assets_read": False,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "SUMMARY.md").write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
