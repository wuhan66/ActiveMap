#!/usr/bin/env python3
"""Export a complete paper table and promotion decision from seed aggregation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRICS = (
    "selection_macro_f1",
    "realized_utility_mean",
    "exact_evidence_recall",
    "mean_regret",
    "false_call_rate",
    "harmful_call_rate_all_states",
    "mean_cost_per_state",
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def build_rows(aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    if aggregate.get("schema_version") != "active-catalog-fixed-seed-aoi-bootstrap-v1":
        raise ValueError("unexpected aggregate schema")
    if aggregate.get("test_assets_read") is not False:
        raise ValueError("paper table input must not read test assets")
    intervals = aggregate["fixed_seed_mean_intervals"]
    active = {"method": "ActiveMap (Qwen3-VL-4B)", "kind": "learned"}
    for metric in METRICS:
        item = intervals[metric]
        active[metric] = item["observed_fixed_seed_mean"]
        active[f"{metric}_ci95_low"] = item["ci95_low"]
        active[f"{metric}_ci95_high"] = item["ci95_high"]
    rows = [active]
    for name, metrics in sorted(aggregate["baseline_metrics"].items()):
        row = {"method": name, "kind": "baseline"}
        for metric in METRICS:
            row[metric] = metrics[metric]
            row[f"{metric}_ci95_low"] = None
            row[f"{metric}_ci95_high"] = None
        rows.append(row)
    return rows


def promotion_gate(aggregate: dict[str, Any]) -> dict[str, Any]:
    intervals = aggregate["fixed_seed_mean_intervals"]
    model = {
        key: value["observed_fixed_seed_mean"] for key, value in intervals.items()
    }
    random = aggregate["baseline_metrics"]["random"]
    checks = {
        "three_or_more_seeds": int(aggregate["seed_count"]) >= 3,
        "utility_aoi_ci_above_zero": intervals["realized_utility_mean"]["ci95_low"] > 0,
        "macro_f1_delta_aoi_ci_above_zero": intervals[
            "selection_macro_f1_delta_vs_always_stop"
        ]["ci95_low"]
        > 0,
        "false_call_rate_at_most_0_10": model["false_call_rate"] <= 0.10,
        "exact_evidence_recall_above_random": model["exact_evidence_recall"]
        > random["exact_evidence_recall"],
        "shared_aoi_resamples": aggregate.get(
            "shared_resample_indices_across_model_seeds"
        )
        is True,
        "test_assets_read_false": aggregate.get("test_assets_read") is False,
    }
    return {
        "schema_version": "active-catalog-paper-promotion-v1",
        "checks": checks,
        "passed": all(checks.values()),
        "seed_count": aggregate["seed_count"],
        "aoi_count": aggregate["aoi_count"],
        "test_assets_read": False,
    }


def format_value(value: Any) -> str:
    return "" if value is None else f"{float(value):.4f}"


def export(aggregate_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    aggregate = load_json(aggregate_path)
    rows = build_rows(aggregate)
    gate = promotion_gate(aggregate)
    output_dir.mkdir(parents=True)
    fields = ["method", "kind"] + [
        field
        for metric in METRICS
        for field in (metric, f"{metric}_ci95_low", f"{metric}_ci95_high")
    ]
    with (output_dir / "main_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    markdown_fields = ("method",) + METRICS
    lines = [
        "| " + " | ".join(markdown_fields) + " |",
        "| " + " | ".join("---" for _ in markdown_fields) + " |",
    ]
    for row in rows:
        values = [str(row["method"])]
        for metric in METRICS:
            value = format_value(row[metric])
            if row["kind"] == "learned":
                value += (
                    f" [{format_value(row[f'{metric}_ci95_low'])}, "
                    f"{format_value(row[f'{metric}_ci95_high'])}]"
                )
            values.append(value)
        lines.append("| " + " | ".join(values) + " |")
    (output_dir / "main_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "promotion_gate.json").write_text(
        json.dumps(gate, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": "active-catalog-paper-export-v1",
        "aggregate": str(aggregate_path.resolve()),
        "seed_count": aggregate["seed_count"],
        "aoi_count": aggregate["aoi_count"],
        "promotion_passed": gate["passed"],
        "outputs": ["main_results.csv", "main_results.md", "promotion_gate.json"],
        "test_assets_read": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("aggregate", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.aggregate, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
