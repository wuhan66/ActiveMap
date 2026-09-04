#!/usr/bin/env python3
"""Aggregate matched SN7 protocol-control summaries across model seeds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any


METHODS = ("direct", "react", "plan", "geo", "sense")
METRICS = (
    "terminal_accuracy",
    "false_edit_rate",
    "missed_edit_rate",
    "mean_acquisitions",
    "mean_cost",
    "mean_tool_calls",
    "mean_quality_cost_utility",
    "valid_action_rate",
    "fallback_episode_rate",
)


def _seed_dir(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected SEED=DIRECTORY")
    seed, directory = value.split("=", 1)
    return seed, Path(directory)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed-dir", action="append", type=_seed_dir, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    sources: list[dict[str, str]] = []
    for seed, directory in args.seed_dir:
        for method in METHODS:
            path = directory / f"{method}.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("split") != "val" or data.get("test_assets_read") is not False:
                raise ValueError(f"not a validation-only result: {path}")
            metrics = data["metrics"]
            row = {
                "seed": seed,
                "method": method,
                "sample_count": int(data["sample_count"]),
                **{metric: float(metrics[metric]) for metric in METRICS},
            }
            rows.append(row)
            sources.append({"path": str(path), "sha256": _sha256(path)})

    aggregates: list[dict[str, Any]] = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        aggregates.append(
            {
                "method": method,
                "seed_count": len(selected),
                "sample_count_per_seed": selected[0]["sample_count"],
                **{
                    f"mean_{metric}": mean(row[metric] for row in selected)
                    for metric in METRICS
                },
                "strict_action_validity_pass_all_seeds": all(
                    row["valid_action_rate"] >= 0.95 for row in selected
                ),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    fields = ("seed", "method", "sample_count", *METRICS)
    with (args.output_dir / "per_seed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# SN7 Modern Protocol Controls",
        "",
        "Clean-room controls share the same Qwen3-VL backbone, SFT data, validation",
        "episodes, budgets, and executable evaluator. Only the controller protocol differs.",
        "",
        "| Method | Seeds | Accuracy | False edit | Acquisition | Cost | Tool calls | Q-C utility | Valid action | Strict gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in aggregates:
        lines.append(
            "| {method} | {seed_count} | {accuracy} | {false_edit} | {acquisition} | "
            "{cost} | {tools} | {utility} | {valid} | {gate} |".format(
                method=row["method"],
                seed_count=row["seed_count"],
                accuracy=_fmt(row["mean_terminal_accuracy"]),
                false_edit=_fmt(row["mean_false_edit_rate"]),
                acquisition=_fmt(row["mean_mean_acquisitions"]),
                cost=_fmt(row["mean_mean_cost"]),
                tools=_fmt(row["mean_mean_tool_calls"]),
                utility=_fmt(row["mean_mean_quality_cost_utility"]),
                valid=_fmt(row["mean_valid_action_rate"]),
                gate="PASS"
                if row["strict_action_validity_pass_all_seeds"]
                else "FAIL",
            )
        )
    lines.extend(
        [
            "",
            "These are protocol controls, not official reproductions of the named systems.",
            "Zero tool calls are reported as a negative control result, not as cost-aware success.",
            "",
        ]
    )
    (args.output_dir / "table.md").write_text("\n".join(lines), encoding="utf-8")

    bundle = {
        "schema_version": "sn7-modern-protocol-control-table-v1",
        "split": "val",
        "methods": list(METHODS),
        "seeds": [seed for seed, _ in args.seed_dir],
        "rows": rows,
        "aggregates": aggregates,
        "sources": sources,
        "test_assets_read": False,
    }
    (args.output_dir / "table.json").write_text(
        json.dumps(bundle, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
