#!/usr/bin/env python3
"""Build an aligned validation-only SN7 controller table from rollout traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.compare_active_catalog_closed_loop import (
    load_rows,
    paired_aoi_bootstrap,
)
from scripts.evaluate_active_catalog_closed_loop import metrics


DISPLAY_METRICS = (
    ("terminal_accuracy", "Accuracy"),
    ("false_edit_rate", "False edit"),
    ("missed_edit_rate", "Missed edit"),
    ("mean_quality_gain", "Quality gain"),
    ("mean_cost", "Cost"),
    ("mean_quality_cost_utility", "Q-cost utility"),
    ("mean_acquisitions", "Acquisitions"),
    ("mean_tool_calls", "Tool calls"),
    ("mean_tool_cost", "Tool cost"),
    ("valid_action_rate", "Valid action"),
    ("zero_acquisition_rate", "Zero acquire"),
    ("predicted_keep_rate", "Pred KEEP"),
    ("prediction_entropy_normalized", "Action entropy"),
)

FRONTIER_METRICS = (
    ("terminal_accuracy", "Accuracy"),
    ("false_edit_rate", "False edit"),
    ("mean_quality_gain", "Quality gain"),
    ("mean_cost", "Cost"),
    ("mean_quality_cost_utility", "Q-cost utility"),
    ("mean_acquisitions", "Acquisitions"),
)


def _parse_method(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("method must be label=/path/to/traces.jsonl")
    return label, Path(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_table(
    method_paths: dict[str, Path],
    *,
    reference: str,
    candidate: str | None,
    expected_records: int,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    if reference not in method_paths:
        raise ValueError(f"missing reference method: {reference}")
    if candidate is not None and candidate not in method_paths:
        raise ValueError(f"missing candidate method: {candidate}")
    rows = {label: load_rows(path) for label, path in method_paths.items()}
    support = set(rows[reference])
    if len(support) != expected_records:
        raise ValueError(
            f"reference has {len(support)} records, expected {expected_records}"
        )
    for label, values in rows.items():
        if set(values) != support:
            raise ValueError(f"{label} does not have identical episode-budget support")
    for key in support:
        reference_row = rows[reference][key]
        for label, values in rows.items():
            row = values[key]
            if (
                row["target_edit"] != reference_row["target_edit"]
                or str(row["aoi_id"]) != str(reference_row["aoi_id"])
                or float(row["budget"]) != float(reference_row["budget"])
            ):
                raise ValueError(f"{label} task metadata mismatch for {key}")

    absolute = {
        label: metrics(list(values.values()))
        for label, values in rows.items()
    }
    versus_reference = {
        label: paired_aoi_bootstrap(
            values,
            rows[reference],
            repetitions=repetitions,
            seed=seed,
        )
        for label, values in rows.items()
        if label != reference
    }
    candidate_comparisons = (
        {
            label: paired_aoi_bootstrap(
                rows[candidate],
                values,
                repetitions=repetitions,
                seed=seed,
            )
            for label, values in rows.items()
            if label != candidate
        }
        if candidate is not None
        else None
    )
    budgets = sorted(
        {float(row["budget"]) for row in rows[reference].values()}
    )
    by_budget = {}
    for budget in budgets:
        budget_rows = {
            label: {
                key: row
                for key, row in values.items()
                if float(row["budget"]) == budget
            }
            for label, values in rows.items()
        }
        budget_support = budget_rows[reference]
        by_budget[f"{budget:g}"] = {
            "budget": budget,
            "record_count": len(budget_support),
            "aoi_count": len(
                {str(row["aoi_id"]) for row in budget_support.values()}
            ),
            "methods": {
                label: metrics(list(values.values()))
                for label, values in budget_rows.items()
            },
            "paired_vs_reference": {
                label: paired_aoi_bootstrap(
                    values,
                    budget_support,
                    repetitions=repetitions,
                    seed=seed,
                )
                for label, values in budget_rows.items()
                if label != reference
            },
        }
    return {
        "schema_version": "sn7-common-controller-validation-table-v1",
        "split": "val",
        "record_count": len(support),
        "aoi_count": len(
            {str(row["aoi_id"]) for row in rows[reference].values()}
        ),
        "reference": reference,
        "candidate": candidate,
        "methods": absolute,
        "paired_vs_reference": versus_reference,
        "candidate_paired_vs_all": candidate_comparisons,
        "by_budget": by_budget,
        "bootstrap": {
            "unit": "aoi_id",
            "repetitions": repetitions,
            "seed": seed,
        },
        "sources": {
            label: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for label, path in method_paths.items()
        },
        "quality_gain_semantics": "frozen_teacher_proxy",
        "test_assets_read": False,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    headers = ["Controller", *(label for _, label in DISPLAY_METRICS)]
    lines = [
        "# SN7 Common Controller Validation",
        "",
        (
            f"Aligned validation-only comparison on {payload['record_count']} "
            f"episode-budget records and {payload['aoi_count']} AOIs."
        ),
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * len(DISPLAY_METRICS)) + " |",
    ]
    for label, values in payload["methods"].items():
        cells = [label]
        for key, _ in DISPLAY_METRICS:
            value = values[key]
            cells.append(f"{value:.6f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            (
                f"Paired AOI bootstrap uses {payload['bootstrap']['repetitions']} "
                f"draws. Reference: `{payload['reference']}`."
            ),
            "",
            "Quality gain is a frozen-teacher proxy; executable writeback remains separate.",
            "Test assets were not read.",
            "",
            "## Matched Budget Frontier",
            "",
            "| Budget | Controller | "
            + " | ".join(label for _, label in FRONTIER_METRICS)
            + " |",
            "| ---: | --- | "
            + " | ".join("---:" for _ in FRONTIER_METRICS)
            + " |",
        ]
    )
    for budget, budget_payload in payload["by_budget"].items():
        for label, values in budget_payload["methods"].items():
            cells = [
                budget,
                label,
                *(f"{values[key]:.6f}" for key, _ in FRONTIER_METRICS),
            ]
            lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "Each budget uses identical episode and AOI support across controllers.",
            "Paired confidence intervals are stored in the JSON payload.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--method", action="append", type=_parse_method, required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate")
    parser.add_argument("--expected-records", type=int, default=512)
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    method_paths = dict(args.method)
    if len(method_paths) != len(args.method):
        raise ValueError("duplicate method labels")
    payload = build_table(
        method_paths,
        reference=args.reference,
        candidate=args.candidate,
        expected_records=args.expected_records,
        repetitions=args.repetitions,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "controller_table.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "controller_table.md").write_text(
        render_markdown(payload),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
