#!/usr/bin/env python3
"""Summarize the bounded three-seed candidate-GRPO mechanism experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PRIMARY_CANDIDATES = (
    "episode_utility_v2_balanced_auc",
    "quality_cost_utility_auc",
    "raster_iou_gain_auc",
)
DISPLAY_METRICS = (
    "episode_utility_v2_balanced_auc",
    "raster_iou_auc",
    "false_edit_auc",
    "missed_edit_auc",
    "spent_cost_auc",
)


def summarize(aggregate: dict[str, Any]) -> dict[str, Any]:
    protocol = aggregate.get("protocol", {})
    if protocol.get("test_assets_read") is not False:
        raise ValueError("formal summary requires explicit test isolation")
    metrics = aggregate["candidate_minus_seed_matched_sft"]
    primary = next((name for name in PRIMARY_CANDIDATES if name in metrics), None)
    if primary is None:
        raise ValueError("aggregate has no supported primary utility metric")
    if "false_edit_auc" not in metrics:
        raise ValueError("aggregate has no false-edit safety metric")

    utility = metrics[primary]
    false_edit = metrics["false_edit_auc"]
    gates = {
        "all_seed_utility_positive": all(
            float(value) > 0.0 for value in utility["seed_deltas"]
        ),
        "utility_seed_mean_ci_excludes_zero": (
            float(utility["seed_mean_ci95_low"]) > 0.0
        ),
        "false_edit_mean_nonincreasing": float(false_edit["seed_mean_delta"]) <= 0.0,
        "false_edit_ci_noninferior_0p01": (
            float(false_edit["seed_mean_ci95_high"]) <= 0.01
        ),
    }
    return {
        "schema_version": "activemap-candidate-grpo-formal-summary-v1",
        "claim_boundary": (
            "33 naturally reachable validation states; bounded mechanism result only"
        ),
        "protocol": protocol,
        "primary_metric": primary,
        "gates": gates,
        "promote_to_main": all(gates.values()),
        "decision": (
            "bounded_main_text_candidate" if all(gates.values()) else "appendix_diagnostic"
        ),
        "metrics": {name: metrics[name] for name in DISPLAY_METRICS if name in metrics},
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Candidate-GRPO Formal Result",
        "",
        f"- Decision: **{summary['decision']}**",
        f"- Primary metric: `{summary['primary_metric']}`",
        f"- Boundary: {summary['claim_boundary']}",
        "- Test assets read: `false`",
        "",
        "## Promotion Gates",
        "",
        "| Gate | Pass |",
        "| --- | ---: |",
    ]
    for name, passed in summary["gates"].items():
        lines.append(f"| `{name}` | {'yes' if passed else 'no'} |")
    lines.extend(
        [
            "",
            "## Candidate minus seed-matched SFT",
            "",
            "| Metric | Seed deltas | Mean | Seed-level 95% CI |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for name, row in summary["metrics"].items():
        deltas = ", ".join(f"{float(value):+.6f}" for value in row["seed_deltas"])
        interval = (
            f"[{float(row['seed_mean_ci95_low']):+.6f}, "
            f"{float(row['seed_mean_ci95_high']):+.6f}]"
        )
        lines.append(
            f"| `{name}` | {deltas} | {float(row['seed_mean_delta']):+.6f} | {interval} |"
        )
    lines.extend(
        [
            "",
            "This table is a bounded recurrent-control mechanism experiment. It does not",
            "replace the complete-validation evidence for the main ActiveMap claims.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("aggregate", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("output_markdown", type=Path)
    args = parser.parse_args()
    summary = summarize(json.loads(args.aggregate.read_text(encoding="utf-8")))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
