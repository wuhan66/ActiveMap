#!/usr/bin/env python3
"""Audit whether observable rollout features can safely arbitrate policy edits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


FEATURES = (
    "initial_evidence_quality",
    "final_evidence_quality",
    "evidence_quality_gain",
    "mean_tool_belief_l1_delta",
    "spent_cost",
    "tool_calls",
)


def load_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[str(row["sample_id"])] = row
    return rows


def metric(rows: Iterable[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return sum(values) / len(values)


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "terminal_accuracy": metric(rows, "terminal_correct"),
        "false_edit_rate": metric(rows, "false_edit"),
        "missed_edit_rate": metric(rows, "missed_edit"),
        "balanced_utility": metric(rows, "episode_utility_v2_proxy_balanced"),
        "safety_utility": metric(rows, "episode_utility_v2_proxy_safety"),
        "mean_cost": metric(rows, "spent_cost"),
        "mean_tool_calls": metric(rows, "tool_calls"),
    }


def quantiles(values: list[float], count: int) -> list[float]:
    ordered = sorted(set(values))
    if not ordered:
        return []
    if len(ordered) <= count:
        return ordered
    return sorted(
        {
            ordered[round(index * (len(ordered) - 1) / (count - 1))]
            for index in range(count)
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--baseline-name", default="edit_conditioned_proactive_tools.jsonl")
    parser.add_argument(
        "--candidate-name",
        default="qwen3_4b_sft_calibrated_tool_to_belief.jsonl",
    )
    parser.add_argument("--quantiles", type=int, default=101)
    args = parser.parse_args()

    seed_dirs = sorted(
        path
        for path in args.root.glob("seed*")
        if path.is_dir()
        and (path / args.baseline_name).is_file()
        and (path / args.candidate_name).is_file()
    )
    if (
        not seed_dirs
        and (args.root / args.baseline_name).is_file()
        and (args.root / args.candidate_name).is_file()
    ):
        seed_dirs = [args.root]
    if not seed_dirs:
        raise FileNotFoundError(f"no paired seed directories under {args.root}")

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for seed_dir in seed_dirs:
        baseline = load_rows(seed_dir / args.baseline_name)
        candidate = load_rows(seed_dir / args.candidate_name)
        if baseline.keys() != candidate.keys():
            raise ValueError(f"sample mismatch in {seed_dir}")
        pairs.extend((baseline[key], candidate[key]) for key in sorted(baseline))

    baseline_rows = [baseline for baseline, _ in pairs]
    candidate_rows = [candidate for _, candidate in pairs]
    baseline_summary = summarize(baseline_rows)
    candidate_summary = summarize(candidate_rows)
    frontier: list[dict[str, Any]] = []

    for feature in FEATURES:
        values = [
            float(candidate[feature])
            for baseline, candidate in pairs
            if baseline["prediction"] != candidate["prediction"]
        ]
        for threshold in quantiles(values, args.quantiles):
            for direction in ("ge", "le"):
                selected: list[dict[str, Any]] = []
                accepted = 0
                for baseline, candidate in pairs:
                    disagreement = baseline["prediction"] != candidate["prediction"]
                    value = float(candidate[feature])
                    passes = value >= threshold if direction == "ge" else value <= threshold
                    use_candidate = disagreement and passes
                    accepted += int(use_candidate)
                    selected.append(candidate if use_candidate else baseline)
                summary = summarize(selected)
                frontier.append(
                    {
                        "feature": feature,
                        "direction": direction,
                        "threshold": threshold,
                        "accepted_disagreements": accepted,
                        **summary,
                        **{
                            f"delta_{key}": value - baseline_summary[key]
                            for key, value in summary.items()
                        },
                    }
                )

    safe = [
        row
        for row in frontier
        if row["false_edit_rate"] <= baseline_summary["false_edit_rate"] + 1e-12
    ]
    best_safe = max(
        safe,
        key=lambda row: (
            row["balanced_utility"],
            row["terminal_accuracy"],
            -row["missed_edit_rate"],
        ),
        default=None,
    )
    output = {
        "schema_version": "terminal-arbitration-frontier-v1",
        "protocol": {
            "diagnostic_only": True,
            "validation_labels_used_for_oracle_sweep": True,
            "deployable_features_only": True,
            "source_directories": [str(path) for path in seed_dirs],
            "rows": len(pairs),
        },
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "best_matched_false_edit_oracle": best_safe,
        "conclusion": (
            "train a frozen safety arbiter only if the matched-false-edit oracle "
            "retains a positive balanced-utility or missed-edit gain"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    if best_safe is not None:
        (args.output_dir / "frozen_gate.json").write_text(
            json.dumps(
                {
                    "schema_version": "terminal-arbitration-gate-v1",
                    "fit_split": next(iter({row[0]["split"] for row in pairs})),
                    "feature": best_safe["feature"],
                    "direction": best_safe["direction"],
                    "threshold": best_safe["threshold"],
                    "selection_objective": (
                        "maximize balanced utility subject to false edit "
                        "not exceeding the baseline"
                    ),
                    "uses_ground_truth_at_inference": False,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    with (args.output_dir / "frontier.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frontier[0]))
        writer.writeheader()
        writer.writerows(frontier)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
