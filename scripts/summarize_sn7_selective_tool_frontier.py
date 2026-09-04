#!/usr/bin/env python3
"""Summarize the SN7 selective-tool quality/cost/safety frontier."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from summarize_sn7_post_tool_causal_seeds import (
    _hierarchical_bootstrap,
    _paired_rows,
    _read_jsonl,
)

VARIANTS = ("notool", "benefit", "q05", "q15", "q30", "forced")
COMPARISONS = (
    ("benefit", "notool"),
    ("q05", "notool"),
    ("q15", "notool"),
    ("q30", "notool"),
    ("forced", "notool"),
    ("benefit", "q05"),
    ("benefit", "forced"),
    ("q15", "q05"),
    ("q30", "q15"),
    ("q05", "forced"),
    ("q15", "forced"),
    ("q30", "forced"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_dir(
    root: Path,
    variant: str,
    seed: int,
    limit: int,
    *,
    baseline_tag: str,
    frontier_tag: str,
) -> Path:
    if variant == "notool":
        return root / (
            f"sn7_step0_causal_notool_n{limit}_seed{seed}_{baseline_tag}"
        )
    if variant == "forced":
        return root / (
            f"sn7_step0_causal_updated_n{limit}_seed{seed}_{baseline_tag}"
        )
    if variant == "benefit":
        return root / (
            f"sn7_step0_selective_benefit_n{limit}_seed{seed}_benefit_gate_v1"
        )
    return root / (
        f"sn7_step0_selective_updated_{variant}_n{limit}_seed{seed}_{frontier_tag}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seeds", default="20260730,20260731,20260801")
    parser.add_argument("--limit", type=int, default=6369)
    parser.add_argument("--baseline-tag", default="fullval_v1")
    parser.add_argument("--frontier-tag", default="frontier_v1")
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260729)
    args = parser.parse_args()

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    traces: dict[int, dict[str, list[dict[str, Any]]]] = {}
    inputs = []
    per_seed = []
    for seed in seeds:
        traces[seed] = {}
        for variant in VARIANTS:
            root = run_dir(
                args.run_root,
                variant,
                seed,
                args.limit,
                baseline_tag=args.baseline_tag,
                frontier_tag=args.frontier_tag,
            )
            trace_path = root / "edit_utility.jsonl"
            summary_path = root / "summary.json"
            traces[seed][variant] = _read_jsonl(trace_path)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            metrics = summary["policies"]["edit_utility"]["metrics"]
            per_seed.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "terminal_accuracy": metrics["terminal_accuracy"],
                    "false_edit_rate": metrics["false_edit_rate"],
                    "missed_edit_rate": metrics["missed_edit_rate"],
                    "mean_quality_gain": metrics["mean_quality_gain"],
                    "mean_quality_cost_utility": metrics[
                        "mean_quality_cost_utility"
                    ],
                    "mean_tool_calls": metrics["mean_tool_calls"],
                    "tool_call_episode_rate": metrics["tool_call_episode_rate"],
                    "mean_tool_belief_l1_delta": metrics[
                        "mean_tool_belief_l1_delta"
                    ],
                }
            )
            inputs.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "trace": str(trace_path.resolve()),
                    "trace_sha256": sha256(trace_path),
                    "summary": str(summary_path.resolve()),
                    "summary_sha256": sha256(summary_path),
                }
            )

    comparisons = {}
    for left, right in COMPARISONS:
        paired = {
            seed: _paired_rows(traces[seed][left], traces[seed][right])
            for seed in seeds
        }
        comparisons[f"{left}_vs_{right}"] = {
            "per_seed": [
                {
                    "seed": seed,
                    "episodes": len(rows),
                    "tool_episodes": sum(row["tool_episode"] for row in rows),
                    "action_flips": sum(row["action_flip"] for row in rows),
                    "correct_delta_count": int(
                        sum(row["delta"]["terminal_correct"] for row in rows)
                    ),
                    "false_edit_delta_count": int(
                        sum(row["delta"]["false_edit"] for row in rows)
                    ),
                }
                for seed, rows in paired.items()
            ],
            "hierarchical_seed_aoi_bootstrap": _hierarchical_bootstrap(
                paired,
                repetitions=args.bootstrap_repetitions,
                seed=args.bootstrap_seed,
            ),
        }

    output = {
        "schema_version": "sn7-selective-tool-frontier-three-seed-v1",
        "seeds": seeds,
        "variants": list(VARIANTS),
        "per_seed": per_seed,
        "comparisons": comparisons,
        "inputs": inputs,
        "protocol": {
            "gate_calibration_split": "train",
            "validation_used_for_gate_calibration": False,
            "test_assets_read": False,
            "matched_samples_and_budgets": True,
            "hierarchical_bootstrap_unit": "seed_then_aoi",
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "bootstrap_seed": args.bootstrap_seed,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
