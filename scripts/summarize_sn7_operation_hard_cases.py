#!/usr/bin/env python3
"""Audit per-operation causal effects and export SN7 tool hard cases."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from summarize_sn7_post_tool_causal_seeds import (
    METRICS,
    _hierarchical_bootstrap,
    _read_jsonl,
)

OPERATIONS = ("KEEP", "ADD", "DELETE", "RESHAPE")
COMPARISONS = ("benefit", "forced")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    reference = {row["sample_id"]: row for row in right}
    if {row["sample_id"] for row in left} != set(reference):
        raise ValueError("paired traces do not contain identical samples")
    output = []
    for row in left:
        base = reference[row["sample_id"]]
        if row["target_edit"] != base["target_edit"]:
            raise ValueError(f"target mismatch: {row['sample_id']}")
        output.append(
            {
                "sample_id": row["sample_id"],
                "source_episode": row["source_episode"],
                "aoi_id": row["aoi_id"],
                "target_edit": row["target_edit"],
                "baseline_prediction": base["predicted_edit"],
                "intervention_prediction": row["predicted_edit"],
                "action_flip": row["predicted_edit"] != base["predicted_edit"],
                "tool_episode": int(row["tool_calls"]) > 0,
                "corrected": bool(row["terminal_correct"])
                and not bool(base["terminal_correct"]),
                "regressed": not bool(row["terminal_correct"])
                and bool(base["terminal_correct"]),
                "delta": {
                    metric: float(row[metric]) - float(base[metric])
                    for metric in METRICS
                },
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seeds", default="20260730,20260731,20260801")
    parser.add_argument("--limit", type=int, default=6369)
    parser.add_argument("--baseline-tag", default="fullval_v1")
    parser.add_argument("--benefit-tag", default="benefit_gate_v1")
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260729)
    args = parser.parse_args()

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    traces: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    inputs = []
    for seed in seeds:
        paths = {
            "notool": args.run_root
            / f"sn7_step0_causal_notool_n{args.limit}_seed{seed}_{args.baseline_tag}"
            / "edit_utility.jsonl",
            "forced": args.run_root
            / f"sn7_step0_causal_updated_n{args.limit}_seed{seed}_{args.baseline_tag}"
            / "edit_utility.jsonl",
            "benefit": args.run_root
            / f"sn7_step0_selective_benefit_n{args.limit}_seed{seed}_{args.benefit_tag}"
            / "edit_utility.jsonl",
        }
        for variant, path in paths.items():
            traces[seed][variant] = _read_jsonl(path)
            inputs.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "path": str(path.resolve()),
                    "sha256": sha256(path),
                }
            )

    comparisons = {}
    hard_cases = []
    for variant in COMPARISONS:
        paired = {
            seed: pair(traces[seed][variant], traces[seed]["notool"])
            for seed in seeds
        }
        operation_results = {}
        for operation in OPERATIONS:
            subset = {
                seed: [
                    row for row in rows if row["target_edit"] == operation
                ]
                for seed, rows in paired.items()
            }
            operation_results[operation] = {
                "per_seed": [
                    {
                        "seed": seed,
                        "episodes": len(rows),
                        "tool_episodes": sum(row["tool_episode"] for row in rows),
                        "action_flips": sum(row["action_flip"] for row in rows),
                        "corrected": sum(row["corrected"] for row in rows),
                        "regressed": sum(row["regressed"] for row in rows),
                    }
                    for seed, rows in subset.items()
                ],
                "hierarchical_seed_aoi_bootstrap": _hierarchical_bootstrap(
                    subset,
                    repetitions=args.bootstrap_repetitions,
                    seed=args.bootstrap_seed,
                ),
            }
        comparisons[f"{variant}_vs_notool"] = operation_results
        for seed, rows in paired.items():
            for row in rows:
                if row["action_flip"] or row["tool_episode"]:
                    hard_cases.append(
                        {
                            "comparison": f"{variant}_vs_notool",
                            "seed": seed,
                            **{
                                key: value
                                for key, value in row.items()
                                if key != "delta"
                            },
                            "delta": row["delta"],
                        }
                    )

    output = {
        "schema_version": "sn7-operation-hard-case-audit-v1",
        "seeds": seeds,
        "operations": list(OPERATIONS),
        "comparisons": comparisons,
        "hard_cases": hard_cases,
        "inputs": inputs,
        "protocol": {
            "paired_by_sample_id": True,
            "hierarchical_bootstrap_unit": "seed_then_aoi",
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "test_assets_read": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
