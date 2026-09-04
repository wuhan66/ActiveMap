#!/usr/bin/env python3
"""Summarize no-tool, forced, and benefit Step-0 traces across model seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import fmean
from typing import Any

from scripts.summarize_sn7_post_tool_causal_seeds import (
    _hierarchical_bootstrap,
    _paired_rows,
    _read_jsonl,
)

VARIANTS = ("notool", "benefit", "forced")


def _parse_seed_path(value: str) -> tuple[int, Path]:
    seed, separator, path = value.partition("=")
    if not separator or not seed.isdigit() or not path:
        raise argparse.ArgumentTypeError("expected SEED=/path/to/trace.jsonl")
    return int(seed), Path(path)


def _means(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "terminal_accuracy": fmean(float(row["terminal_correct"]) for row in rows),
        "false_edit_rate": fmean(float(row["false_edit"]) for row in rows),
        "missed_edit_rate": fmean(float(row["missed_edit"]) for row in rows),
        "mean_quality_gain": fmean(float(row["quality_gain"]) for row in rows),
        "mean_quality_cost_utility": fmean(
            float(row["quality_cost_utility"]) for row in rows
        ),
        "mean_tool_calls": fmean(float(row["tool_calls"]) for row in rows),
        "tool_call_episode_rate": fmean(
            float(row["tool_calls"]) > 0 for row in rows
        ),
        "mean_tool_belief_l1_delta": fmean(
            float(row["tool_belief_l1_delta"]) for row in rows
        ),
    }


def summarize(
    traces: dict[int, dict[str, list[dict[str, Any]]]],
    *,
    split: str,
    repetitions: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    expected_test_access = split == "test"
    seeds = sorted(traces)
    for seed in seeds:
        if set(traces[seed]) != set(VARIANTS):
            raise ValueError(f"seed {seed} lacks the three required variants")
        support = None
        for variant in VARIANTS:
            rows = traces[seed][variant]
            if not rows:
                raise ValueError(f"empty {variant} trace for seed {seed}")
            if any(row.get("split") != split for row in rows):
                raise ValueError("mixed trace splits")
            if any(
                bool(row.get("test_assets_read")) != expected_test_access
                for row in rows
            ):
                raise ValueError("trace test-access provenance mismatch")
            keys = {str(row["sample_id"]) for row in rows}
            if support is None:
                support = keys
            elif keys != support:
                raise ValueError("variants require identical sample support")

    comparisons = {}
    for left, right in (("benefit", "notool"), ("benefit", "forced")):
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
                paired, repetitions=repetitions, seed=bootstrap_seed
            ),
        }
    return {
        "schema_version": "sn7-selective-tool-frontier-three-seed-v1",
        "seeds": seeds,
        "variants": list(VARIANTS),
        "per_seed": [
            {"seed": seed, "variant": variant, **_means(traces[seed][variant])}
            for seed in seeds
            for variant in VARIANTS
        ],
        "comparisons": comparisons,
        "protocol": {
            "gate_calibration_split": "train",
            "validation_used_for_gate_calibration": False,
            "test_assets_read": expected_test_access,
            "split": split,
            "matched_samples_and_budgets": True,
            "hierarchical_bootstrap_unit": "seed_then_aoi",
            "bootstrap_repetitions": repetitions,
            "bootstrap_seed": bootstrap_seed,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for variant in VARIANTS:
        parser.add_argument(
            f"--{variant}", action="append", type=_parse_seed_path, required=True
        )
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260729)
    parser.add_argument("--frozen-test", action="store_true")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.split == "test":
        if not args.frozen_test:
            raise PermissionError("test summary requires --frozen-test")
        from activemap.frozen_test import assert_frozen_test_access

        assert_frozen_test_access()
    elif args.frozen_test:
        raise ValueError("--frozen-test is valid only for test")
    paths = {
        variant: dict(getattr(args, variant))
        for variant in VARIANTS
    }
    if any(set(values) != set(paths["notool"]) for values in paths.values()):
        raise ValueError("all variants must use identical seeds")
    traces = {
        seed: {
            variant: _read_jsonl(paths[variant][seed])
            for variant in VARIANTS
        }
        for seed in sorted(paths["notool"])
    }
    result = summarize(
        traces,
        split=args.split,
        repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.bootstrap_seed,
    )
    result["inputs"] = [
        {
            "seed": seed,
            "variant": variant,
            "trace": str(paths[variant][seed].resolve()),
            "trace_sha256": hashlib.sha256(
                paths[variant][seed].read_bytes()
            ).hexdigest(),
        }
        for seed in sorted(paths["notool"])
        for variant in VARIANTS
    ]
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["comparisons"], indent=2))


if __name__ == "__main__":
    main()
