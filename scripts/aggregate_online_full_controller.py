#!/usr/bin/env python3
"""Aggregate true sequential controller traces by trajectory horizon.

The evaluator emits one target-free, policy-specific trace for each controller
seed. This script keeps policies paired within a chronological chain, averages
chains inside each AOI, then bootstraps controller seeds followed by AOIs.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_POLICIES = (
    "direct_current_hypothesis",
    "direct_current_hypothesis_safe",
    "active_selective_safe",
    "active_forced_safe",
)
PRIMARY_COMPARISONS = {
    "active_minus_direct": (
        "active_selective_safe",
        "direct_current_hypothesis",
    ),
    "active_minus_direct_safe": (
        "active_selective_safe",
        "direct_current_hypothesis_safe",
    ),
    "active_minus_forced_safe": (
        "active_selective_safe",
        "active_forced_safe",
    ),
    "direct_safe_minus_direct": (
        "direct_current_hypothesis_safe",
        "direct_current_hypothesis",
    ),
}
METRICS = (
    "final_map_quality",
    "mean_map_quality",
    "cumulative_false_writes",
    "cumulative_missed_writes",
    "cumulative_wrong_writes",
    "cumulative_acquisitions",
    "cumulative_tool_calls",
    "cumulative_spent_cost",
    "cumulative_tool_cost",
    "cumulative_commits",
    "mean_safe_rejection_rate",
    "mean_noncanonical_prior_rate",
)


def parse_record(value: str) -> tuple[str, Path]:
    seed, separator, raw_path = value.partition("=")
    if not separator or not seed:
        raise argparse.ArgumentTypeError(
            "record must use SEED=/path/to/online_full_controller_traces.jsonl"
        )
    return seed, Path(raw_path)


def load_trace(
    path: Path, *, policies: tuple[str, ...]
) -> dict[tuple[str, int, str], dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty online controller trace: {path}")
    result: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row_number, row in enumerate(rows, 1):
        if row.get("test_assets_read") is not False or row.get("split") != "val":
            raise ValueError(f"trace must be validation-only: {path}:{row_number}")
        required = {
            "policy",
            "chain_id",
            "step",
            "aoi_id",
            "final_raster_iou",
            "false_edit",
            "missed_edit",
            "wrong_edit",
            "acquisitions",
            "tool_calls",
            "spent_cost",
            "tool_cost",
            "commit_accepted",
            "safe_commit_rejected",
            "prior_input_sha256",
            "canonical_prior_sha256",
        }
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"trace row {row_number} lacks fields: {missing}")
        policy = str(row["policy"])
        if policy not in policies:
            continue
        key = (str(row["chain_id"]), int(row["step"]), policy)
        if key in result:
            raise ValueError(f"duplicate trace identity in {path}: {key}")
        result[key] = row
    if not result:
        raise ValueError(f"trace has no requested policies: {path}")
    support = {(chain_id, step) for chain_id, step, _ in result}
    for chain_id, step in support:
        observed = {
            policy
            for candidate_chain, candidate_step, policy in result
            if (candidate_chain, candidate_step) == (chain_id, step)
        }
        missing = sorted(set(policies) - observed)
        if missing:
            raise ValueError(
                "incomplete policy support in "
                f"{path}: chain={chain_id} step={step} missing={missing}"
            )
    return result


def _chain_prefix_metrics(rows: list[dict[str, Any]], horizon: int) -> dict[str, float]:
    prefix = rows[:horizon]
    if len(prefix) != horizon:
        raise ValueError("trajectory does not cover requested horizon")
    return {
        "final_map_quality": float(prefix[-1]["final_raster_iou"]),
        "mean_map_quality": float(np.mean([row["final_raster_iou"] for row in prefix])),
        "cumulative_false_writes": float(sum(bool(row["false_edit"]) for row in prefix)),
        "cumulative_missed_writes": float(sum(bool(row["missed_edit"]) for row in prefix)),
        "cumulative_wrong_writes": float(sum(bool(row["wrong_edit"]) for row in prefix)),
        "cumulative_acquisitions": float(sum(float(row["acquisitions"]) for row in prefix)),
        "cumulative_tool_calls": float(sum(float(row["tool_calls"]) for row in prefix)),
        "cumulative_spent_cost": float(sum(float(row["spent_cost"]) for row in prefix)),
        "cumulative_tool_cost": float(sum(float(row["tool_cost"]) for row in prefix)),
        "cumulative_commits": float(sum(bool(row["commit_accepted"]) for row in prefix)),
        "mean_safe_rejection_rate": float(
            np.mean([bool(row["safe_commit_rejected"]) for row in prefix])
        ),
        "mean_noncanonical_prior_rate": float(
            np.mean(
                [
                    row["prior_input_sha256"] != row["canonical_prior_sha256"]
                    for row in prefix
                ]
            )
        ),
    }


def trace_to_prefixes(
    trace: dict[tuple[str, int, str], dict[str, Any]], *, policies: tuple[str, ...]
) -> tuple[dict[int, dict[str, dict[str, dict[str, float]]]], dict[str, str]]:
    """Return horizon -> chain -> policy prefix metrics and chain AOI labels."""

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    chain_aois: dict[str, str] = {}
    for (chain_id, _step, policy), row in trace.items():
        grouped[chain_id][policy].append(row)
        aoi_id = str(row["aoi_id"])
        previous = chain_aois.setdefault(chain_id, aoi_id)
        if previous != aoi_id:
            raise ValueError(f"chain {chain_id} spans AOIs")

    horizons: dict[int, dict[str, dict[str, dict[str, float]]]] = defaultdict(dict)
    for chain_id, by_policy in grouped.items():
        lengths = set()
        for policy in policies:
            rows = sorted(by_policy[policy], key=lambda row: int(row["step"]))
            expected_steps = list(range(len(rows)))
            if [int(row["step"]) for row in rows] != expected_steps:
                raise ValueError(f"chain {chain_id} policy {policy} has non-contiguous steps")
            by_policy[policy] = rows
            lengths.add(len(rows))
        if len(lengths) != 1:
            raise ValueError(f"chain {chain_id} policies have unequal lengths")
        for horizon in range(1, next(iter(lengths)) + 1):
            horizons[horizon][chain_id] = {
                policy: _chain_prefix_metrics(by_policy[policy], horizon)
                for policy in policies
            }
    return dict(horizons), chain_aois


def _mean_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    return {metric: float(np.mean([item[metric] for item in items])) for metric in METRICS}


def _bootstrap_comparison(
    differences: np.ndarray, *, repetitions: int, rng: np.random.Generator
) -> dict[str, dict[str, float]]:
    """Bootstrap a [seed, AOI, metric] paired-difference tensor."""

    seed_count, aoi_count, _ = differences.shape
    observed = differences.mean(axis=(0, 1))
    sampled_seeds = rng.integers(0, seed_count, size=(repetitions, seed_count))
    sampled_aois = rng.integers(0, aoi_count, size=(repetitions, seed_count, aoi_count))
    seed_estimates = np.empty((repetitions, seed_count, len(METRICS)), dtype=np.float64)
    row_indices = np.arange(repetitions)[:, None]
    for slot in range(seed_count):
        selected = differences[sampled_seeds[:, slot]]
        seed_estimates[:, slot, :] = selected[row_indices, sampled_aois[:, slot], :].mean(axis=1)
    draws = seed_estimates.mean(axis=1)
    return {
        metric: {
            "observed_delta": float(observed[index]),
            "ci95_low": float(np.quantile(draws[:, index], 0.025)),
            "ci95_high": float(np.quantile(draws[:, index], 0.975)),
        }
        for index, metric in enumerate(METRICS)
    }


def aggregate(
    traces: dict[str, dict[tuple[str, int, str], dict[str, Any]]],
    *,
    policies: tuple[str, ...],
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    if len(traces) < 2:
        raise ValueError("sequential aggregation requires at least two controller seeds")
    prefixes_by_seed: dict[str, dict[int, dict[str, dict[str, dict[str, float]]]]] = {}
    aoi_by_seed: dict[str, dict[str, str]] = {}
    for label, trace in traces.items():
        prefixes_by_seed[label], aoi_by_seed[label] = trace_to_prefixes(trace, policies=policies)

    seed_labels = sorted(traces)
    reference_horizons = set(prefixes_by_seed[seed_labels[0]])
    for label in seed_labels[1:]:
        if set(prefixes_by_seed[label]) != reference_horizons:
            raise ValueError("controller seeds do not share the same horizon support")
    reference_aois = set(aoi_by_seed[seed_labels[0]].values())
    for label in seed_labels[1:]:
        if set(aoi_by_seed[label].values()) != reference_aois:
            raise ValueError("controller seeds do not share the same AOI support")

    rng = np.random.default_rng(seed)
    horizon_results: dict[str, Any] = {}
    for horizon in sorted(reference_horizons):
        complete_chains = {
            label: prefixes_by_seed[label][horizon]
            for label in seed_labels
        }
        reference_chains = set(complete_chains[seed_labels[0]])
        if any(set(complete_chains[label]) != reference_chains for label in seed_labels[1:]):
            raise ValueError("controller seeds do not share the same chain support")
        aois = sorted(reference_aois)
        if any(
            not any(aoi_by_seed[label][chain_id] == aoi for chain_id in complete_chains[label])
            for label in seed_labels
            for aoi in aois
        ):
            continue

        policy_seed_summary: dict[str, dict[str, dict[str, float]]] = {}
        policy_aoi = np.empty((len(seed_labels), len(policies), len(aois), len(METRICS)))
        for seed_index, label in enumerate(seed_labels):
            chains = complete_chains[label]
            policy_seed_summary[label] = {}
            for policy_index, policy in enumerate(policies):
                rows = [values[policy] for values in chains.values()]
                policy_seed_summary[label][policy] = _mean_metrics(rows)
                for aoi_index, aoi in enumerate(aois):
                    aoi_rows = [
                        values[policy]
                        for chain_id, values in chains.items()
                        if aoi_by_seed[label][chain_id] == aoi
                    ]
                    policy_aoi[seed_index, policy_index, aoi_index, :] = [
                        _mean_metrics(aoi_rows)[metric] for metric in METRICS
                    ]

        policy_summary = {
            policy: {
                metric: {
                    "mean": float(np.mean([
                        policy_seed_summary[label][policy][metric]
                        for label in seed_labels
                    ])),
                    "seed_std": float(
                        np.std(
                            [policy_seed_summary[label][policy][metric] for label in seed_labels],
                            ddof=1,
                        )
                    ),
                }
                for metric in METRICS
            }
            for policy in policies
        }
        policy_index = {policy: index for index, policy in enumerate(policies)}
        comparisons = {
            name: _bootstrap_comparison(
                policy_aoi[:, policy_index[left], :, :] - policy_aoi[:, policy_index[right], :, :],
                repetitions=repetitions,
                rng=rng,
            )
            for name, (left, right) in PRIMARY_COMPARISONS.items()
        }
        horizon_results[str(horizon)] = {
            "horizon": horizon,
            "chain_count": len(complete_chains[seed_labels[0]]),
            "aoi_count": len(aois),
            "policy_metrics": policy_summary,
            "paired_comparisons": comparisons,
        }

    if not horizon_results:
        raise ValueError("no horizon is supported by every seed and AOI")
    return {
        "schema_version": "activemap-online-full-controller-horizon-aggregate-v1",
        "split": "val",
        "test_assets_read": False,
        "policies": list(policies),
        "seed_count": len(seed_labels),
        "horizons": horizon_results,
        "bootstrap": {
            "unit": "controller seed then AOI with paired chains",
            "repetitions": repetitions,
            "seed": seed,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--record", action="append", type=parse_record, required=True)
    parser.add_argument("--policy", action="append", choices=REQUIRED_POLICIES)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    policies = tuple(args.policy or REQUIRED_POLICIES)
    if tuple(policies) != REQUIRED_POLICIES:
        raise ValueError("the four-policy sequential protocol requires the registered policy order")
    records = dict(args.record)
    if len(records) != len(args.record):
        raise ValueError("duplicate controller seed")
    result = aggregate(
        {label: load_trace(path, policies=policies) for label, path in records.items()},
        policies=policies,
        repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    result["sources"] = {label: str(path.resolve()) for label, path in records.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
