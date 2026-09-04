#!/usr/bin/env python3
"""Causally separate evidence selection from Safe Commit on matched writebacks.

This is an analysis-only 2x2 protocol. It never changes selection traces,
updater outputs, or evidence costs. The same fixed observable gate is applied
to each evidence policy, so the selection and terminal-commit factors can be
compared without retuning one branch on validation outcomes.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from activemap.evaluation.episode_utility import score_episode_profiles
from activemap.models import EditOperation


METRICS = (
    "map_quality_after",
    "map_quality_gain",
    "balanced_utility",
    "false_edit_rate",
    "missed_edit_rate",
    "wrong_edit_rate",
    "commit_rate",
    "tool_call_rate",
    "mean_cost",
)

ROW_FIELDS = {
    "map_quality_after": "map_quality_after",
    "map_quality_gain": "map_quality_gain",
    "balanced_utility": "balanced_utility",
    "false_edit_rate": "false_edit",
    "missed_edit_rate": "missed_edit",
    "wrong_edit_rate": "wrong_edit",
    "commit_rate": "commit",
    "tool_call_rate": "tool_call",
    "mean_cost": "spent_cost",
}


@dataclass(frozen=True)
class Gate:
    confidence_threshold: float
    replay_iou_threshold: float
    require_topology: bool = True

    def accepts(self, row: dict[str, Any]) -> bool:
        if not bool(row["writeback_changed"]):
            return True
        return bool(
            float(row["fused_confidence"]) >= self.confidence_threshold
            and float(row["vector_replay_iou"]) >= self.replay_iou_threshold
            and (
                not self.require_topology
                or bool(row["vector_delta_topology_valid"])
            )
        )


def parse_record(value: str) -> tuple[str, str, Path]:
    head, separator, location = value.partition("=")
    seed, divider, policy = head.partition(":")
    if not separator or not divider or not seed or not policy:
        raise argparse.ArgumentTypeError("record must be SEED:POLICY=/path/to/writeback.jsonl")
    return seed, policy, Path(location)


def terminal_operation(value: str) -> EditOperation:
    if value == "REJECT":
        return EditOperation.KEEP
    prefix = "COMMIT:"
    if not value.startswith(prefix):
        raise ValueError(f"unsupported terminal action {value!r}")
    return EditOperation(value[len(prefix) :])


def _errors(target: EditOperation, executed: EditOperation) -> tuple[bool, bool, bool]:
    false_edit = target == EditOperation.KEEP and executed != EditOperation.KEEP
    missed_edit = target != EditOperation.KEEP and executed == EditOperation.KEEP
    wrong_edit = (
        target != EditOperation.KEEP
        and executed != EditOperation.KEEP
        and executed != target
    )
    return false_edit, missed_edit, wrong_edit


def load_rows(path: Path) -> dict[tuple[str, float], dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty writeback: {path}")
    required = {
        "task_id", "aoi_id", "budget", "target", "effective_operation",
        "writeback_changed", "fused_confidence", "vector_replay_iou",
        "vector_delta_topology_valid", "raster_iou", "prior_raster_iou",
        "spent_cost", "semantic_tool_called", "split", "test_assets_read",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"writeback schema missing {missing}: {path}")
    result = {}
    for row in rows:
        if row["split"] != "val" or row["test_assets_read"] is not False:
            raise ValueError(f"2x2 audit accepts validation-only rows: {path}")
        key = (str(row["task_id"]), float(row["budget"]))
        if key in result:
            raise ValueError(f"duplicate task-budget identity in {path}: {key}")
        # Historical executable writebacks predate the generic map-quality aliases.
        result[key] = {
            **row,
            "map_quality_before": float(row["prior_raster_iou"]),
            "map_quality_after": float(row["raster_iou"]),
        }
    return result


def apply_gate(row: dict[str, Any], gate: Gate) -> dict[str, Any]:
    accepted = gate.accepts(row)
    effective = EditOperation(str(row["effective_operation"])) if accepted else EditOperation.KEEP
    target = terminal_operation(str(row["target"]))
    false_edit, missed_edit, wrong_edit = _errors(target, effective)
    after = float(row["map_quality_after"] if accepted else row["map_quality_before"])
    before = float(row["map_quality_before"])
    utility = score_episode_profiles(
        final_map_quality=after,
        prior_map_quality=before,
        spent_cost=float(row["spent_cost"]),
        budget=float(row["budget"]),
        false_edit=false_edit,
        missed_edit=missed_edit,
        wrong_edit=wrong_edit,
        topology_valid=bool(row["vector_delta_topology_valid"]),
    )
    return {
        "task_id": str(row["task_id"]),
        "aoi_id": str(row["aoi_id"]),
        "budget": float(row["budget"]),
        "target": target.value,
        "gate_accepted": accepted,
        "executed_operation": effective.value,
        "map_quality_before": before,
        "map_quality_after": after,
        "map_quality_gain": after - before,
        "balanced_utility": float(utility["balanced"]["value"]),
        "false_edit": false_edit,
        "missed_edit": missed_edit,
        "wrong_edit": wrong_edit,
        "commit": bool(row["writeback_changed"]) and accepted,
        "tool_call": bool(row["semantic_tool_called"]),
        "spent_cost": float(row["spent_cost"]),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot summarize empty cell")
    return {
        "map_quality_after": float(np.mean([row["map_quality_after"] for row in rows])),
        "map_quality_gain": float(np.mean([row["map_quality_gain"] for row in rows])),
        "balanced_utility": float(np.mean([row["balanced_utility"] for row in rows])),
        "false_edit_rate": float(np.mean([row["false_edit"] for row in rows])),
        "missed_edit_rate": float(np.mean([row["missed_edit"] for row in rows])),
        "wrong_edit_rate": float(np.mean([row["wrong_edit"] for row in rows])),
        "commit_rate": float(np.mean([row["commit"] for row in rows])),
        "tool_call_rate": float(np.mean([row["tool_call"] for row in rows])),
        "mean_cost": float(np.mean([row["spent_cost"] for row in rows])),
    }


def _paired_delta(
    left: list[dict[str, Any]], right: list[dict[str, Any]], name: str
) -> float:
    left_index = {(row["task_id"], row["budget"]): row for row in left}
    right_index = {(row["task_id"], row["budget"]): row for row in right}
    if left_index.keys() != right_index.keys():
        raise ValueError("comparison cells must be task-budget matched")
    field = ROW_FIELDS[name]
    return float(np.mean([left_index[key][field] - right_index[key][field] for key in left_index]))


def analyze(
    inputs: dict[str, dict[str, dict[tuple[str, float], dict[str, Any]]]],
    *,
    no_extra_policy: str,
    learned_policy: str,
    gate: Gate,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    seeds = sorted(inputs)
    if len(seeds) < 2:
        raise ValueError("2x2 audit requires at least two independently trained seeds")
    expected_policies = {no_extra_policy, learned_policy}
    if any(set(by_policy) != expected_policies for by_policy in inputs.values()):
        raise ValueError("every seed must contain exactly the two requested evidence policies")
    support = None
    for by_policy in inputs.values():
        first = next(iter(by_policy.values()))
        if any(rows.keys() != first.keys() for rows in by_policy.values()):
            raise ValueError("evidence policies do not have identical task-budget support")
        if support is None:
            support = set(first)
        elif support != set(first):
            raise ValueError("model seeds do not have identical task-budget support")
    assert support is not None

    cells: dict[str, dict[str, list[dict[str, Any]]]] = {}
    per_seed: dict[str, dict[str, dict[str, float]]] = {}
    for model_seed, by_policy in inputs.items():
        cells[model_seed] = {}
        per_seed[model_seed] = {}
        for policy, indexed in by_policy.items():
            base = list(indexed.values())
            safe = [apply_gate(row, gate) for row in base]
            always = [apply_gate(row, Gate(0.0, 0.0, False)) for row in base]
            for commit, rows in (("always_commit", always), ("safe_commit", safe)):
                label = f"{policy}__{commit}"
                cells[model_seed][label] = rows
                per_seed[model_seed][label] = summarize(rows)

    cell_labels = sorted(next(iter(cells.values())))
    aggregate = {
        label: {
            metric: {
                "mean": float(np.mean([per_seed[s][label][metric] for s in seeds])),
                "seed_std": float(np.std([per_seed[s][label][metric] for s in seeds], ddof=1)),
            }
            for metric in METRICS
        }
        for label in cell_labels
    }

    comparisons = {
        "selection_gain_always_commit": (f"{learned_policy}__always_commit", f"{no_extra_policy}__always_commit"),
        "selection_gain_safe_commit": (f"{learned_policy}__safe_commit", f"{no_extra_policy}__safe_commit"),
        "safe_commit_gain_no_extra": (f"{no_extra_policy}__safe_commit", f"{no_extra_policy}__always_commit"),
        "safe_commit_gain_learned": (f"{learned_policy}__safe_commit", f"{learned_policy}__always_commit"),
    }
    observed = {
        name: {
            metric: float(np.mean([_paired_delta(cells[s][left], cells[s][right], metric) for s in seeds]))
            for metric in METRICS
        }
        for name, (left, right) in comparisons.items()
    }

    groups = sorted({str(row["aoi_id"]) for row in next(iter(cells.values()))[cell_labels[0]]})
    group_index = {aoi: index for index, aoi in enumerate(groups)}
    group_count = len(groups)
    metric_index = {name: index for index, name in enumerate(METRICS)}
    comparison_labels = list(comparisons)
    # One AOI contributes its task-weighted mean delta. We keep the count so
    # a bootstrap draw preserves the original task weighting within every AOI.
    group_sums = np.zeros((len(seeds), len(comparison_labels), group_count, len(METRICS)))
    group_counts = np.zeros((len(seeds), len(comparison_labels), group_count))
    for seed_index, model_seed in enumerate(seeds):
        for comparison_index, name in enumerate(comparison_labels):
            left, right = comparisons[name]
            left_index = {(row["task_id"], row["budget"]): row for row in cells[model_seed][left]}
            right_index = {(row["task_id"], row["budget"]): row for row in cells[model_seed][right]}
            for key, left_row in left_index.items():
                right_row = right_index[key]
                index = group_index[str(left_row["aoi_id"])]
                group_counts[seed_index, comparison_index, index] += 1.0
                for metric, field in ROW_FIELDS.items():
                    group_sums[
                        seed_index, comparison_index, index, metric_index[metric]
                    ] += left_row[field] - right_row[field]
    rng = np.random.default_rng(seed)
    sampled_aois = rng.integers(0, group_count, size=(repetitions, group_count))
    seed_draws = np.empty(
        (len(seeds), repetitions, len(comparison_labels), len(METRICS)), dtype=np.float64
    )
    for seed_index in range(len(seeds)):
        for comparison_index in range(len(comparison_labels)):
            counts = group_counts[seed_index, comparison_index][sampled_aois].sum(axis=1)
            if np.any(counts == 0):
                raise ValueError("an AOI bootstrap draw has no paired support")
            sums = group_sums[seed_index, comparison_index][sampled_aois].sum(axis=1)
            seed_draws[seed_index, :, comparison_index, :] = sums / counts[:, None]
    sampled_seeds = rng.integers(0, len(seeds), size=(repetitions, len(seeds)))
    model_seed_draws = np.moveaxis(seed_draws, 0, 1)
    hierarchical_draws = model_seed_draws[
        np.arange(repetitions)[:, None], sampled_seeds
    ].mean(axis=1)
    for name in comparisons:
        for metric in METRICS:
            comparison_index = comparison_labels.index(name)
            values = hierarchical_draws[:, comparison_index, metric_index[metric]]
            aggregate_interval = {
                "observed_delta": observed[name][metric],
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
            }
            # Store intervals separately to keep pair definitions readable in provenance.
            observed[name][metric] = aggregate_interval

    return {
        "schema_version": "activemap-selection-safe-commit-2x2-v1",
        "split": "val",
        "test_assets_read": False,
        "protocol": {
            "evidence_policies": {"no_extra": no_extra_policy, "learned": learned_policy},
            "commit_policies": ["always_commit", "safe_commit"],
            "gate": {
                "confidence_threshold": gate.confidence_threshold,
                "replay_iou_threshold": gate.replay_iou_threshold,
                "require_topology": gate.require_topology,
                "tuning": "fixed before this matched 2x2 aggregation",
            },
            "selection_traces_and_costs": "unchanged from supplied writebacks",
            "bootstrap": "hierarchical model-seed then AOI paired resampling",
        },
        "seed_count": len(seeds),
        "aoi_count": len(groups),
        "task_budget_count": len(support),
        "per_seed": per_seed,
        "cells": aggregate,
        "paired_comparisons": observed,
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Selection x Safe Commit 2x2 Audit",
        "",
        "| Evidence policy | Commit policy | Map quality | Gain | False edit | Missed edit | Commit rate | Tool-call rate | Cost |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, metrics in result["cells"].items():
        evidence, commit = label.split("__", 1)
        lines.append(
            "| {} | {} | {:.6f} | {:+.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |".format(
                evidence,
                commit,
                metrics["map_quality_after"]["mean"],
                metrics["map_quality_gain"]["mean"],
                metrics["false_edit_rate"]["mean"],
                metrics["missed_edit_rate"]["mean"],
                metrics["commit_rate"]["mean"],
                metrics["tool_call_rate"]["mean"],
                metrics["mean_cost"]["mean"],
            )
        )
    lines.extend(["", "Paired deltas use hierarchical seed then AOI resampling.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--record", action="append", type=parse_record, required=True)
    parser.add_argument("--no-extra-policy", required=True)
    parser.add_argument("--learned-policy", required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument("--replay-iou-threshold", type=float, default=0.99)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    inputs: dict[str, dict[str, dict[tuple[str, float], dict[str, Any]]]] = defaultdict(dict)
    for model_seed, policy, path in args.record:
        if policy in inputs[model_seed]:
            raise ValueError(f"duplicate seed-policy input: {model_seed}:{policy}")
        inputs[model_seed][policy] = load_rows(path)
    result = analyze(
        dict(inputs),
        no_extra_policy=args.no_extra_policy,
        learned_policy=args.learned_policy,
        gate=Gate(args.confidence_threshold, args.replay_iou_threshold),
        repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "table.md").write_text(_markdown(result), encoding="utf-8")
    with (args.output_dir / "cells.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cell", *METRICS])
        for label, values in result["cells"].items():
            writer.writerow([label, *[values[name]["mean"] for name in METRICS]])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
