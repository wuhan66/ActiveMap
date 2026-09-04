#!/usr/bin/env python3
"""Aggregate the registered V5 direct/selection x commit 2x2 evaluation.

Every statistic is paired by source episode and budget. Confidence intervals
resample model seeds first and AOIs inside each sampled seed, so they propagate
both independent updater/selector training variation and geographic variation.
The optional, preregistered forced-acquisition control is evaluated only from
real writebacks. A missing control remains an unmet promotion condition rather
than being inferred from cached utilities.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from activemap.evaluation.episode_utility import score_episode_profiles
from activemap.models import EditOperation


FACTORIAL_POLICIES = (
    "direct_commit",
    "direct_safe_commit",
    "selected_commit",
    "selected_safe_commit",
)
FORCED_ACQUISITION_POLICY = "forced_safe_commit"
# Kept as the public name for the four primary factorial cells.
POLICIES = FACTORIAL_POLICIES
ALLOWED_POLICIES = FACTORIAL_POLICIES + (FORCED_ACQUISITION_POLICY,)
OPERATIONS = ("ADD", "DELETE", "RESHAPE")
METRICS = (
    "final_map_quality",
    "map_quality_gain",
    "balanced_utility",
    "false_edit_rate",
    "missed_edit_rate",
    "wrong_edit_rate",
    "commit_rate",
    "additional_evidence_rate",
    "additional_cost",
)


def parse_record(value: str) -> tuple[int, str, Path]:
    seed_policy, separator, raw_path = value.partition("=")
    raw_seed, colon, policy = seed_policy.partition(":")
    if not separator or not colon or policy not in ALLOWED_POLICIES or not raw_path:
        raise argparse.ArgumentTypeError(
            "record must be SEED:POLICY=WRITEBACK_JSONL, where POLICY is one of "
            + ", ".join(ALLOWED_POLICIES)
        )
    return int(raw_seed), policy, Path(raw_path)


def bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def target_operation(row: dict[str, Any]) -> EditOperation:
    target = str(row["target"])
    if target == "REJECT":
        return EditOperation.KEEP
    if not target.startswith("COMMIT:"):
        raise ValueError(f"unsupported terminal target {target!r}")
    return EditOperation(target.removeprefix("COMMIT:"))


def _error_flags(row: dict[str, Any]) -> tuple[bool, bool, bool]:
    names = ("false_edit", "missed_edit", "wrong_edit")
    if all(name in row for name in names):
        return tuple(bool_value(row[name]) for name in names)  # type: ignore[return-value]
    target = target_operation(row)
    executed = EditOperation(str(row["effective_operation"]))
    return (
        target == EditOperation.KEEP and executed != EditOperation.KEEP,
        target != EditOperation.KEEP and executed == EditOperation.KEEP,
        target != EditOperation.KEEP
        and executed != EditOperation.KEEP
        and executed != target,
    )


def balanced_utility(row: dict[str, Any]) -> float:
    value = row.get("episode_utility_v2_balanced")
    if value is not None:
        return float(value)
    profile = row.get("episode_utility_v2")
    if isinstance(profile, dict):
        balanced = profile.get("balanced")
        if isinstance(balanced, dict) and "value" in balanced:
            return float(balanced["value"])
    false_edit, missed_edit, wrong_edit = _error_flags(row)
    return float(
        score_episode_profiles(
            final_map_quality=float(row["raster_iou"]),
            prior_map_quality=float(row["prior_raster_iou"]),
            spent_cost=float(row["spent_cost"]),
            budget=float(row["budget"]),
            false_edit=false_edit,
            missed_edit=missed_edit,
            wrong_edit=wrong_edit,
            topology_valid=bool_value(row["vector_delta_topology_valid"]),
        )["balanced"]["value"]
    )


def row_metrics(row: dict[str, Any]) -> dict[str, float]:
    false_edit, missed_edit, wrong_edit = _error_flags(row)
    final = float(row["raster_iou"])
    prior = float(row["prior_raster_iou"])
    return {
        "final_map_quality": final,
        "map_quality_gain": final - prior,
        "balanced_utility": balanced_utility(row),
        "false_edit_rate": float(false_edit),
        "missed_edit_rate": float(missed_edit),
        "wrong_edit_rate": float(wrong_edit),
        "commit_rate": float(bool_value(row["writeback_changed"])),
        "additional_evidence_rate": float(bool_value(row.get("semantic_tool_called", False))),
        "additional_cost": float(row["spent_cost"]),
    }


def load_rows(path: Path) -> dict[tuple[str, float], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: dict[tuple[str, float], dict[str, Any]] = {}
    required = {
        "task_id",
        "aoi_id",
        "budget",
        "target",
        "split",
        "test_assets_read",
        "raster_iou",
        "prior_raster_iou",
        "spent_cost",
        "writeback_changed",
        "effective_operation",
        "vector_delta_topology_valid",
    }
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"{path}:{line_number} missing fields {missing}")
        if row["split"] != "val" or row["test_assets_read"] is not False:
            raise ValueError(f"{path}:{line_number} is not validation-only evidence")
        key = (str(row["task_id"]), float(row["budget"]))
        if key in rows:
            raise ValueError(f"duplicate task-budget key in {path}: {key}")
        rows[key] = row
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def _mean(values: list[dict[str, float]]) -> dict[str, float]:
    return {name: float(np.mean([value[name] for value in values])) for name in METRICS}


def _validate_records(
    records: dict[int, dict[str, Path]],
) -> tuple[
    dict[int, dict[str, dict[tuple[str, float], dict[str, Any]]]], tuple[str, ...]
]:
    if sorted(records) != [20260817, 20260818, 20260819]:
        raise ValueError("V5 factorial aggregation requires exactly seeds 20260817/18/19")
    loaded: dict[int, dict[str, dict[tuple[str, float], dict[str, Any]]]] = {}
    canonical_keys: set[tuple[str, float]] | None = None
    canonical_metadata: dict[tuple[str, float], tuple[str, str]] | None = None
    present_policies: tuple[str, ...] | None = None
    for seed, by_policy in sorted(records.items()):
        actual_policies = set(by_policy)
        if not set(FACTORIAL_POLICIES).issubset(actual_policies) or not actual_policies.issubset(
            set(ALLOWED_POLICIES)
        ):
            raise ValueError(
                f"seed {seed} must contain {FACTORIAL_POLICIES} and may additionally contain "
                f"{FORCED_ACQUISITION_POLICY}"
            )
        current_policies = tuple(policy for policy in ALLOWED_POLICIES if policy in actual_policies)
        if present_policies is None:
            present_policies = current_policies
        elif current_policies != present_policies:
            raise ValueError("all seeds must contain the same registered V5 policies")
        loaded[seed] = {policy: load_rows(path) for policy, path in by_policy.items()}
        reference = loaded[seed][FACTORIAL_POLICIES[0]]
        for policy, rows in loaded[seed].items():
            if rows.keys() != reference.keys():
                raise ValueError(f"seed {seed} {policy} lacks paired support")
            for key in reference:
                left, right = reference[key], rows[key]
                if (str(left["aoi_id"]), str(left["target"])) != (
                    str(right["aoi_id"]),
                    str(right["target"]),
                ):
                    raise ValueError(f"seed {seed} {policy} has mismatched provenance at {key}")
        metadata = {
            key: (str(row["aoi_id"]), str(row["target"])) for key, row in reference.items()
        }
        if canonical_keys is None:
            canonical_keys, canonical_metadata = set(reference), metadata
        elif set(reference) != canonical_keys or metadata != canonical_metadata:
            raise ValueError(f"seed {seed} does not share the registered validation population")
    if present_policies is None:
        raise ValueError("no V5 policy records were supplied")
    return loaded, present_policies


def _keys_for_operation(
    rows: dict[tuple[str, float], dict[str, Any]], operation: str | None
) -> list[tuple[str, float]]:
    if operation is None:
        return sorted(rows)
    return sorted(
        key for key, row in rows.items() if target_operation(row).value == operation
    )


def _seed_aoi_values(
    loaded: dict[int, dict[str, dict[tuple[str, float], dict[str, Any]]]],
    *,
    baseline: str,
    candidate: str,
    operation: str | None,
) -> tuple[dict[int, dict[str, dict[str, float]]], dict[int, int]]:
    result: dict[int, dict[str, dict[str, float]]] = {}
    counts: dict[int, int] = {}
    for seed, by_policy in sorted(loaded.items()):
        reference = by_policy[baseline]
        grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
        for key in _keys_for_operation(reference, operation):
            left = row_metrics(reference[key])
            right = row_metrics(by_policy[candidate][key])
            grouped[str(reference[key]["aoi_id"])].append(
                {name: right[name] - left[name] for name in METRICS}
            )
        if not grouped:
            raise ValueError(f"seed {seed} has no {operation or 'all'} rows")
        result[seed] = {aoi: _mean(values) for aoi, values in grouped.items()}
        counts[seed] = sum(len(values) for values in grouped.values())
    return result, counts


def contrast(
    loaded: dict[int, dict[str, dict[tuple[str, float], dict[str, Any]]]],
    *,
    baseline: str,
    candidate: str,
    operation: str | None,
    repetitions: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    values, row_counts = _seed_aoi_values(
        loaded, baseline=baseline, candidate=candidate, operation=operation
    )
    seed_means = {seed: _mean(list(by_aoi.values())) for seed, by_aoi in values.items()}
    observed = _mean(list(seed_means.values()))
    seeds = sorted(values)
    rng = np.random.default_rng(bootstrap_seed)
    draws = np.empty((repetitions, len(METRICS)), dtype=np.float64)
    for draw_index in range(repetitions):
        seed_values = []
        for seed in rng.choice(seeds, size=len(seeds), replace=True):
            by_aoi = values[int(seed)]
            aoi_ids = sorted(by_aoi)
            sampled = rng.choice(aoi_ids, size=len(aoi_ids), replace=True)
            seed_values.append(_mean([by_aoi[str(aoi)] for aoi in sampled]))
        draw = _mean(seed_values)
        draws[draw_index] = [draw[name] for name in METRICS]
    intervals = {
        name: {
            "delta": observed[name],
            "ci95_low": float(np.quantile(draws[:, index], 0.025)),
            "ci95_high": float(np.quantile(draws[:, index], 0.975)),
        }
        for index, name in enumerate(METRICS)
    }
    return {
        "baseline": baseline,
        "candidate": candidate,
        "operation": operation or "ALL",
        "row_count_per_seed": row_counts,
        "aoi_count_per_seed": {seed: len(by_aoi) for seed, by_aoi in values.items()},
        "per_seed_aoi_macro_delta": seed_means,
        "paired_delta": intervals,
    }


def policy_metrics(
    loaded: dict[int, dict[str, dict[tuple[str, float], dict[str, Any]]]],
    policies: tuple[str, ...],
) -> dict[str, Any]:
    result = {}
    for policy in policies:
        by_seed = {}
        for seed, policies in sorted(loaded.items()):
            grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
            for row in policies[policy].values():
                grouped[str(row["aoi_id"])].append(row_metrics(row))
            by_seed[seed] = _mean([_mean(rows) for rows in grouped.values()])
        result[policy] = {
            "per_seed_aoi_macro": by_seed,
            "three_seed_aoi_macro": _mean(list(by_seed.values())),
        }
    return result


def promotion(
    selection: dict[str, Any],
    safe_commit: dict[str, Any],
    forced_acquisition_cost: dict[str, Any] | None,
) -> dict[str, Any]:
    selection_delta = selection["paired_delta"]
    safe_delta = safe_commit["paired_delta"]
    core = {
        "selection_final_map_quality_lower_positive": selection_delta["final_map_quality"][
            "ci95_low"
        ] > 0.0,
        "selection_false_edit_upper_nonpositive": selection_delta["false_edit_rate"][
            "ci95_high"
        ] <= 0.0,
        "selection_missed_edit_upper_nonpositive": selection_delta["missed_edit_rate"][
            "ci95_high"
        ] <= 0.0,
        "safe_commit_false_edit_upper_nonpositive": safe_delta["false_edit_rate"][
            "ci95_high"
        ] <= 0.0,
        "safe_commit_final_map_quality_lower_nonnegative": safe_delta["final_map_quality"][
            "ci95_low"
        ] >= 0.0,
    }
    if forced_acquisition_cost is None:
        return {
            "checks": core,
            "forced_acquisition_cost_control": "not_evaluated",
            "eligible_for_extension_claim": False,
            "reason": (
                "The registered forced-acquisition cost control is not part of this four-cell "
                "writeback matrix. Report the 2x2 causal contrasts, but do not promote a full "
                "quality-cost claim from this file alone."
            ),
        }

    forced_delta = forced_acquisition_cost["paired_delta"]
    core["selected_cost_upper_strictly_below_forced"] = (
        forced_delta["additional_cost"]["ci95_high"] < 0.0
    )
    return {
        "checks": core,
        "forced_acquisition_cost_control": forced_acquisition_cost,
        "eligible_for_extension_claim": all(core.values()),
        "reason": (
            "A V5 extension remains eligible only when the registered selection, Safe Commit, "
            "and forced-acquisition cost checks all pass."
        ),
    }


def aggregate(
    records: dict[int, dict[str, Path]], *, repetitions: int, bootstrap_seed: int
) -> dict[str, Any]:
    if repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")
    loaded, present_policies = _validate_records(records)
    selection = contrast(
        loaded,
        baseline="direct_safe_commit",
        candidate="selected_safe_commit",
        operation=None,
        repetitions=repetitions,
        bootstrap_seed=bootstrap_seed,
    )
    safe = contrast(
        loaded,
        baseline="selected_commit",
        candidate="selected_safe_commit",
        operation=None,
        repetitions=repetitions,
        bootstrap_seed=bootstrap_seed + 1,
    )
    forced_cost = None
    if FORCED_ACQUISITION_POLICY in present_policies:
        forced_cost = contrast(
            loaded,
            baseline=FORCED_ACQUISITION_POLICY,
            candidate="selected_safe_commit",
            operation=None,
            repetitions=repetitions,
            bootstrap_seed=bootstrap_seed + 2,
        )
    operation_slices = {
        operation: {
            "selection_factor": contrast(
                loaded,
                baseline="direct_safe_commit",
                candidate="selected_safe_commit",
                operation=operation,
                repetitions=repetitions,
                bootstrap_seed=bootstrap_seed + 10 + index,
            ),
            "safe_commit_factor": contrast(
                loaded,
                baseline="selected_commit",
                candidate="selected_safe_commit",
                operation=operation,
                repetitions=repetitions,
                bootstrap_seed=bootstrap_seed + 20 + index,
            ),
        }
        for index, operation in enumerate(OPERATIONS)
    }
    return {
        "schema_version": "sn7-v5-matched-nonkeep-factorial-v2",
        "split": "val",
        "test_assets_read": False,
        "policies": list(present_policies),
        "model_seeds": sorted(records),
        "bootstrap": {
            "unit": "model seed, then AOI within each sampled seed",
            "repetitions": repetitions,
            "seed": bootstrap_seed,
        },
        "policy_metrics": policy_metrics(loaded, present_policies),
        "selection_factor": selection,
        "safe_commit_factor": safe,
        "operation_slices": operation_slices,
        "promotion": promotion(selection, safe, forced_cost),
        "inputs": {
            str(seed): {policy: str(path.resolve()) for policy, path in by_policy.items()}
            for seed, by_policy in sorted(records.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--record", action="append", type=parse_record, required=True)
    parser.add_argument("--repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    records: dict[int, dict[str, Path]] = defaultdict(dict)
    for model_seed, policy, path in args.record:
        if policy in records[model_seed]:
            raise ValueError(f"duplicate {model_seed}:{policy} record")
        records[model_seed][policy] = path
    result = aggregate(dict(records), repetitions=args.repetitions, bootstrap_seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
