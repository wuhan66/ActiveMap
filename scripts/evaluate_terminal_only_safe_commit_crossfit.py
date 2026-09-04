#!/usr/bin/env python3
"""AOI-cross-fitted Safe Commit that changes only the terminal operation."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from activemap.evaluation.episode_utility import score_episode_profiles

try:
    from scripts.fit_active_catalog_terminal_arbiter import (
        observable_features,
    )
except ModuleNotFoundError:
    from fit_active_catalog_terminal_arbiter import (  # type: ignore[no-redef]
        observable_features,
    )


METRICS = (
    "terminal_accuracy",
    "false_edit_rate",
    "missed_edit_rate",
    "wrong_edit_rate",
    "mean_cost",
    "mean_quality_gain",
    "mean_quality_cost_utility",
    "balanced_utility",
    "safety_utility",
    "cost_aware_utility",
)


def load_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row["sample_id"])
            if sample_id in rows:
                raise ValueError(f"duplicate sample ID: {sample_id}")
            rows[sample_id] = row
    if not rows:
        raise ValueError(f"empty trace: {path}")
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "terminal_accuracy": fmean(float(row["terminal_correct"]) for row in rows),
        "false_edit_rate": fmean(float(row["false_edit"]) for row in rows),
        "missed_edit_rate": fmean(float(row["missed_edit"]) for row in rows),
        "wrong_edit_rate": fmean(float(row["wrong_edit"]) for row in rows),
        "mean_cost": fmean(float(row["spent_cost"]) for row in rows),
        "mean_quality_gain": fmean(float(row["quality_gain"]) for row in rows),
        "mean_quality_cost_utility": fmean(float(row["quality_cost_utility"]) for row in rows),
        "balanced_utility": fmean(float(row["episode_utility_v2_proxy_balanced"]) for row in rows),
        "safety_utility": fmean(float(row["episode_utility_v2_proxy_safety"]) for row in rows),
        "cost_aware_utility": fmean(
            float(row["episode_utility_v2_proxy_cost_aware"]) for row in rows
        ),
    }


def quantile_thresholds(values: list[float], count: int = 101) -> list[float]:
    ordered = sorted(set(values))
    if not ordered:
        return []
    if len(ordered) > count:
        ordered = sorted(
            {ordered[round(index * (len(ordered) - 1) / (count - 1))] for index in range(count)}
        )
    epsilon = 1e-9
    return [ordered[0] - epsilon, *ordered, ordered[-1] + epsilon]


def metric_vector(row: dict[str, Any]) -> list[float]:
    summary = summarize([row])
    return [summary[metric] for metric in METRICS]


def terminal_only_row(
    row: dict[str, Any], *, suppress: bool, gate: dict[str, Any]
) -> dict[str, Any]:
    result = dict(row)
    original = str(row["predicted_edit"])
    prediction = "KEEP" if suppress and original != "KEEP" else original
    target = str(row["target_edit"])
    false_edit = target == "KEEP" and prediction != "KEEP"
    missed_edit = target != "KEEP" and prediction == "KEEP"
    wrong_edit = target != "KEEP" and prediction not in {"KEEP", target}
    utilities = score_episode_profiles(
        final_map_quality=float(prediction == target),
        prior_map_quality=float(target == "KEEP"),
        spent_cost=float(row["spent_cost"]),
        budget=float(row["budget"]),
        false_edit=false_edit,
        missed_edit=missed_edit,
        wrong_edit=wrong_edit,
    )
    result.update(
        {
            "predicted_edit": prediction,
            "terminal_correct": prediction == target,
            "false_edit": false_edit,
            "missed_edit": missed_edit,
            "wrong_edit": wrong_edit,
            "episode_utility_v2_proxy": utilities,
            "episode_utility_v2_proxy_balanced": utilities["balanced"]["value"],
            "episode_utility_v2_proxy_safety": utilities["safety"]["value"],
            "episode_utility_v2_proxy_cost_aware": utilities["cost_aware"]["value"],
            "safe_commit": {
                **gate,
                "original_prediction": original,
                "terminal_suppressed": prediction != original,
                "trajectory_preserved": True,
                "uses_ground_truth_at_inference": False,
            },
        }
    )
    return result


def gate_value(row: dict[str, Any], gate: dict[str, Any]) -> float:
    if gate.get("model_type", "stump") == "stump":
        return float(observable_features(row)[str(gate["feature"])])
    if gate["model_type"] != "logistic_false_edit_risk":
        raise ValueError(f"unknown gate model: {gate['model_type']}")
    features = observable_features(row)
    names = list(gate["feature_names"])
    values = np.asarray([features[name] for name in names], dtype=np.float64)
    mean = np.asarray(gate["scaler_mean"], dtype=np.float64)
    scale = np.asarray(gate["scaler_scale"], dtype=np.float64)
    coefficient = np.asarray(gate["coefficient"], dtype=np.float64)
    logit = float(np.dot((values - mean) / scale, coefficient) + gate["intercept"])
    return float(1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0))))


def apply_gate(rows: list[dict[str, Any]], gate: dict[str, Any]) -> list[dict[str, Any]]:
    feature = str(gate["feature"])
    direction = str(gate["direction"])
    threshold = float(gate["threshold"])
    output = []
    for row in rows:
        value = gate_value(row, gate)
        commit = value >= threshold if direction == "ge" else value <= threshold
        output.append(
            terminal_only_row(
                row,
                suppress=str(row["predicted_edit"]) != "KEEP" and not commit,
                gate={
                    "schema_version": "terminal-only-safe-commit-v1",
                    "model_type": gate.get("model_type", "stump"),
                    "feature": feature,
                    "direction": direction,
                    "threshold": threshold,
                    "observed_value": value,
                    "fit_split": gate["fit_split"],
                },
            )
        )
    return output


def fit_gate(
    candidate: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    *,
    gate_model: str = "stump",
) -> tuple[dict[str, Any], dict[str, float]]:
    reference_metrics = summarize(reference)
    features = observable_features(candidate[0])
    dummy_gate = {
        "schema_version": "terminal-only-safe-commit-v1",
        "feature": "fit_cache",
        "direction": "none",
        "threshold": 0.0,
        "fit_split": "fit_cache",
    }
    original_metrics = np.asarray([metric_vector(row) for row in candidate])
    suppressed_rows = [terminal_only_row(row, suppress=True, gate=dummy_gate) for row in candidate]
    suppressed_metrics = np.asarray([metric_vector(row) for row in suppressed_rows])
    nonkeep = np.asarray([row["predicted_edit"] != "KEEP" for row in candidate])
    configurations: list[tuple[str, str, np.ndarray, dict[str, Any]]] = []
    if gate_model == "stump":
        for feature in features:
            feature_values = np.asarray(
                [observable_features(row)[feature] for row in candidate],
                dtype=np.float64,
            )
            for direction in ("ge", "le"):
                configurations.append((feature, direction, feature_values, {"model_type": "stump"}))
    elif gate_model == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        feature_names = sorted(features)
        matrix = np.asarray(
            [[observable_features(row)[feature] for feature in feature_names] for row in candidate],
            dtype=np.float64,
        )
        targets = np.asarray([bool(row["false_edit"]) for row in candidate])
        if len(set(targets[nonkeep].tolist())) != 2:
            raise ValueError("logistic Safe Commit requires both risk classes")
        scaler = StandardScaler().fit(matrix[nonkeep])
        classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=20260801,
            solver="liblinear",
        ).fit(scaler.transform(matrix[nonkeep]), targets[nonkeep])
        risks = classifier.predict_proba(scaler.transform(matrix))[:, 1]
        configurations.append(
            (
                "false_edit_risk",
                "le",
                risks,
                {
                    "model_type": "logistic_false_edit_risk",
                    "feature_names": feature_names,
                    "scaler_mean": scaler.mean_.tolist(),
                    "scaler_scale": scaler.scale_.tolist(),
                    "coefficient": classifier.coef_[0].tolist(),
                    "intercept": float(classifier.intercept_[0]),
                    "class_weight": "balanced",
                },
            )
        )
    else:
        raise ValueError(f"unknown gate model: {gate_model}")

    best: tuple[tuple[float, ...], dict[str, Any], dict[str, float]] | None = None
    for feature, direction, feature_values, model_payload in configurations:
        for threshold in quantile_thresholds(feature_values[nonkeep].tolist()):
            gate = {
                **model_payload,
                "feature": feature,
                "direction": direction,
                "threshold": threshold,
                "fit_split": "validation_aoi_crossfit_train_fold",
            }
            commit = (
                feature_values >= threshold if direction == "ge" else feature_values <= threshold
            )
            selected = np.where(commit[:, None], original_metrics, suppressed_metrics).mean(axis=0)
            metrics = dict(zip(METRICS, selected.tolist(), strict=True))
            suppressed = int(np.count_nonzero(nonkeep & ~commit))
            false_safe = metrics["false_edit_rate"] <= reference_metrics["false_edit_rate"] + 1e-12
            missed_safe = (
                metrics["missed_edit_rate"] <= reference_metrics["missed_edit_rate"] + 1e-12
            )
            accuracy_safe = (
                metrics["terminal_accuracy"] >= reference_metrics["terminal_accuracy"] - 1e-12
            )
            eligible = false_safe and missed_safe and accuracy_safe
            violation = (
                max(
                    metrics["false_edit_rate"] - reference_metrics["false_edit_rate"],
                    0.0,
                )
                + max(
                    metrics["missed_edit_rate"] - reference_metrics["missed_edit_rate"],
                    0.0,
                )
                + max(
                    reference_metrics["terminal_accuracy"] - metrics["terminal_accuracy"],
                    0.0,
                )
            )
            key = (
                float(eligible),
                -violation,
                metrics["balanced_utility"],
                metrics["terminal_accuracy"],
                -float(suppressed),
            )
            if best is None or key > best[0]:
                best = key, gate, metrics
    if best is None:
        raise ValueError("cannot fit Safe Commit without non-KEEP predictions")
    gate = {
        "schema_version": "terminal-only-safe-commit-gate-v1",
        **best[1],
        "selection_constraints": [
            "false_edit_not_above_reference_on_fit_support",
            "missed_edit_not_above_reference_on_fit_support",
            "terminal_accuracy_not_below_reference_on_fit_support",
        ],
        "selection_objective": "balanced_utility_then_accuracy_then_missed_edit",
        "fit_constraints_satisfied": (
            best[2]["false_edit_rate"] <= reference_metrics["false_edit_rate"] + 1e-12
            and best[2]["missed_edit_rate"] <= reference_metrics["missed_edit_rate"] + 1e-12
            and best[2]["terminal_accuracy"] >= reference_metrics["terminal_accuracy"] - 1e-12
        ),
        "uses_ground_truth_at_inference": False,
    }
    return gate, best[2]


def crossfit_seed(
    candidate: dict[str, dict[str, Any]],
    reference: dict[str, dict[str, Any]],
    *,
    gate_model: str = "stump",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if candidate.keys() != reference.keys():
        raise ValueError("candidate and reference supports differ")
    aoi_ids = sorted({str(row["aoi_id"]) for row in candidate.values()})
    if len(aoi_ids) < 3:
        raise ValueError("AOI cross-fit requires at least three AOIs")
    output: list[dict[str, Any]] = []
    fold_gates: dict[str, dict[str, Any]] = {}
    for held_out in aoi_ids:
        train_ids = [
            sample_id for sample_id, row in candidate.items() if str(row["aoi_id"]) != held_out
        ]
        held_ids = [
            sample_id for sample_id, row in candidate.items() if str(row["aoi_id"]) == held_out
        ]
        gate, _ = fit_gate(
            [candidate[key] for key in train_ids],
            [reference[key] for key in train_ids],
            gate_model=gate_model,
        )
        fold_gates[held_out] = gate
        output.extend(apply_gate([candidate[key] for key in held_ids], gate))
    final_gate, final_fit_metrics = fit_gate(
        list(candidate.values()), list(reference.values()), gate_model=gate_model
    )
    final_gate["fit_split"] = "validation_all_aoi_frozen_for_test"
    return output, {
        "fold_gates": fold_gates,
        "final_frozen_test_gate": final_gate,
        "final_validation_fit_metrics_not_oof": final_fit_metrics,
    }


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = round(probability * (len(ordered) - 1))
    return ordered[index]


def paired_aoi_bootstrap(
    candidates: list[list[dict[str, Any]]],
    comparators: list[list[dict[str, Any]]],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    aoi_ids = sorted({str(row["aoi_id"]) for row in candidates[0]})
    rng = random.Random(seed)

    grouped: list[dict[str, tuple[int, dict[str, float]]]] = []
    for candidate, comparator in zip(candidates, comparators, strict=True):
        seed_groups: dict[str, tuple[int, dict[str, float]]] = {}
        for aoi_id in aoi_ids:
            c_rows = [row for row in candidate if str(row["aoi_id"]) == aoi_id]
            r_rows = [row for row in comparator if str(row["aoi_id"]) == aoi_id]
            if len(c_rows) != len(r_rows) or not c_rows:
                raise ValueError(f"invalid paired AOI support: {aoi_id}")
            c_metrics, r_metrics = summarize(c_rows), summarize(r_rows)
            seed_groups[aoi_id] = (
                len(c_rows),
                {
                    metric: (c_metrics[metric] - r_metrics[metric]) * len(c_rows)
                    for metric in METRICS
                },
            )
        grouped.append(seed_groups)

    def delta_for(sampled: list[str]) -> dict[str, float]:
        deltas = defaultdict(list)
        counts = Counter(sampled)
        for seed_groups in grouped:
            denominator = sum(counts[aoi_id] * seed_groups[aoi_id][0] for aoi_id in aoi_ids)
            for metric in METRICS:
                numerator = sum(
                    counts[aoi_id] * seed_groups[aoi_id][1][metric] for aoi_id in aoi_ids
                )
                deltas[metric].append(numerator / denominator)
        return {metric: fmean(values) for metric, values in deltas.items()}

    observed = delta_for(aoi_ids)
    draws = {metric: [] for metric in METRICS}
    for _ in range(repetitions):
        sampled = [rng.choice(aoi_ids) for _ in aoi_ids]
        values = delta_for(sampled)
        for metric in METRICS:
            draws[metric].append(values[metric])
    return {
        metric: {
            "observed_delta": observed[metric],
            "ci95_low": _quantile(draws[metric], 0.025),
            "ci95_high": _quantile(draws[metric], 0.975),
        }
        for metric in METRICS
    }


def _parse_candidate(value: str) -> tuple[str, Path]:
    seed, separator, path = value.partition("=")
    if not separator or not seed or not path:
        raise argparse.ArgumentTypeError("candidate must use SEED=PATH")
    return seed, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--candidate", action="append", type=_parse_candidate, required=True)
    parser.add_argument("--repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260801)
    parser.add_argument("--gate-model", choices=("stump", "logistic"), default="stump")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    reference = load_rows(args.reference)
    candidate_sets, safe_sets, reference_sets = [], [], []
    seed_summaries: dict[str, Any] = {}
    args.output_dir.mkdir(parents=True)
    for seed, path in args.candidate:
        candidate = load_rows(path)
        safe_rows, gates = crossfit_seed(candidate, reference, gate_model=args.gate_model)
        candidate_rows = list(candidate.values())
        reference_rows = [reference[str(row["sample_id"])] for row in safe_rows]
        candidate_sets.append(candidate_rows)
        safe_sets.append(safe_rows)
        reference_sets.append(reference_rows)
        seed_dir = args.output_dir / f"seed{seed}"
        seed_dir.mkdir()
        with (seed_dir / "oof_traces.jsonl").open("w", encoding="utf-8") as handle:
            for row in sorted(safe_rows, key=lambda item: str(item["sample_id"])):
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        (seed_dir / "gates.json").write_text(json.dumps(gates, indent=2) + "\n", encoding="utf-8")
        seed_summaries[seed] = {
            "candidate": summarize(candidate_rows),
            "safe_commit_oof": summarize(safe_rows),
            "reference": summarize(reference_rows),
            "suppressed": sum(row["safe_commit"]["terminal_suppressed"] for row in safe_rows),
        }

    summary = {
        "schema_version": "terminal-only-safe-commit-crossfit-v1",
        "protocol": {
            "split": "val",
            "grouping": "leave_one_AOI_out",
            "terminal_only": True,
            "trajectory_and_cost_preserved": True,
            "test_assets_read": False,
            "final_test_gate_fitted_on_all_validation": True,
            "final_test_gate_not_used_for_reported_oof_metrics": True,
            "gate_model": args.gate_model,
        },
        "seeds": seed_summaries,
        "safe_minus_candidate": paired_aoi_bootstrap(
            safe_sets,
            candidate_sets,
            repetitions=args.repetitions,
            seed=args.bootstrap_seed,
        ),
        "safe_minus_reference": paired_aoi_bootstrap(
            safe_sets,
            reference_sets,
            repetitions=args.repetitions,
            seed=args.bootstrap_seed + 1,
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
