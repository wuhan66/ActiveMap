#!/usr/bin/env python3
"""Fit a task-grouped expected-utility gate on frozen visual VLM states."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from joblib import dump
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from activemap.agent.visual_gate import binary_call_metrics
from scripts.evaluate_sequential_selector import selector_metrics, task_bootstrap
from scripts.train_visual_tool_gate import _load, utility_proxy_metrics


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_thresholds(
    scores: np.ndarray, call_rates: tuple[float, ...]
) -> tuple[float, ...]:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("scores must be a finite vector")
    if not call_rates or any(not 0.0 < rate <= 1.0 for rate in call_rates):
        raise ValueError("candidate call rates must be in (0, 1]")
    return tuple(
        sorted(
            {
                float(np.quantile(values, 1.0 - rate, method="higher"))
                for rate in call_rates
            },
            reverse=True,
        )
    )


def _model(alpha: float) -> Any:
    return make_pipeline(
        StandardScaler(),
        Ridge(alpha=alpha, solver="lsqr", tol=1e-3, max_iter=2000),
    )


def feature_protocol_metadata(
    train_summary: dict[str, Any], val_summary: dict[str, Any]
) -> dict[str, str]:
    """Validate and preserve the observable feature-stage contract."""
    stage = train_summary.get("controller_stage")
    val_stage = val_summary.get("controller_stage")
    if stage not in {"PRE_TOOL", "SELECT"}:
        raise ValueError(f"unsupported training controller stage: {stage!r}")
    if val_stage != stage:
        raise ValueError("train and validation controller stages differ")

    pooling = train_summary.get("pooling")
    if pooling not in {"last", "last_mean"}:
        raise ValueError(f"unsupported training feature pooling: {pooling!r}")
    if val_summary.get("pooling") != pooling:
        raise ValueError("train and validation feature pooling differs")

    return {
        "controller_stage": stage,
        "pooling": pooling,
        "feature_protocol": f"frozen-adapter-{stage}-prompt-state-with-recorded-pooling",
    }


def validation_seed_metrics(
    labels: np.ndarray,
    utilities: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    model_seeds: list[int | None],
) -> dict[str, dict[str, float]]:
    if len(model_seeds) != len(labels):
        raise ValueError("model seed metadata must align with validation rows")
    observed = sorted({seed for seed in model_seeds if seed is not None})
    result = {}
    for seed in observed:
        selected = np.asarray([value == seed for value in model_seeds])
        metrics = binary_call_metrics(
            labels[selected], scores[selected], threshold
        )
        metrics.update(
            utility_proxy_metrics(
                scores[selected], utilities[selected], threshold
            )
        )
        result[str(seed)] = metrics
    return result


def select_expected_utility_gate(
    features: np.ndarray,
    labels: np.ndarray,
    utilities: np.ndarray,
    groups: np.ndarray,
    *,
    alphas: tuple[float, ...],
    call_rates: tuple[float, ...],
    seed: int,
    max_call_rate: float,
    min_oof_recall: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not np.all(np.isfinite(utilities)):
        raise ValueError("expected-utility regression requires finite utility metadata")
    folds = min(5, len(np.unique(groups[labels == 1])), len(np.unique(groups[labels == 0])))
    if folds < 2:
        raise ValueError("insufficient positive and negative task groups for OOF selection")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    grid: list[dict[str, Any]] = []
    for alpha in alphas:
        oof_score = np.full(len(labels), np.nan, dtype=np.float64)
        for fit_indices, holdout_indices in splitter.split(features, labels, groups):
            model = _model(alpha).fit(features[fit_indices], utilities[fit_indices])
            oof_score[holdout_indices] = model.predict(features[holdout_indices])
        if not np.all(np.isfinite(oof_score)):
            raise RuntimeError("grouped OOF left examples unevaluated")
        for threshold in candidate_thresholds(oof_score, call_rates):
            metrics = binary_call_metrics(labels, oof_score, threshold)
            metrics.update(utility_proxy_metrics(oof_score, utilities, threshold))
            grid.append({"alpha": alpha, "threshold": threshold, "metrics": metrics})
    feasible = [
        row
        for row in grid
        if 0.0 < row["metrics"]["call_rate"] <= max_call_rate
        and row["metrics"]["recall"] >= min_oof_recall
    ]
    if not feasible:
        raise RuntimeError("no OOF utility gate satisfies call-rate and recall constraints")
    selected = max(
        feasible,
        key=lambda row: (
            row["metrics"]["proxy_utility_mean"],
            -row["metrics"]["proxy_risk_mean"],
            row["metrics"]["precision"],
            -row["metrics"]["call_rate"],
            -row["alpha"],
        ),
    )
    return selected, grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_features", type=Path)
    parser.add_argument("val_features", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--alphas", default="0.1,1,10,100,1000")
    parser.add_argument(
        "--candidate-call-rates",
        default="0.01,0.02,0.03,0.05,0.075,0.10,0.15,0.20,0.30,0.40,0.50",
    )
    parser.add_argument("--max-call-rate", type=float, default=0.50)
    parser.add_argument("--min-oof-recall", type=float, default=0.10)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    train_x, train_y, train_utility, train_groups, _, train_summary = _load(
        args.train_features, "train"
    )
    val_x, val_y, val_utility, _, val_ids, val_summary = _load(args.val_features, "val")
    val_rows = [
        json.loads(line)
        for line in (args.val_features / "records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    val_model_seeds = [
        int(row["model_seed"]) if row.get("model_seed") is not None else None
        for row in val_rows
    ]
    if train_x.shape[1] != val_x.shape[1]:
        raise ValueError("train and validation feature dimensions differ")
    utility_target = train_summary.get("utility_metadata")
    if utility_target != val_summary.get("utility_metadata"):
        raise ValueError("train and validation utility metadata differ")
    feature_metadata = feature_protocol_metadata(train_summary, val_summary)

    selected, grid = select_expected_utility_gate(
        train_x,
        train_y,
        train_utility,
        train_groups,
        alphas=tuple(float(value) for value in args.alphas.split(",")),
        call_rates=tuple(float(value) for value in args.candidate_call_rates.split(",")),
        seed=args.seed,
        max_call_rate=args.max_call_rate,
        min_oof_recall=args.min_oof_recall,
    )
    model = _model(float(selected["alpha"])).fit(train_x, train_utility)
    val_score = model.predict(val_x)
    threshold = float(selected["threshold"])
    validation = binary_call_metrics(val_y, val_score, threshold)
    validation.update(utility_proxy_metrics(val_score, val_utility, threshold))
    per_seed_validation = validation_seed_metrics(
        val_y,
        val_utility,
        val_score,
        threshold,
        val_model_seeds,
    )
    validation_traces = [
        {
            "example_id": example_id,
            "task_id": str(row["task_id"]),
            "split": "val",
            "target_selection": "ACQUIRE" if bool(target) else "STOP",
            "predicted_selection": "ACQUIRE" if float(score) >= threshold else "STOP",
            "policy_relative_advantage": float(utility),
            "valid_action": True,
        }
        for example_id, target, utility, score, row in zip(
            val_ids, val_y, val_utility, val_score, val_rows, strict=True
        )
    ]
    validation_selector_metrics = selector_metrics(validation_traces)
    validation_task_bootstrap = task_bootstrap(
        validation_traces,
        repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    promotion = {
        "oof_utility_positive": selected["metrics"]["proxy_utility_sum"] > 0.0,
        "validation_nonzero_calls": validation["predicted_calls"] > 0,
        "validation_call_rate_within_cap": validation["call_rate"] <= args.max_call_rate,
        "validation_recall_at_least_0_10": validation["recall"] >= 0.10,
        "validation_proxy_utility_positive": validation["proxy_utility_sum"] > 0.0,
        "validation_task_utility_ci_above_zero": validation_task_bootstrap["intervals"][
            "realized_utility_mean"
        ]["ci95_low"]
        > 0.0,
        "all_model_seed_validation_proxy_utility_nonnegative": (
            not per_seed_validation
            or all(
                metrics["proxy_utility_sum"] >= 0.0
                for metrics in per_seed_validation.values()
            )
        ),
    }

    args.output_dir.mkdir(parents=True)
    dump(model, args.output_dir / "gate.joblib")
    (args.output_dir / "selection_grid.json").write_text(
        json.dumps(grid, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "validation_predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for example_id, target, utility, score, model_seed in zip(
            val_ids,
            val_y,
            val_utility,
            val_score,
            val_model_seeds,
            strict=True,
        ):
            handle.write(
                json.dumps(
                    {
                        "example_id": example_id,
                        "target_use_tool": bool(target),
                        "policy_relative_realized_advantage": float(utility),
                        "predicted_utility": float(score),
                        "predicted_use_tool": bool(score >= threshold),
                        "model_seed": model_seed,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
    with (args.output_dir / "validation_traces.jsonl").open("w", encoding="utf-8") as handle:
        for row in validation_traces:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {
        "schema_version": "semantic-vlm-visual-utility-gate-v1",
        "selection_protocol": "stratified-task-grouped-OOF-train-only",
        **feature_metadata,
        "score_type": "predicted_utility",
        "selection_objective": "proxy_utility",
        "fit_weighting": "continuous_realized_advantage",
        "utility_target": utility_target,
        "min_oof_recall": args.min_oof_recall,
        "max_call_rate": args.max_call_rate,
        "seed": args.seed,
        "train_examples": len(train_y),
        "validation_examples": len(val_y),
        "feature_dim": int(train_x.shape[1]),
        "selected": selected,
        "validation": validation,
        "validation_selector_metrics": validation_selector_metrics,
        "validation_task_bootstrap": validation_task_bootstrap,
        "per_model_seed_validation": per_seed_validation,
        "promotion_gate": {**promotion, "passed": all(promotion.values())},
        "sources": {
            "train": {
                "path": str(args.train_features.resolve()),
                "summary_sha256": _sha256(args.train_features / "summary.json"),
            },
            "val": {
                "path": str(args.val_features.resolve()),
                "summary_sha256": _sha256(args.val_features / "summary.json"),
            },
        },
        "test_assets_read": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
