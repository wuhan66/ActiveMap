#!/usr/bin/env python3
"""Train and freeze a task-grouped CALL/STOP classifier on visual VLM states."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from activemap.agent.visual_gate import binary_call_metrics


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(
    root: Path, expected_split: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("test_assets_read") is not False:
        raise ValueError("feature summary does not preserve the frozen-test protocol")
    features = np.load(root / "features.npy").astype(np.float32)
    rows = [
        json.loads(line)
        for line in (root / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != len(features):
        raise ValueError("feature and record counts differ")
    if {row["split"] for row in rows} != {expected_split}:
        raise ValueError(f"expected only split={expected_split}")
    labels = np.asarray([int(row["oracle_use_tool"]) for row in rows], dtype=np.int64)
    utilities = np.asarray(
        [float(row.get("consensus_mean_utility_gain", "nan")) for row in rows],
        dtype=np.float64,
    )
    groups = np.asarray([str(row["task_id"]) for row in rows])
    ids = [str(row["example_id"]) for row in rows]
    return features, labels, utilities, groups, ids, summary


def utility_proxy_metrics(
    probability: np.ndarray, utilities: np.ndarray, threshold: float
) -> dict[str, float]:
    if not np.all(np.isfinite(utilities)):
        raise ValueError("utility-aware selection requires finite utility metadata")
    called = probability >= threshold
    realized = np.where(called, utilities, 0.0)
    risk = np.where(called, np.maximum(-utilities, 0.0), 0.0)
    return {
        "proxy_utility_sum": float(realized.sum()),
        "proxy_utility_mean": float(realized.mean()),
        "proxy_risk_sum": float(risk.sum()),
        "proxy_risk_mean": float(risk.mean()),
    }


def utility_sample_weights(
    labels: np.ndarray,
    utilities: np.ndarray,
    *,
    floor: float = 0.05,
) -> np.ndarray:
    """Weight costly mistakes more heavily without changing the class prior."""
    if len(labels) != len(utilities):
        raise ValueError("label and utility counts differ")
    if floor <= 0.0:
        raise ValueError("utility weight floor must be positive")
    if not np.all(np.isfinite(utilities)):
        raise ValueError("utility-magnitude weighting requires finite utility metadata")
    weights = np.maximum(np.abs(utilities), floor).astype(np.float64)
    for label in np.unique(labels):
        mask = labels == label
        class_mean = float(weights[mask].mean())
        if class_mean <= 0.0:
            raise ValueError(f"non-positive utility weight mean for label={label}")
        weights[mask] /= class_mean
    return weights


def utility_risk_weights(
    utilities: np.ndarray,
    *,
    floor: float = 0.05,
) -> np.ndarray:
    """Preserve the empirical class prior and asymmetric realized call costs."""
    if floor <= 0.0:
        raise ValueError("utility weight floor must be positive")
    values = np.asarray(utilities, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("utility-risk weighting requires finite utility metadata")
    return np.maximum(np.abs(values), floor)


def utility_promotion_checks(
    selected_metrics: dict[str, float],
    validation_metrics: dict[str, float],
    *,
    selection_objective: str,
) -> dict[str, bool]:
    if selection_objective != "proxy_utility":
        return {}
    return {
        "oof_proxy_utility_positive": selected_metrics["proxy_utility_sum"] > 0.0,
        "validation_proxy_utility_positive": validation_metrics["proxy_utility_sum"] > 0.0,
    }


def _model(c_value: float, seed: int, *, balance_classes: bool = True) -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            class_weight="balanced" if balance_classes else None,
            max_iter=5000,
            random_state=seed,
        ),
    )


def _fit_model(
    model: Any,
    features: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray | None,
) -> Any:
    fit_kwargs = {}
    if sample_weights is not None:
        fit_kwargs["logisticregression__sample_weight"] = sample_weights
    return model.fit(features, labels, **fit_kwargs)


def _select(
    features: np.ndarray,
    labels: np.ndarray,
    utilities: np.ndarray,
    groups: np.ndarray,
    *,
    c_values: tuple[float, ...],
    thresholds: tuple[float, ...],
    seed: int,
    max_call_rate: float,
    max_false_call_rate: float = 1.0,
    min_oof_recall: float,
    selection_objective: str,
    fit_weighting: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(np.unique(labels)) != 2:
        raise ValueError("training split must contain CALL and STOP examples")
    folds = min(5, len(np.unique(groups[labels == 1])), len(np.unique(groups[labels == 0])))
    if folds < 2:
        raise ValueError("insufficient positive and negative task groups for OOF selection")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    if fit_weighting == "utility_magnitude":
        sample_weights = utility_sample_weights(labels, utilities)
    elif fit_weighting == "utility_risk":
        sample_weights = utility_risk_weights(utilities)
    else:
        sample_weights = None
    balance_classes = fit_weighting != "utility_risk"
    grid = []
    for c_value in c_values:
        probability = np.full(len(labels), np.nan, dtype=np.float64)
        for fit_indices, holdout_indices in splitter.split(features, labels, groups):
            model = _model(c_value, seed, balance_classes=balance_classes)
            _fit_model(
                model,
                features[fit_indices],
                labels[fit_indices],
                None if sample_weights is None else sample_weights[fit_indices],
            )
            probability[holdout_indices] = model.predict_proba(features[holdout_indices])[:, 1]
        if not np.all(np.isfinite(probability)):
            raise RuntimeError("grouped OOF left examples unevaluated")
        for threshold in thresholds:
            metrics = binary_call_metrics(labels, probability, threshold)
            if np.all(np.isfinite(utilities)):
                metrics.update(utility_proxy_metrics(probability, utilities, threshold))
            grid.append({"C": c_value, "threshold": threshold, "metrics": metrics})
    feasible = [
        row
        for row in grid
        if 0.0 < row["metrics"]["call_rate"] <= max_call_rate
        and row["metrics"]["false_call_rate"] <= max_false_call_rate
        and row["metrics"]["recall"] >= min_oof_recall
    ]
    if not feasible:
        raise RuntimeError("no OOF gate satisfies call-rate and recall constraints")
    if selection_objective == "proxy_utility":
        if not np.all(np.isfinite(utilities)):
            raise ValueError("proxy_utility objective requires utility metadata")
        selected = max(
            feasible,
            key=lambda row: (
                row["metrics"]["proxy_utility_mean"],
                -row["metrics"]["proxy_risk_mean"],
                row["metrics"]["precision"],
                row["metrics"]["f0_5"],
                -row["metrics"]["call_rate"],
                -row["C"],
            ),
        )
    else:
        selected = max(
            feasible,
            key=lambda row: (
                row["metrics"]["f0_5"],
                row["metrics"]["precision"],
                row["metrics"]["f1"],
                -row["metrics"]["call_rate"],
                -row["C"],
            ),
        )
    return selected, grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_features", type=Path)
    parser.add_argument("val_features", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--C", dest="c_values", default="0.001,0.01,0.1,1,10")
    parser.add_argument(
        "--thresholds",
        default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.70,0.80,0.90",
    )
    parser.add_argument("--max-call-rate", type=float, default=0.50)
    parser.add_argument("--max-false-call-rate", type=float, default=1.0)
    parser.add_argument("--min-oof-recall", type=float, default=0.10)
    parser.add_argument(
        "--selection-objective",
        choices=("f0_5", "proxy_utility"),
        default="f0_5",
    )
    parser.add_argument(
        "--fit-weighting",
        choices=("balanced", "utility_magnitude", "utility_risk"),
        default="balanced",
    )
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if not 0.0 < args.max_call_rate <= 1.0:
        raise ValueError("max-call-rate must be in (0, 1]")
    if not 0.0 <= args.max_false_call_rate <= 1.0:
        raise ValueError("max-false-call-rate must be in [0, 1]")
    if not 0.0 <= args.min_oof_recall <= 1.0:
        raise ValueError("min-oof-recall must be in [0, 1]")
    train_x, train_y, train_utility, train_groups, _, train_feature_summary = _load(
        args.train_features, "train"
    )
    val_x, val_y, val_utility, _, val_ids, val_feature_summary = _load(
        args.val_features, "val"
    )
    if train_x.shape[1] != val_x.shape[1]:
        raise ValueError("train and validation feature dimensions differ")
    train_utility_metadata = train_feature_summary.get(
        "utility_metadata", "consensus-cost-adjusted-mean-gain"
    )
    val_utility_metadata = val_feature_summary.get(
        "utility_metadata", "consensus-cost-adjusted-mean-gain"
    )
    if train_utility_metadata != val_utility_metadata:
        raise ValueError("train and validation utility metadata differ")
    selected, grid = _select(
        train_x,
        train_y,
        train_utility,
        train_groups,
        c_values=tuple(float(value) for value in args.c_values.split(",")),
        thresholds=tuple(float(value) for value in args.thresholds.split(",")),
        seed=args.seed,
        max_call_rate=args.max_call_rate,
        max_false_call_rate=args.max_false_call_rate,
        min_oof_recall=args.min_oof_recall,
        selection_objective=args.selection_objective,
        fit_weighting=args.fit_weighting,
    )
    model = _model(
        float(selected["C"]),
        args.seed,
        balance_classes=args.fit_weighting != "utility_risk",
    )
    if args.fit_weighting == "utility_magnitude":
        train_sample_weights = utility_sample_weights(train_y, train_utility)
    elif args.fit_weighting == "utility_risk":
        train_sample_weights = utility_risk_weights(train_utility)
    else:
        train_sample_weights = None
    _fit_model(model, train_x, train_y, train_sample_weights)
    val_probability = model.predict_proba(val_x)[:, 1]
    validation = binary_call_metrics(val_y, val_probability, float(selected["threshold"]))
    if np.all(np.isfinite(val_utility)):
        validation.update(
            utility_proxy_metrics(
                val_probability, val_utility, float(selected["threshold"])
            )
        )
    promotion = {
        "nonzero_calls": validation["predicted_calls"] > 0,
        "call_rate_within_cap": validation["call_rate"] <= args.max_call_rate,
        "false_call_rate_within_cap": (
            validation["false_call_rate"] <= args.max_false_call_rate
        ),
        "recall_at_least_0_10": validation["recall"] >= 0.10,
        "precision_at_least_prevalence": validation["precision"] >= validation["target_call_rate"],
        **utility_promotion_checks(
            selected["metrics"],
            validation,
            selection_objective=args.selection_objective,
        ),
    }
    args.output_dir.mkdir(parents=True)
    dump(model, args.output_dir / "gate.joblib")
    (args.output_dir / "selection_grid.json").write_text(
        json.dumps(grid, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "validation_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for example_id, target, utility, probability in zip(
            val_ids, val_y, val_utility, val_probability, strict=True
        ):
            handle.write(
                json.dumps(
                    {
                        "example_id": example_id,
                        "target_use_tool": bool(target),
                        "consensus_mean_utility_gain": (
                            float(utility) if np.isfinite(utility) else None
                        ),
                        "call_probability": float(probability),
                        "predicted_use_tool": bool(probability >= selected["threshold"]),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
    summary = {
        "schema_version": "semantic-vlm-visual-tool-gate-v1",
        "selection_protocol": "stratified-task-grouped-OOF-train-only",
        "feature_protocol": "frozen-adapter-PRE_TOOL-prompt-state-with-recorded-pooling",
        "selection_objective": args.selection_objective,
        "fit_weighting": args.fit_weighting,
        "fit_weight_summary": (
            {
                str(label): {
                    "count": int((train_y == label).sum()),
                    "mean": float(train_sample_weights[train_y == label].mean()),
                    "min": float(train_sample_weights[train_y == label].min()),
                    "max": float(train_sample_weights[train_y == label].max()),
                }
                for label in np.unique(train_y)
            }
            if train_sample_weights is not None
            else None
        ),
        "utility_target": (
            train_utility_metadata if np.all(np.isfinite(train_utility)) else None
        ),
        "min_oof_recall": args.min_oof_recall,
        "seed": args.seed,
        "train_examples": len(train_y),
        "validation_examples": len(val_y),
        "feature_dim": int(train_x.shape[1]),
        "selected": selected,
        "validation": validation,
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
