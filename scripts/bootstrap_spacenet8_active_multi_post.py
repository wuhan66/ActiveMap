#!/usr/bin/env python3
"""Bootstrap active-selection gains and export the Safe Commit frontier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.evaluate_spacenet8_active_multi_post import (
    aggregate,
    feature_matrix,
    group_rows,
)


def interval(values: np.ndarray, draws: int, rng: np.random.Generator) -> dict:
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "probability_positive": float(np.mean(samples > 0)),
    }


def terminal_rows(chosen: list[dict], decisions: np.ndarray) -> dict[str, np.ndarray]:
    quality, false_edit, missed = [], [], []
    for row, commit in zip(chosen, decisions):
        no_edit = 0.0 if row["target_positive"] else 1.0
        quality.append(row["map_iou"] if commit else no_edit)
        false_edit.append(commit and not row["target_positive"] and row["false_positive_pixels"] > 0)
        missed.append(row["target_positive"] and (not commit or row["true_positive_pixels"] <= 0))
    return {
        "map_iou": np.asarray(quality, dtype=float),
        "false_edit": np.asarray(false_edit, dtype=float),
        "missed_edit": np.asarray(missed, dtype=float),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("candidate_results", type=Path, nargs="+")
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)

    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rows = aggregate(args.candidate_results)
    train_rows = [row for row in rows if row["split"] == "train"]
    train_groups = group_rows(rows, "train")
    val_groups = group_rows(rows, "val")
    ambiguous = [group for group in val_groups if len(group) > 1]
    ranker = GradientBoostingRegressor(
        n_estimators=100, max_depth=2, learning_rate=0.03, random_state=args.seed
    ).fit(feature_matrix(train_rows), [row["map_iou"] for row in train_rows])
    for row, score in zip(rows, ranker.predict(feature_matrix(rows))):
        row["ranker_score"] = float(score)

    learned = [max(group, key=lambda row: row["ranker_score"]) for group in ambiguous]
    first = [group[0] for group in ambiguous]
    entropy = [min(group, key=lambda row: row["mean_entropy"]) for group in ambiguous]
    oracle = [max(group, key=lambda row: row["map_iou"]) for group in ambiguous]
    random_quality = np.asarray([np.mean([row["map_iou"] for row in group]) for group in ambiguous])
    learned_quality = np.asarray([row["map_iou"] for row in learned])
    rng = np.random.default_rng(args.seed)
    selection_bootstrap = {
        "learned_minus_first": interval(
            learned_quality - np.asarray([row["map_iou"] for row in first]), args.draws, rng
        ),
        "learned_minus_random_expected": interval(learned_quality - random_quality, args.draws, rng),
        "learned_minus_min_entropy": interval(
            learned_quality - np.asarray([row["map_iou"] for row in entropy]), args.draws, rng
        ),
        "oracle_minus_learned": interval(
            np.asarray([row["map_iou"] for row in oracle]) - learned_quality, args.draws, rng
        ),
    }

    gate = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", random_state=args.seed))
    labels = np.asarray([
        row["map_iou"] > (0.0 if row["target_positive"] else 1.0) + 1e-9 for row in train_rows
    ], dtype=int)
    gate.fit(feature_matrix(train_rows), labels)
    train_chosen = [max(group, key=lambda row: row["ranker_score"]) for group in train_groups]
    train_probability = gate.predict_proba(feature_matrix(train_chosen))[:, 1]
    thresholds = np.linspace(0.05, 0.95, 19)
    frontier = []
    train_best = None
    for threshold in thresholds:
        metrics = terminal_rows(train_chosen, train_probability >= threshold)
        utility = metrics["map_iou"].mean() - 0.25 * metrics["false_edit"].mean()
        candidate = (float(utility), -float(metrics["false_edit"].mean()), float(threshold))
        train_best = candidate if train_best is None or candidate > train_best else train_best
    chosen = [max(group, key=lambda row: row["ranker_score"]) for group in val_groups]
    probability = gate.predict_proba(feature_matrix(chosen))[:, 1]
    for threshold in thresholds:
        metrics = terminal_rows(chosen, probability >= threshold)
        frontier.append({
            "threshold": float(threshold),
            "map_iou": float(metrics["map_iou"].mean()),
            "false_edit_rate": float(metrics["false_edit"].mean()),
            "missed_edit_rate": float(metrics["missed_edit"].mean()),
            "commit_rate": float(np.mean(probability >= threshold)),
        })
    safe = terminal_rows(chosen, probability >= train_best[2])
    commit = terminal_rows(chosen, np.ones(len(chosen), dtype=bool))
    reject = terminal_rows(chosen, np.zeros(len(chosen), dtype=bool))
    terminal_bootstrap = {}
    for comparison, right in (("safe_minus_commit", commit), ("safe_minus_reject", reject)):
        terminal_bootstrap[comparison] = {
            metric: interval(safe[metric] - right[metric], args.draws, rng)
            for metric in safe
        }

    result = {
        "schema_version": "activemap-spacenet8-active-bootstrap-v1",
        "draws": args.draws,
        "ambiguous_val_episodes": len(ambiguous),
        "all_val_episodes": len(val_groups),
        "train_selected_threshold": train_best[2],
        "selection_bootstrap": selection_bootstrap,
        "terminal_bootstrap": terminal_bootstrap,
        "safe_commit_frontier": frontier,
        "test_assets_read": False,
    }
    args.output_root.mkdir(parents=True)
    (args.output_root / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
