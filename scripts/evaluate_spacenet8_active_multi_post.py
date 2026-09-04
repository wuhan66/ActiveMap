#!/usr/bin/env python3
"""Evaluate active POST selection and safe commit on frozen candidate outcomes."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


FEATURES = ("bounds_iou", "valid_fraction", "mean_probability", "mean_entropy", "predicted_fraction")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def aggregate(paths: list[Path]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for path in paths:
        for row in read_jsonl(path):
            grouped[(str(row["sample_id"]), str(row["candidate_id"]))].append(row)
    outputs = []
    for rows in grouped.values():
        first = rows[0]
        expected = len(paths)
        if len(rows) != expected:
            raise ValueError(f"candidate {first['candidate_id']} has {len(rows)}/{expected} seeds")
        positive = float(np.mean([row["positive_pixels"] for row in rows]))
        false_negative = float(np.mean([row["false_negative_pixels"] for row in rows]))
        false_positive = float(np.mean([row["false_positive_pixels"] for row in rows]))
        true_positive = max(0.0, positive - false_negative)
        union = true_positive + false_positive + false_negative
        outputs.append(
            {
                **{key: first[key] for key in first if key not in {
                    "iou", "f1", "precision", "recall", "false_positive_pixels",
                    "false_negative_pixels", "mean_probability", "mean_entropy",
                    "predicted_fraction", "valid_fraction", "mask_path"
                }},
                **{key: float(np.mean([row[key] for row in rows])) for key in FEATURES},
                "true_positive_pixels": true_positive,
                "false_positive_pixels": false_positive,
                "false_negative_pixels": false_negative,
                "target_positive": positive > 0,
                "prediction_positive": true_positive + false_positive > 0,
                "map_iou": true_positive / union if union > 0 else 1.0,
                "seed_count": len(rows),
            }
        )
    return outputs


def feature_matrix(rows: list[dict]) -> np.ndarray:
    return np.asarray([[float(row[key]) for key in FEATURES] for row in rows], dtype=np.float64)


def group_rows(rows: list[dict], split: str) -> list[list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["split"] == split:
            grouped[str(row["sample_id"])].append(row)
    return [sorted(group, key=lambda row: int(row["candidate_index"])) for group in grouped.values()]


def selection_summary(groups: list[list[dict]], chooser) -> tuple[dict, list[dict]]:
    chosen = [chooser(group) for group in groups]
    oracle = [max(group, key=lambda row: row["map_iou"]) for group in groups]
    return (
        {
            "episodes": len(groups),
            "mean_map_iou": float(np.mean([row["map_iou"] for row in chosen])),
            "mean_oracle_regret": float(np.mean([
                best["map_iou"] - row["map_iou"] for row, best in zip(chosen, oracle)
            ])),
            "exact_oracle_selection": float(np.mean([
                row["candidate_id"] == best["candidate_id"] for row, best in zip(chosen, oracle)
            ])),
            "false_edit_rate": float(np.mean([
                (not row["target_positive"]) and row["false_positive_pixels"] > 0 for row in chosen
            ])),
        },
        chosen,
    )


def terminal_summary(chosen: list[dict], commit: list[bool]) -> dict:
    qualities, false_edits, missed = [], [], []
    for row, decision in zip(chosen, commit):
        no_edit_quality = 0.0 if row["target_positive"] else 1.0
        qualities.append(row["map_iou"] if decision else no_edit_quality)
        false_edits.append(decision and not row["target_positive"] and row["false_positive_pixels"] > 0)
        missed.append(row["target_positive"] and (not decision or row["true_positive_pixels"] <= 0))
    return {
        "episodes": len(chosen),
        "mean_map_iou": float(np.mean(qualities)),
        "false_edit_rate": float(np.mean(false_edits)),
        "missed_edit_rate": float(np.mean(missed)),
        "commit_rate": float(np.mean(commit)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("candidate_results", type=Path, nargs="+")
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
    ambiguous_val = [group for group in val_groups if len(group) > 1]
    if not ambiguous_val:
        raise ValueError("validation split has no multi-POST episodes")

    ranker = GradientBoostingRegressor(
        n_estimators=100, max_depth=2, learning_rate=0.03, random_state=args.seed
    )
    ranker.fit(feature_matrix(train_rows), np.asarray([row["map_iou"] for row in train_rows]))
    for row, score in zip(rows, ranker.predict(feature_matrix(rows))):
        row["ranker_score"] = float(score)

    policies = {
        "first": lambda group: group[0],
        "max_coverage": lambda group: max(group, key=lambda row: row["bounds_iou"]),
        "min_entropy": lambda group: min(group, key=lambda row: row["mean_entropy"]),
        "learned_ranker": lambda group: max(group, key=lambda row: row["ranker_score"]),
        "oracle": lambda group: max(group, key=lambda row: row["map_iou"]),
    }
    selection = {}
    selected = {}
    for name, chooser in policies.items():
        selection[name], selected[name] = selection_summary(ambiguous_val, chooser)
    selection["random_expected"] = {
        "episodes": len(ambiguous_val),
        "mean_map_iou": float(np.mean([
            np.mean([row["map_iou"] for row in group]) for group in ambiguous_val
        ])),
        "mean_oracle_regret": float(np.mean([
            max(row["map_iou"] for row in group) - np.mean([row["map_iou"] for row in group])
            for group in ambiguous_val
        ])),
    }

    gate = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", random_state=args.seed))
    beneficial = np.asarray([
        row["map_iou"] > (0.0 if row["target_positive"] else 1.0) + 1e-9 for row in train_rows
    ], dtype=np.int64)
    gate.fit(feature_matrix(train_rows), beneficial)
    train_chosen = [max(group, key=lambda row: row["ranker_score"]) for group in train_groups]
    train_probability = gate.predict_proba(feature_matrix(train_chosen))[:, 1]
    thresholds = np.linspace(0.05, 0.95, 19)
    best = None
    for threshold in thresholds:
        summary = terminal_summary(train_chosen, list(train_probability >= threshold))
        utility = summary["mean_map_iou"] - 0.25 * summary["false_edit_rate"]
        candidate = (utility, -summary["false_edit_rate"], float(threshold))
        if best is None or candidate > best:
            best = candidate
    threshold = best[2]
    val_chosen = [max(group, key=lambda row: row["ranker_score"]) for group in val_groups]
    val_probability = gate.predict_proba(feature_matrix(val_chosen))[:, 1]
    terminal = {
        "always_reject": terminal_summary(val_chosen, [False] * len(val_chosen)),
        "always_commit": terminal_summary(val_chosen, [True] * len(val_chosen)),
        "safe_commit": terminal_summary(val_chosen, list(val_probability >= threshold)),
    }
    result = {
        "schema_version": "activemap-spacenet8-active-multi-post-v1",
        "protocol": {
            "features": FEATURES,
            "train_episodes": len(train_groups),
            "val_episodes": len(val_groups),
            "ambiguous_val_episodes": len(ambiguous_val),
            "seed_count": len(args.candidate_results),
            "safe_commit_threshold": threshold,
            "test_assets_read": False,
        },
        "ambiguous_selection": selection,
        "terminal_all_validation": terminal,
    }
    args.output_root.mkdir(parents=True)
    (args.output_root / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (args.output_root / "aggregated_candidates.jsonl").open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
