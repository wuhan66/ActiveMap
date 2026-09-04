#!/usr/bin/env python3
"""Measure the utility ceiling for conditional stopping in Tool-to-Belief rollouts."""

from __future__ import annotations

import argparse
import json
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

EDIT_ORDER = ("KEEP", "ADD", "DELETE", "RESHAPE")


def _reward(target: int, prediction: int) -> float:
    keep = EDIT_ORDER.index("KEEP")
    if target == prediction:
        return 1.0
    if target == keep:
        return -1.0
    if prediction == keep:
        return -0.75
    return -0.5


def _ece(confidence: np.ndarray, correctness: np.ndarray, bins: int = 10) -> float:
    error = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        selected = (confidence >= edges[index]) & (
            confidence <= edges[index + 1]
            if index == bins - 1
            else confidence < edges[index + 1]
        )
        if np.any(selected):
            error += float(np.mean(selected)) * abs(
                float(np.mean(correctness[selected])) - float(np.mean(confidence[selected]))
            )
    return error


def _summary(
    name: str,
    targets: list[int],
    predictions: list[int],
    confidences: list[float],
    costs: list[float],
) -> dict[str, Any]:
    target = np.asarray(targets)
    prediction = np.asarray(predictions)
    confidence = np.asarray(confidences)
    cost = np.asarray(costs)
    keep = EDIT_ORDER.index("KEEP")
    f1_scores = []
    class_metrics = {}
    for index, operation in enumerate(EDIT_ORDER):
        tp = int(np.sum((target == index) & (prediction == index)))
        fp = int(np.sum((target != index) & (prediction == index)))
        fn = int(np.sum((target == index) & (prediction != index)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        f1_scores.append(f1)
        class_metrics[operation] = {
            "support": int(np.sum(target == index)),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    rewards = np.asarray([_reward(t, p) for t, p in zip(target, prediction, strict=True)])
    return {
        "method": name,
        "sample_count": len(targets),
        "accuracy": float(np.mean(target == prediction)),
        "macro_f1": float(np.mean(f1_scores)),
        "false_edit_rate": float(
            np.sum((target == keep) & (prediction != keep))
            / max(np.sum(target == keep), 1)
        ),
        "missed_edit_rate": float(
            np.sum((target != keep) & (prediction == keep))
            / max(np.sum(target != keep), 1)
        ),
        "expected_calibration_error": _ece(confidence, target == prediction),
        "mean_confidence": float(np.mean(confidence)),
        "mean_tool_cost": float(np.mean(cost)),
        "mean_terminal_reward": float(np.mean(rewards)),
        "mean_joint_utility": float(np.mean(rewards - cost)),
        "class_metrics": class_metrics,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid {path}:{line_number}: {exc}") from exc
    if not rows:
        raise ValueError(f"empty sequence details: {path}")
    return rows


def analyze_stopping(
    report: dict[str, Any], details: list[dict[str, Any]]
) -> dict[str, Any]:
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for row in details:
        grouped.setdefault(str(row["sequence_id"]), []).append(row)
    targets = []
    predictions = []
    confidences = []
    costs = []
    selected_steps: Counter[int] = Counter()
    oracle_rows = []

    for sequence_id, rows in grouped.items():
        rows.sort(key=lambda item: int(item["step"]))
        if [int(item["step"]) for item in rows] != [1, 2, 3]:
            raise ValueError(f"{sequence_id} does not contain steps 1, 2, 3")
        target = str(rows[0]["target"])
        identity = str(rows[0]["baseline"])
        choices = [
            {
                "step": 0,
                "prediction": identity,
                "confidence": float(
                    report["summaries"]["identity"]["mean_confidence"]
                ),
                "cost": 0.0,
            }
        ]
        for row in rows:
            step = int(row["step"])
            default_cost = float(
                report["summaries"][f"paired_{step}"]["mean_tool_cost"]
            )
            choices.append(
                {
                    "step": step,
                    "prediction": str(row["paired"]),
                    "confidence": float(row.get("paired_confidence", 0.5)),
                    "cost": float(row.get("spent_cost", default_cost)),
                }
            )
        best = max(
            choices,
            key=lambda item: (
                _reward(EDIT_ORDER.index(target), EDIT_ORDER.index(item["prediction"]))
                - item["cost"],
                -item["cost"],
            ),
        )
        targets.append(EDIT_ORDER.index(target))
        predictions.append(EDIT_ORDER.index(best["prediction"]))
        confidences.append(best["confidence"])
        costs.append(best["cost"])
        selected_steps[best["step"]] += 1
        oracle_rows.append(
            {
                "sequence_id": sequence_id,
                "target": target,
                "selected_step": best["step"],
                "prediction": best["prediction"],
                "cost": best["cost"],
            }
        )

    oracle = _summary(
        "validation_stopping_oracle", targets, predictions, confidences, costs
    )
    fixed = {
        "step_0": report["summaries"]["identity"],
        **{
            f"step_{step}": report["summaries"][f"paired_{step}"]
            for step in range(1, 4)
        },
    }
    best_fixed_name, best_fixed = max(
        fixed.items(), key=lambda item: item[1]["mean_joint_utility"]
    )
    return {
        "protocol": {
            "schema_version": "tool-belief-stopping-opportunity-v1",
            "split": "val",
            "sequence_count": len(grouped),
            "oracle_is_upper_bound_not_deployable": True,
            "test_assets_read": False,
        },
        "fixed_stages": {
            name: {
                key: metrics[key]
                for key in (
                    "macro_f1",
                    "false_edit_rate",
                    "missed_edit_rate",
                    "mean_tool_cost",
                    "mean_terminal_reward",
                    "mean_joint_utility",
                )
            }
            for name, metrics in fixed.items()
        },
        "best_fixed_stage": best_fixed_name,
        "validation_oracle": oracle,
        "oracle_selected_step_counts": {
            str(step): selected_steps[step] for step in range(4)
        },
        "oracle_utility_gain_over_identity": (
            oracle["mean_joint_utility"]
            - fixed["step_0"]["mean_joint_utility"]
        ),
        "oracle_utility_gain_over_best_fixed": (
            oracle["mean_joint_utility"] - best_fixed["mean_joint_utility"]
        ),
        "oracle_rows": oracle_rows,
    }


def render_stopping(result: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fixed = result["fixed_stages"]
    labels = ["STOP", "1 step", "2 steps", "3 steps", "Oracle"]
    utilities = [fixed[f"step_{step}"]["mean_joint_utility"] for step in range(4)]
    utilities.append(result["validation_oracle"]["mean_joint_utility"])
    counts = [result["oracle_selected_step_counts"][str(step)] for step in range(4)]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].bar(
        labels,
        utilities,
        color=["#64748b", "#d97706", "#d97706", "#d97706", "#059669"],
    )
    axes[0].axhline(0.0, color="#111827", linewidth=0.8)
    axes[0].set(title="Utility of fixed stopping rules", ylabel="Mean joint utility")
    axes[1].bar(
        ["STOP", "Step 1", "Step 2", "Step 3"],
        counts,
        color=["#64748b", "#2563eb", "#2563eb", "#2563eb"],
    )
    axes[1].set(title="Validation-oracle selected actions", ylabel="Episodes")
    for axis in axes:
        axis.grid(axis="y", alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Conditional Tool-Use Opportunity", fontsize=14, fontweight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence_summary", type=Path)
    parser.add_argument("sequence_details", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    report = json.loads(args.sequence_summary.read_text(encoding="utf-8"))
    result = analyze_stopping(report, _read_jsonl(args.sequence_details))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.plot is not None:
        render_stopping(result, args.plot)
    printable = {key: value for key, value in result.items() if key != "oracle_rows"}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
