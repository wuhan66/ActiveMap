"""Validation-only aggregation and promotion for agent checkpoints."""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path
from typing import Any


def _weighted_mean(rows: list[dict[str, Any]], key: str) -> float:
    total = sum(int(row["sample_count"]) for row in rows)
    if total == 0:
        raise ValueError("cannot aggregate zero samples")
    return sum(float(row[key]) * int(row["sample_count"]) for row in rows) / total


def _normalized_auc(rows: list[dict[str, Any]], key: str) -> float:
    ordered = sorted(rows, key=lambda row: float(row["budget"]))
    if not ordered:
        raise ValueError("cannot compute AUC without rows")
    if len(ordered) == 1:
        return float(ordered[0][key])
    area = 0.0
    for left, right in pairwise(ordered):
        width = float(right["budget"]) - float(left["budget"])
        if width <= 0:
            raise ValueError("AUC budgets must be unique and strictly increasing")
        area += width * (float(left[key]) + float(right[key])) / 2.0
    span = float(ordered[-1]["budget"]) - float(ordered[0]["budget"])
    return area / span


def load_checkpoint_metrics(evaluation_root: Path, label: str) -> dict[str, Any]:
    actions_path = evaluation_root / label / "actions" / "summary.json"
    rollouts_path = evaluation_root / label / "rollouts" / "summary.json"
    actions = json.loads(actions_path.read_text(encoding="utf-8"))
    rollouts = json.loads(rollouts_path.read_text(encoding="utf-8"))
    if actions.get("test_assets_read") is not False:
        raise ValueError(f"{label}: static action result does not prove test isolation")
    if rollouts.get("protocol", {}).get("test_assets_read") is not False:
        raise ValueError(f"{label}: rollout result does not prove test isolation")

    result_rows = rollouts["results"]
    llm_rows = [row for row in result_rows if row["method"] == "qwen3_4b_sft"]
    greedy_rows = [
        row for row in result_rows if row["method"] == "edit_conditioned_selector"
    ]
    if not greedy_rows:
        greedy_rows = [row for row in result_rows if row["method"] == "greedy"]
    if not llm_rows or not greedy_rows:
        raise ValueError(
            f"{label}: missing qwen3_4b_sft or edit-conditioned selector rollout rows"
        )
    if {row["budget"] for row in llm_rows} != {row["budget"] for row in greedy_rows}:
        raise ValueError(f"{label}: Qwen and greedy budgets differ")

    validity = rollouts["llm_validity"]
    acquire = actions.get("per_action", {}).get("ACQUIRE", {})
    utility_key = (
        "mean_quality_cost_utility"
        if all("mean_quality_cost_utility" in row for row in llm_rows + greedy_rows)
        else "mean_joint_utility"
    )
    llm_auc = _normalized_auc(llm_rows, utility_key)
    greedy_auc = _normalized_auc(greedy_rows, utility_key)
    legacy_llm_auc = _normalized_auc(llm_rows, "mean_joint_utility")
    legacy_greedy_auc = _normalized_auc(greedy_rows, "mean_joint_utility")
    return {
        "label": label,
        "sample_count": int(actions["sample_count"]),
        "schema_valid_rate": float(actions["schema_valid_rate"]),
        "executable_valid_rate": float(actions["executable_valid_rate"]),
        "exact_action_accuracy": float(actions["exact_action_accuracy"]),
        "macro_f1": float(actions["macro_f1"]),
        "acquire_recall": float(acquire.get("recall", 0.0)),
        "mean_policy_utility": actions.get("mean_policy_utility"),
        "mean_regret": actions.get("mean_regret"),
        "rollout_schema_valid_rate": float(validity["schema_valid_rate"]),
        "rollout_executable_valid_rate": float(validity["executable_valid_rate"]),
        "rollout_fallback_rate": float(validity["fallback_rate"]),
        "terminal_accuracy": _weighted_mean(llm_rows, "terminal_accuracy"),
        "false_edit_rate": _weighted_mean(llm_rows, "false_edit_rate"),
        "missed_edit_rate": _weighted_mean(llm_rows, "missed_edit_rate"),
        "mean_cost": _weighted_mean(llm_rows, "mean_cost"),
        "mean_acquisitions": _weighted_mean(llm_rows, "mean_acquisitions"),
        "primary_utility_auc": llm_auc,
        "selector_primary_utility_auc": greedy_auc,
        "primary_utility_delta_vs_selector": llm_auc - greedy_auc,
        "joint_utility_auc": legacy_llm_auc,
        "greedy_joint_utility_auc": legacy_greedy_auc,
        "joint_utility_delta_vs_greedy": legacy_llm_auc - legacy_greedy_auc,
        "primary_utility_metric": utility_key,
        "budgets": sorted(float(row["budget"]) for row in llm_rows),
        "test_assets_read": False,
    }


def select_checkpoint(
    metrics: list[dict[str, Any]],
    *,
    min_schema_valid: float = 0.99,
    min_executable_valid: float = 0.99,
    min_acquire_recall: float = 0.05,
    max_false_edit: float = 0.05,
    max_fallback: float = 0.01,
    min_utility_delta_vs_greedy: float = 0.0,
) -> dict[str, Any]:
    assessed = []
    for row in metrics:
        failures = []
        gates = (
            (row["schema_valid_rate"] >= min_schema_valid, "schema_valid_rate"),
            (row["executable_valid_rate"] >= min_executable_valid, "executable_valid_rate"),
            (
                row["rollout_schema_valid_rate"] >= min_schema_valid,
                "rollout_schema_valid_rate",
            ),
            (
                row["rollout_executable_valid_rate"] >= min_executable_valid,
                "rollout_executable_valid_rate",
            ),
            (row["acquire_recall"] >= min_acquire_recall, "acquire_recall"),
            (row["false_edit_rate"] <= max_false_edit, "false_edit_rate"),
            (row["rollout_fallback_rate"] <= max_fallback, "rollout_fallback_rate"),
            (
                row.get(
                    "primary_utility_delta_vs_selector",
                    row["joint_utility_delta_vs_greedy"],
                )
                >= min_utility_delta_vs_greedy,
                "primary_utility_delta_vs_selector",
            ),
        )
        failures.extend(name for passed, name in gates if not passed)
        assessed.append({**row, "eligible": not failures, "failed_gates": failures})

    def ranking(row: dict[str, Any]) -> tuple[float, float, float, float]:
        return (
            float(row.get("primary_utility_auc", row["joint_utility_auc"])),
            float(row["macro_f1"]),
            float(row["exact_action_accuracy"]),
            -float(row["mean_regret"] if row["mean_regret"] is not None else 1e9),
        )
    eligible = [row for row in assessed if row["eligible"]]
    best_observed = max(assessed, key=ranking) if assessed else None
    selected = max(eligible, key=ranking) if eligible else None
    return {
        "protocol": {
            "split": "val",
            "test_assets_read": False,
            "selection_order": [
                "primary_utility_auc",
                "macro_f1",
                "exact_action_accuracy",
                "negative_mean_regret",
            ],
            "gates": {
                "min_schema_valid": min_schema_valid,
                "min_executable_valid": min_executable_valid,
                "min_acquire_recall": min_acquire_recall,
                "max_false_edit": max_false_edit,
                "max_fallback": max_fallback,
                "min_utility_delta_vs_greedy": min_utility_delta_vs_greedy,
            },
        },
        "selected_checkpoint": selected["label"] if selected else None,
        "best_observed_checkpoint": best_observed["label"] if best_observed else None,
        "promotion_passed": selected is not None,
        "checkpoints": assessed,
    }


def select_intervention_base(
    metrics: list[dict[str, Any]],
    *,
    min_schema_valid: float = 0.99,
    min_executable_valid: float = 0.99,
    min_acquire_recall: float = 0.05,
    max_fallback: float = 0.01,
) -> dict[str, Any]:
    """Select a functional active-policy base for a safety/utility intervention."""

    assessed = []
    for row in metrics:
        gates = (
            (row["schema_valid_rate"] >= min_schema_valid, "schema_valid_rate"),
            (row["executable_valid_rate"] >= min_executable_valid, "executable_valid_rate"),
            (
                row["rollout_schema_valid_rate"] >= min_schema_valid,
                "rollout_schema_valid_rate",
            ),
            (
                row["rollout_executable_valid_rate"] >= min_executable_valid,
                "rollout_executable_valid_rate",
            ),
            (row["acquire_recall"] >= min_acquire_recall, "acquire_recall"),
            (row["rollout_fallback_rate"] <= max_fallback, "rollout_fallback_rate"),
        )
        failures = [name for passed, name in gates if not passed]
        assessed.append({**row, "eligible_as_base": not failures, "failed_base_gates": failures})

    def ranking(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
        return (
            float(row.get("primary_utility_auc", row["joint_utility_auc"])),
            -float(row["false_edit_rate"]),
            float(row["macro_f1"]),
            float(row["exact_action_accuracy"]),
            float(row["acquire_recall"]),
        )

    eligible = [row for row in assessed if row["eligible_as_base"]]
    selected = max(eligible, key=ranking) if eligible else None
    return {
        "protocol": {
            "purpose": "safety_or_utility_intervention_base",
            "split": "val",
            "test_assets_read": False,
            "selection_order": [
                "primary_utility_auc",
                "negative_false_edit_rate",
                "macro_f1",
                "exact_action_accuracy",
                "acquire_recall",
            ],
            "gates": {
                "min_schema_valid": min_schema_valid,
                "min_executable_valid": min_executable_valid,
                "min_acquire_recall": min_acquire_recall,
                "max_fallback": max_fallback,
            },
            "note": "false-edit and utility promotion gates are intervention targets",
        },
        "selected_base": selected["label"] if selected else None,
        "selection_passed": selected is not None,
        "candidates": assessed,
    }
