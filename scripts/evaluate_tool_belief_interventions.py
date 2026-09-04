#!/usr/bin/env python3
"""Evaluate one-step and recurrent grounded Tool-to-Belief interventions."""

from __future__ import annotations

import argparse
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from activemap.agent.records import AgentBelief
from activemap.agent.tool_belief_data import ToolBeliefExample
from activemap.agent.tool_belief_model import LearnedToolBeliefUpdater
from activemap.geo_tools.records import GeoToolName
from activemap.models import EditOperation

EDIT_ORDER = list(EditOperation)


def _read(path: Path) -> list[ToolBeliefExample]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = ToolBeliefExample.model_validate_json(line)
            except Exception as exc:
                raise ValueError(f"invalid {path}:{line_number}: {exc}") from exc
            if row.split != "val":
                raise ValueError(f"evaluation accepts validation rows only: {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"empty validation data: {path}")
    return rows


def _reward(target: int, prediction: int) -> float:
    keep = EDIT_ORDER.index(EditOperation.KEEP)
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
    keep = EDIT_ORDER.index(EditOperation.KEEP)
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
        class_metrics[operation.value] = {
            "support": int(np.sum(target == index)),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    false_edit = float(
        np.sum((target == keep) & (prediction != keep)) / max(np.sum(target == keep), 1)
    )
    missed_edit = float(
        np.sum((target != keep) & (prediction == keep)) / max(np.sum(target != keep), 1)
    )
    rewards = np.asarray([_reward(t, p) for t, p in zip(target, prediction, strict=True)])
    return {
        "method": name,
        "sample_count": len(targets),
        "accuracy": float(np.mean(target == prediction)),
        "macro_f1": float(np.mean(f1_scores)),
        "false_edit_rate": false_edit,
        "missed_edit_rate": missed_edit,
        "expected_calibration_error": _ece(confidence, target == prediction),
        "mean_confidence": float(np.mean(confidence)),
        "mean_tool_cost": float(np.mean(cost)),
        "mean_terminal_reward": float(np.mean(rewards)),
        "mean_joint_utility": float(np.mean(rewards - cost)),
        "class_metrics": class_metrics,
    }


def _belief_delta(before: AgentBelief, after: AgentBelief) -> dict[str, float]:
    return {
        "probability_l1": math.fsum(
            abs(left - right)
            for left, right in zip(
                before.edit_probabilities, after.edit_probabilities, strict=True
            )
        ),
        "confidence_absolute": abs(before.confidence - after.confidence),
        "geometry_mae": math.fsum(
            abs(left - right)
            for left, right in zip(before.geometry_delta, after.geometry_delta, strict=True)
        )
        / len(before.geometry_delta),
    }


def _operation_index(belief: AgentBelief) -> int:
    return EDIT_ORDER.index(belief.predicted_edit)


def evaluate(
    rows: list[ToolBeliefExample],
    updater: LearnedToolBeliefUpdater,
    *,
    max_false_edit_increase: float,
    max_missed_edit_increase: float,
    max_noop_probability_drift: float,
    max_noop_confidence_drift: float,
    max_noop_geometry_drift: float,
    minimum_macro_f1_gain: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: OrderedDict[str, OrderedDict[str, dict[GeoToolName, ToolBeliefExample]]] = (
        OrderedDict()
    )
    for row in rows:
        evidence = grouped.setdefault(row.episode_id, OrderedDict()).setdefault(
            row.evidence_id, {}
        )
        if row.tool_result.tool in evidence:
            raise ValueError(
                f"duplicate {row.tool_result.tool.value} for {row.episode_id}/{row.evidence_id}"
            )
        evidence[row.tool_result.tool] = row

    required_tools = {GeoToolName.IMAGE_QUALITY, GeoToolName.TEMPORAL_CHANGE}
    details: list[dict[str, Any]] = []
    one_step: dict[str, list[Any]] = {
        "target": [],
        "identity": [],
        "learned": [],
        "teacher": [],
        "no_current": [],
        "identity_confidence": [],
        "learned_confidence": [],
        "teacher_confidence": [],
        "no_current_confidence": [],
        "cost": [],
    }
    recurrent: dict[str, dict[str, list[Any]]] = {}
    noop_deltas = []

    for episode_id, evidence_rows in grouped.items():
        if len(evidence_rows) != 3:
            raise ValueError(f"{episode_id} has {len(evidence_rows)} evidence groups, expected 3")
        first = next(iter(evidence_rows.values()))
        first_row = next(iter(first.values()))
        initial = first_row.prior_belief
        target_index = EDIT_ORDER.index(first_row.gt_edit)
        neutral = AgentBelief(
            edit_probabilities=[0.25] * len(EDIT_ORDER),
            confidence=0.5,
            uncertainty=1.0,
            geometry_delta=[0.0] * 8,
        )
        temporal_belief = initial
        paired_belief = initial
        quality_belief = initial
        no_current_belief = neutral
        spent_temporal = 0.0
        spent_paired = 0.0
        spent_quality = 0.0

        for step, (evidence_id, tools) in enumerate(evidence_rows.items(), start=1):
            if set(tools) != required_tools:
                raise ValueError(
                    f"{episode_id}/{evidence_id} tools are {sorted(item.value for item in tools)}"
                )
            quality = tools[GeoToolName.IMAGE_QUALITY]
            temporal = tools[GeoToolName.TEMPORAL_CHANGE]
            if quality.prior_belief != initial or temporal.prior_belief != initial:
                raise ValueError(f"inconsistent prior belief within {episode_id}")
            if quality.gt_edit != first_row.gt_edit or temporal.gt_edit != first_row.gt_edit:
                raise ValueError(f"inconsistent ground truth within {episode_id}")

            quality_updated = updater.update(initial, quality.tool_result)
            quality_delta = _belief_delta(initial, quality_updated)
            noop_deltas.append(quality_delta)

            learned_single = updater.update(initial, temporal.tool_result)
            no_current_single = updater.update(neutral, temporal.tool_result)
            one_step["target"].append(target_index)
            one_step["identity"].append(_operation_index(initial))
            one_step["learned"].append(_operation_index(learned_single))
            one_step["teacher"].append(_operation_index(temporal.target_belief))
            one_step["no_current"].append(_operation_index(no_current_single))
            one_step["identity_confidence"].append(initial.confidence)
            one_step["learned_confidence"].append(learned_single.confidence)
            one_step["teacher_confidence"].append(temporal.target_belief.confidence)
            one_step["no_current_confidence"].append(no_current_single.confidence)
            one_step["cost"].append(temporal.tool_result.cost)

            temporal_belief = updater.update(temporal_belief, temporal.tool_result)
            paired_belief = updater.update(paired_belief, quality.tool_result)
            paired_belief = updater.update(paired_belief, temporal.tool_result)
            quality_belief = updater.update(quality_belief, quality.tool_result)
            no_current_belief = updater.update(no_current_belief, temporal.tool_result)
            spent_temporal += temporal.tool_result.cost
            spent_paired += quality.tool_result.cost + temporal.tool_result.cost
            spent_quality += quality.tool_result.cost
            for name, belief, cost in (
                ("learned_temporal", temporal_belief, spent_temporal),
                ("learned_quality_temporal", paired_belief, spent_paired),
                ("learned_quality", quality_belief, spent_quality),
                ("learned_no_current_temporal", no_current_belief, spent_temporal),
            ):
                stage = recurrent.setdefault(
                    f"{name}_{step}",
                    {"target": [], "prediction": [], "confidence": [], "cost": []},
                )
                stage["target"].append(target_index)
                stage["prediction"].append(_operation_index(belief))
                stage["confidence"].append(belief.confidence)
                stage["cost"].append(cost)

            details.append(
                {
                    "episode_id": episode_id,
                    "evidence_id": evidence_id,
                    "step": step,
                    "target": first_row.gt_edit.value,
                    "identity": initial.predicted_edit.value,
                    "learned_single": learned_single.predicted_edit.value,
                    "teacher_single": temporal.target_belief.predicted_edit.value,
                    "learned_no_current_single": no_current_single.predicted_edit.value,
                    "learned_temporal_recurrent": temporal_belief.predicted_edit.value,
                    "learned_paired_recurrent": paired_belief.predicted_edit.value,
                    "learned_quality_recurrent": quality_belief.predicted_edit.value,
                    "learned_no_current_recurrent": no_current_belief.predicted_edit.value,
                    "quality_noop_delta": quality_delta,
                }
            )

    zero_cost = [0.0] * len(one_step["target"])
    summaries = {
        "identity_one_step": _summary(
            "identity_one_step",
            one_step["target"],
            one_step["identity"],
            one_step["identity_confidence"],
            zero_cost,
        ),
        "learned_one_step": _summary(
            "learned_one_step",
            one_step["target"],
            one_step["learned"],
            one_step["learned_confidence"],
            one_step["cost"],
        ),
        "teacher_one_step": _summary(
            "teacher_one_step",
            one_step["target"],
            one_step["teacher"],
            one_step["teacher_confidence"],
            one_step["cost"],
        ),
        "learned_no_current_one_step": _summary(
            "learned_no_current_one_step",
            one_step["target"],
            one_step["no_current"],
            one_step["no_current_confidence"],
            one_step["cost"],
        ),
    }
    for name, values in recurrent.items():
        summaries[name] = _summary(
            name,
            values["target"],
            values["prediction"],
            values["confidence"],
            values["cost"],
        )

    identity = summaries["identity_one_step"]
    learned = summaries["learned_one_step"]
    recurrent_final = summaries["learned_temporal_3"]
    noop_max = {
        key: max(item[key] for item in noop_deltas)
        for key in ("probability_l1", "confidence_absolute", "geometry_mae")
    }
    checks = {
        "one_step_macro_f1_gain": (
            learned["macro_f1"] - identity["macro_f1"] >= minimum_macro_f1_gain
        ),
        "false_edit_safety": (
            learned["false_edit_rate"] - identity["false_edit_rate"]
            <= max_false_edit_increase
        ),
        "missed_edit_safety": (
            learned["missed_edit_rate"] - identity["missed_edit_rate"]
            <= max_missed_edit_increase
        ),
        "recurrent_not_worse": recurrent_final["macro_f1"] >= identity["macro_f1"],
        "noop_probability_safety": (
            noop_max["probability_l1"] <= max_noop_probability_drift
        ),
        "noop_confidence_safety": (
            noop_max["confidence_absolute"] <= max_noop_confidence_drift
        ),
        "noop_geometry_safety": noop_max["geometry_mae"] <= max_noop_geometry_drift,
    }
    report = {
        "protocol": {
            "schema_version": "tool-belief-intervention-eval-v1",
            "split": "val",
            "episode_count": len(grouped),
            "one_step_count": len(one_step["target"]),
            "tools_per_episode": 6,
            "recurrent_temporal_calls": 3,
            "inference_ablations": [
                "identity",
                "no_quality",
                "no_temporal",
                "no_current_belief",
            ],
            "test_assets_read": False,
        },
        "summaries": summaries,
        "quality_noop_delta_max": noop_max,
        "gates": {
            "thresholds": {
                "minimum_macro_f1_gain": minimum_macro_f1_gain,
                "max_false_edit_increase": max_false_edit_increase,
                "max_missed_edit_increase": max_missed_edit_increase,
                "max_noop_probability_drift": max_noop_probability_drift,
                "max_noop_confidence_drift": max_noop_confidence_drift,
                "max_noop_geometry_drift": max_noop_geometry_drift,
            },
            "checks": checks,
            "passed": all(checks.values()),
        },
    }
    return report, details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("val_jsonl", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--minimum-macro-f1-gain", type=float, default=0.01)
    parser.add_argument("--max-false-edit-increase", type=float, default=0.02)
    parser.add_argument("--max-missed-edit-increase", type=float, default=0.02)
    parser.add_argument("--max-noop-probability-drift", type=float, default=0.02)
    parser.add_argument("--max-noop-confidence-drift", type=float, default=0.05)
    parser.add_argument("--max-noop-geometry-drift", type=float, default=0.02)
    args = parser.parse_args()
    rows = _read(args.val_jsonl)
    updater = LearnedToolBeliefUpdater.from_checkpoint(args.checkpoint, device=args.device)
    report, details = evaluate(
        rows,
        updater,
        max_false_edit_increase=args.max_false_edit_increase,
        max_missed_edit_increase=args.max_missed_edit_increase,
        max_noop_probability_drift=args.max_noop_probability_drift,
        max_noop_confidence_drift=args.max_noop_confidence_drift,
        max_noop_geometry_drift=args.max_noop_geometry_drift,
        minimum_macro_f1_gain=args.minimum_macro_f1_gain,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "details.jsonl").open("w", encoding="utf-8") as handle:
        for row in details:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
