#!/usr/bin/env python3
"""Build leakage-free PRE_TOOL features and utility labels for Active-Catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from activemap.agent.records import AgentBelief
from activemap.agent.tool_belief_data import PostAcquisitionToolPairExample
from activemap.agent.tool_belief_model import PairedToolBeliefUpdater
from activemap.agent.active_catalog_tool_gate import PRE_TOOL_FEATURE_NAMES, pre_tool_features
from activemap.models import EditOperation


class PairUpdater(Protocol):
    def update_pair(self, belief, quality_result, temporal_result): ...


@dataclass(frozen=True)
class UtilityWeights:
    correctness: float = 0.5
    tool_cost: float = 0.18
    false_edit: float = 0.35
    missed_edit: float = 0.20


BASE_FEATURE_NAMES = PRE_TOOL_FEATURE_NAMES


def _read(path: Path) -> list[PostAcquisitionToolPairExample]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(PostAcquisitionToolPairExample.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"invalid {path}:{line_number}: {exc}") from exc
    if not rows:
        raise ValueError(f"empty input: {path}")
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def deployed_edit(belief: AgentBelief, threshold: float) -> EditOperation:
    probabilities = belief.edit_probabilities
    if 1.0 - probabilities[0] < threshold:
        return EditOperation.KEEP
    return list(EditOperation)[
        max(range(1, len(probabilities)), key=probabilities.__getitem__)
    ]


def utility_gain(
    prior: AgentBelief,
    updated: AgentBelief,
    target: EditOperation,
    threshold: float,
    tool_cost: float,
    weights: UtilityWeights,
) -> tuple[float, dict[str, Any]]:
    target_index = list(EditOperation).index(target)
    nll_gain = math.log(max(updated.edit_probabilities[target_index], 1e-8)) - math.log(
        max(prior.edit_probabilities[target_index], 1e-8)
    )
    prior_edit = deployed_edit(prior, threshold)
    updated_edit = deployed_edit(updated, threshold)
    correctness_gain = float(updated_edit == target) - float(prior_edit == target)
    prior_false = target == EditOperation.KEEP and prior_edit != EditOperation.KEEP
    updated_false = target == EditOperation.KEEP and updated_edit != EditOperation.KEEP
    prior_missed = target != EditOperation.KEEP and prior_edit == EditOperation.KEEP
    updated_missed = target != EditOperation.KEEP and updated_edit == EditOperation.KEEP
    value = (
        nll_gain
        + weights.correctness * correctness_gain
        - weights.tool_cost * tool_cost
        - weights.false_edit * (float(updated_false) - float(prior_false))
        - weights.missed_edit * (float(updated_missed) - float(prior_missed))
    )
    return value, {
        "nll_gain": nll_gain,
        "correctness_gain": correctness_gain,
        "prior_edit": prior_edit.value,
        "forced_tool_edit": updated_edit.value,
        "prior_correct": prior_edit == target,
        "forced_tool_correct": updated_edit == target,
        "prior_false_edit": prior_false,
        "forced_tool_false_edit": updated_false,
        "prior_missed_edit": prior_missed,
        "forced_tool_missed_edit": updated_missed,
    }


def observable_features(row: PostAcquisitionToolPairExample) -> np.ndarray:
    metadata = row.metadata
    candidate = metadata.get("observable_candidate_features")
    if not isinstance(candidate, list) or len(candidate) != 13:
        raise ValueError("grounded pair lacks 13 observable candidate features")
    initial_budget = float(metadata["pre_acquisition_initial_budget"])
    remaining_budget = float(metadata["pre_acquisition_remaining_budget"])
    spent_cost = float(metadata["pre_acquisition_spent_cost"])
    if initial_budget <= 0.0:
        raise ValueError("initial budget must be positive")
    return pre_tool_features(
        row.post_acquisition_belief, [float(value) for value in candidate],
        evidence_cost=row.evidence_cost, tool_cost=row.tool_cost,
        initial_budget=initial_budget, remaining_budget=remaining_budget,
        spent_cost=spent_cost, step=int(metadata["pre_acquisition_step"]),
    )


def build_gate_features(
    rows: list[PostAcquisitionToolPairExample],
    updater: PairUpdater,
    *,
    weights: UtilityWeights,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    splits = {row.split for row in rows}
    if len(splits) != 1:
        raise ValueError("tool-gate features require one split")
    split = next(iter(splits))
    features = []
    records = []
    for row in rows:
        prior = row.post_acquisition_belief
        updated = updater.update_pair(
            prior, row.quality_result, row.temporal_result
        )
        threshold = float(row.metadata["operation_update_threshold"])
        gain, diagnostics = utility_gain(
            prior,
            updated,
            row.gt_edit,
            threshold,
            row.tool_cost,
            weights,
        )
        features.append(observable_features(row))
        records.append(
            {
                "example_id": row.example_id,
                "task_id": row.task_id,
                "split": row.split,
                "stage": "PRE_TOOL",
                "oracle_use_tool": gain > 0.0,
                "consensus_mean_utility_gain": gain,
                "target_edit": row.gt_edit.value,
                "source_transition_id": row.metadata["source_transition_id"],
                "tool_cost": row.tool_cost,
                "operation_update_threshold": threshold,
                "diagnostics": diagnostics,
                "model_input_contains_tool_results": False,
                "model_input_contains_target": False,
                "test_assets_read": False,
            }
        )
    array = np.stack(features)
    labels = [bool(row["oracle_use_tool"]) for row in records]
    utilities = [float(row["consensus_mean_utility_gain"]) for row in records]
    summary = {
        "schema_version": "active-catalog-pre-tool-gate-features-v1",
        "split": split,
        "examples": len(records),
        "tasks": len({row["task_id"] for row in records}),
        "feature_dim": int(array.shape[1]),
        "feature_names": list(BASE_FEATURE_NAMES),
        "call_targets": sum(labels),
        "skip_targets": len(labels) - sum(labels),
        "target_call_rate": sum(labels) / len(labels),
        "utility_mean": math.fsum(utilities) / len(utilities),
        "utility_positive_mean": (
            math.fsum(value for value in utilities if value > 0.0) / max(sum(labels), 1)
        ),
        "utility_negative_mean": (
            math.fsum(value for value in utilities if value <= 0.0)
            / max(len(labels) - sum(labels), 1)
        ),
        "utility_metadata": "tool-belief-counterfactual-quality-cost-safety-gain-v1",
        "utility_weights": asdict(weights),
        "pre_tool_observable_only": True,
        "tool_results_in_features": False,
        "target_in_features": False,
        "test_assets_read": False,
    }
    return array, records, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("grounded_pairs", type=Path)
    parser.add_argument("tool_belief_checkpoint", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--split", required=True, choices=("train", "val"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--correctness-weight", type=float, default=0.5)
    parser.add_argument("--tool-cost-weight", type=float, default=0.18)
    parser.add_argument("--false-edit-weight", type=float, default=0.35)
    parser.add_argument("--missed-edit-weight", type=float, default=0.20)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    weights = UtilityWeights(
        correctness=args.correctness_weight,
        tool_cost=args.tool_cost_weight,
        false_edit=args.false_edit_weight,
        missed_edit=args.missed_edit_weight,
    )
    if min(asdict(weights).values()) < 0.0:
        raise ValueError("utility weights must be non-negative")
    rows = [row for row in _read(args.grounded_pairs) if row.split == args.split]
    if not rows:
        raise ValueError(f"no {args.split} grounded pairs")
    updater = PairedToolBeliefUpdater.from_checkpoint(
        args.tool_belief_checkpoint, device=args.device
    )
    features, records, summary = build_gate_features(rows, updater, weights=weights)
    summary["sources"] = {
        "grounded_pairs": str(args.grounded_pairs.resolve()),
        "grounded_pairs_sha256": _sha256(args.grounded_pairs),
        "tool_belief_checkpoint": str(args.tool_belief_checkpoint.resolve()),
        "tool_belief_checkpoint_sha256": _sha256(args.tool_belief_checkpoint),
    }
    args.output_root.mkdir(parents=True)
    np.save(args.output_root / "features.npy", features)
    (args.output_root / "records.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in records),
        encoding="utf-8",
    )
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
