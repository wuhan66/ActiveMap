#!/usr/bin/env python3
"""Build reachable post-acquisition paired-tool supervision from frozen data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from activemap.agent.identifiers import public_evidence_id, public_task_id
from activemap.agent.records import AgentBelief
from activemap.agent.tool_belief_data import (
    PostAcquisitionToolPairExample,
    ToolBeliefSequenceExample,
)
from activemap.agent.tools import CounterfactualBeliefUpdater
from activemap.models import EditOperation
from activemap.selector_records import SelectorSample


def _read(path: Path, model: type[Any]) -> list[Any]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(model.model_validate_json(line))
        except Exception as exc:
            raise ValueError(f"invalid {path}:{line_number}: {exc}") from exc
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def _initial_samples(samples: list[SelectorSample], split: str) -> dict[str, SelectorSample]:
    selected: dict[str, SelectorSample] = {}
    signatures: dict[str, str] = {}
    for sample in samples:
        metadata = sample.metadata
        initial_id = metadata.get("initial_evidence_id")
        if (
            sample.split != split
            or not isinstance(initial_id, str)
            or metadata.get("selected_evidence_ids") != [initial_id]
            or int(metadata.get("oracle_step", -1)) != 0
        ):
            continue
        task_id = public_task_id(str(metadata.get("source_episode", sample.sample_id)))
        signature = json.dumps(
            {
                "initial": initial_id,
                "predictions": metadata.get("evidence_predictions"),
                "gt": metadata.get("gt_edit"),
            },
            sort_keys=True,
        )
        if task_id in signatures and signatures[task_id] != signature:
            raise ValueError(f"inconsistent initial state for {task_id}")
        signatures[task_id] = signature
        selected.setdefault(task_id, sample)
    if not selected:
        raise ValueError(f"no initial selector states for split={split}")
    return selected


def _target_belief(prior: AgentBelief, target: EditOperation, smoothing: float) -> AgentBelief:
    probabilities = [smoothing / 3.0] * 4
    probabilities[list(EditOperation).index(target)] = 1.0 - smoothing
    confidence = prior.confidence
    entropy = -math.fsum(
        value * math.log(max(value, 1e-12)) for value in probabilities
    ) / math.log(len(probabilities))
    return AgentBelief(
        edit_probabilities=probabilities,
        confidence=confidence,
        geometry_delta=prior.geometry_delta,
        uncertainty=entropy,
        recommended_edit=target,
    )


def build_examples(
    samples: list[SelectorSample],
    sequences: list[ToolBeliefSequenceExample],
    *,
    split: str,
    label_smoothing: float = 0.05,
) -> tuple[list[PostAcquisitionToolPairExample], dict[str, Any]]:
    if not 0.0 < label_smoothing < 0.25:
        raise ValueError("label_smoothing must be between zero and 0.25")
    initial = _initial_samples(samples, split)
    sequence_rows = [row for row in sequences if row.split == split]
    if set(initial) != {row.episode_id for row in sequence_rows}:
        raise ValueError("selector initial states and tool sequences have different task support")

    examples = []
    counts: Counter[str] = Counter()
    baseline_argmax_correct = 0
    baseline_decision_correct = 0
    recommended_count = 0
    for sequence in sequence_rows:
        sample = initial[sequence.episode_id]
        metadata = sample.metadata
        initial_id = str(metadata["initial_evidence_id"])
        raw_by_public = {public_evidence_id(raw): raw for raw in sample.evidence_ids}
        if len(raw_by_public) != len(sample.evidence_ids):
            raise ValueError(f"public evidence collision for {sequence.episode_id}")
        belief_updater = CounterfactualBeliefUpdater(sample)
        gt = EditOperation(str(metadata["gt_edit"]))
        for step in sequence.steps:
            raw_id = raw_by_public.get(step.evidence_id)
            if raw_id is None:
                raise ValueError(
                    f"unknown sequence evidence {sequence.episode_id}/{step.evidence_id}"
                )
            prior = belief_updater.fuse([initial_id, raw_id])
            index = sample.evidence_ids.index(raw_id)
            tool_cost = step.quality_result.cost + step.temporal_result.cost
            identity = f"{sequence.episode_id}|{step.evidence_id}|post-acquisition-v1"
            examples.append(
                PostAcquisitionToolPairExample(
                    example_id=hashlib.sha256(identity.encode()).hexdigest()[:24],
                    task_id=sequence.episode_id,
                    split=split,
                    evidence_id=step.evidence_id,
                    post_acquisition_belief=prior,
                    quality_result=step.quality_result,
                    temporal_result=step.temporal_result,
                    target_belief=_target_belief(prior, gt, label_smoothing),
                    gt_edit=gt,
                    evidence_cost=float(sample.evidence_costs[index]),
                    tool_cost=float(tool_cost),
                    metadata={
                        "teacher": (
                            "smoothed_ground_truth_posterior_with_preserved_confidence"
                        ),
                        "label_smoothing": label_smoothing,
                        "initial_evidence_id": public_evidence_id(initial_id),
                        "operation_update_threshold": float(
                            metadata["operation_update_threshold"]
                        ),
                        "test_assets_read": False,
                    },
                )
            )
            counts[f"gt:{gt.value}"] += 1
            argmax_edit = list(EditOperation)[
                max(
                    range(len(prior.edit_probabilities)),
                    key=prior.edit_probabilities.__getitem__,
                )
            ]
            baseline_argmax_correct += argmax_edit == gt
            baseline_decision_correct += prior.predicted_edit == gt
            recommended_count += prior.recommended_edit is not None

    summary = {
        "schema_version": "post-acquisition-tool-pair-dataset-v1",
        "split": split,
        "task_count": len(sequence_rows),
        "example_count": len(examples),
        "unique_example_count": len({row.example_id for row in examples}),
        "counts": dict(sorted(counts.items())),
        "baseline_argmax_accuracy": baseline_argmax_correct / len(examples),
        "baseline_deployed_decision_accuracy": baseline_decision_correct / len(examples),
        "recommended_edit_rate": recommended_count / len(examples),
        "label_smoothing": label_smoothing,
        "reachable_by_construction": True,
        "test_assets_read": False,
    }
    return examples, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("selector_states", type=Path)
    parser.add_argument("tool_sequences", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", required=True, choices=("train", "val"))
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    args = parser.parse_args()
    examples, summary = build_examples(
        _read(args.selector_states, SelectorSample),
        _read(args.tool_sequences, ToolBeliefSequenceExample),
        split=args.split,
        label_smoothing=args.label_smoothing,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for row in examples:
            handle.write(row.model_dump_json() + "\n")
    args.output.with_suffix(args.output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
