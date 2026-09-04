#!/usr/bin/env python3
"""Evaluate the trained paired Tool-to-Belief path on complete sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from activemap.agent.records import AgentBelief
from activemap.agent.tool_belief_data import ToolBeliefSequenceExample
from activemap.agent.tool_belief_model import PairedToolBeliefUpdater
from activemap.models import EditOperation
from scripts.evaluate_tool_belief_interventions import (
    _belief_delta,
    _operation_index,
    _summary,
)

EDIT_ORDER = list(EditOperation)


def _read(path: Path, expected_split: str) -> list[ToolBeliefSequenceExample]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(ToolBeliefSequenceExample.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"invalid {path}:{line_number}: {exc}") from exc
    if not rows:
        raise ValueError(f"empty Tool-to-Belief sequence dataset: {path}")
    if {row.split for row in rows} != {expected_split}:
        raise ValueError(f"sequence evaluation expected the {expected_split} split")
    return rows


def _stage() -> dict[str, list[Any]]:
    return {"targets": [], "predictions": [], "confidences": [], "costs": []}


def evaluate_sequences(
    rows: list[ToolBeliefSequenceExample],
    updater: PairedToolBeliefUpdater,
    *,
    minimum_macro_f1_gain: float,
    max_false_edit_increase: float,
    max_missed_edit_increase: float,
    max_noop_probability_drift: float,
    max_noop_confidence_drift: float,
    max_noop_geometry_drift: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    split = rows[0].split
    if any(row.split != split for row in rows):
        raise ValueError("sequence evaluation rows mix splits")
    baseline = _stage()
    paired = [_stage() for _ in range(3)]
    no_quality = [_stage() for _ in range(3)]
    no_current = [_stage() for _ in range(3)]
    teacher = [_stage() for _ in range(3)]
    details = []
    noop_deltas = []

    for row in rows:
        if len(row.steps) != 3:
            raise ValueError(f"{row.sequence_id} must contain exactly three steps")
        target_index = EDIT_ORDER.index(row.gt_edit)
        current = row.initial_belief
        no_quality_current = row.initial_belief
        no_current_belief = AgentBelief(
            edit_probabilities=[0.25] * len(EDIT_ORDER),
            confidence=0.5,
            uncertainty=1.0,
            geometry_delta=[0.0] * 8,
        )
        baseline["targets"].append(target_index)
        baseline["predictions"].append(_operation_index(row.initial_belief))
        baseline["confidences"].append(row.initial_belief.confidence)
        baseline["costs"].append(0.0)
        spent = 0.0

        for step_index, step in enumerate(row.steps):
            # Quality changes evidence memory, not the public editable-map belief.
            noop_delta = _belief_delta(current, current)
            noop_deltas.append(noop_delta)
            current = updater.update_pair(
                current, step.quality_result, step.temporal_result, use_quality=True
            )
            no_quality_current = updater.update_pair(
                no_quality_current,
                step.quality_result,
                step.temporal_result,
                use_quality=False,
            )
            no_current_belief = updater.update_pair(
                no_current_belief,
                step.quality_result,
                step.temporal_result,
                use_quality=True,
            )
            spent += step.quality_result.cost + step.temporal_result.cost
            for stage, belief in (
                (paired[step_index], current),
                (no_quality[step_index], no_quality_current),
                (no_current[step_index], no_current_belief),
            ):
                stage["targets"].append(target_index)
                stage["predictions"].append(_operation_index(belief))
                stage["confidences"].append(belief.confidence)
                stage["costs"].append(spent)
            teacher_belief = step.cumulative_target_belief
            teacher[step_index]["targets"].append(target_index)
            teacher[step_index]["predictions"].append(_operation_index(teacher_belief))
            teacher[step_index]["confidences"].append(teacher_belief.confidence)
            teacher[step_index]["costs"].append(spent)
            details.append(
                {
                    "sequence_id": row.sequence_id,
                    "episode_id": row.episode_id,
                    "evidence_id": step.evidence_id,
                    "step": step_index + 1,
                    "target": row.gt_edit.value,
                    "baseline": row.initial_belief.predicted_edit.value,
                    "baseline_probabilities": row.initial_belief.edit_probabilities,
                    "baseline_recommended_edit": (
                        row.initial_belief.recommended_edit.value
                        if row.initial_belief.recommended_edit is not None
                        else None
                    ),
                    "paired": current.predicted_edit.value,
                    "paired_probabilities": current.edit_probabilities,
                    "paired_confidence": current.confidence,
                    "no_quality": no_quality_current.predicted_edit.value,
                    "no_quality_probabilities": no_quality_current.edit_probabilities,
                    "no_quality_confidence": no_quality_current.confidence,
                    "no_current": no_current_belief.predicted_edit.value,
                    "teacher": teacher_belief.predicted_edit.value,
                    "spent_cost": spent,
                    "quality_public_belief_delta": noop_delta,
                }
            )

    summaries = {
        "identity": _summary("identity", **baseline),
    }
    for index in range(3):
        step = index + 1
        summaries[f"paired_{step}"] = _summary(f"paired_{step}", **paired[index])
        summaries[f"no_quality_{step}"] = _summary(
            f"no_quality_{step}", **no_quality[index]
        )
        summaries[f"no_current_{step}"] = _summary(
            f"no_current_{step}", **no_current[index]
        )
        summaries[f"teacher_{step}"] = _summary(f"teacher_{step}", **teacher[index])

    identity = summaries["identity"]
    final = summaries["paired_3"]
    noop_max = {
        key: max(item[key] for item in noop_deltas)
        for key in ("probability_l1", "confidence_absolute", "geometry_mae")
    }
    checks = {
        "macro_f1_gain": (
            final["macro_f1"] - identity["macro_f1"] >= minimum_macro_f1_gain
        ),
        "false_edit_safety": (
            final["false_edit_rate"] - identity["false_edit_rate"]
            <= max_false_edit_increase
        ),
        "missed_edit_safety": (
            final["missed_edit_rate"] - identity["missed_edit_rate"]
            <= max_missed_edit_increase
        ),
        "recurrent_not_worse": final["macro_f1"] >= summaries["paired_1"]["macro_f1"],
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
            "schema_version": "tool-belief-paired-sequence-eval-v1",
            "split": split,
            "sequence_count": len(rows),
            "step_count": len(details),
            "primary_path": "quality_context+temporal_evidence->belief_update",
            "quality_updates_public_belief": False,
            "test_assets_read": split == "test",
        },
        "summaries": summaries,
        "quality_public_belief_delta_max": noop_max,
        "quality_context_macro_f1_delta": (
            final["macro_f1"] - summaries["no_quality_3"]["macro_f1"]
        ),
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
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--frozen-test", action="store_true")
    parser.add_argument("--minimum-macro-f1-gain", type=float, default=0.01)
    parser.add_argument("--max-false-edit-increase", type=float, default=0.02)
    parser.add_argument("--max-missed-edit-increase", type=float, default=0.02)
    parser.add_argument("--max-noop-probability-drift", type=float, default=0.0)
    parser.add_argument("--max-noop-confidence-drift", type=float, default=0.0)
    parser.add_argument("--max-noop-geometry-drift", type=float, default=0.0)
    args = parser.parse_args()
    if args.split == "test":
        if not args.frozen_test:
            raise PermissionError("test belief evaluation requires --frozen-test")
        from activemap.frozen_test import assert_frozen_test_access

        assert_frozen_test_access()
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise FileExistsError(
                f"refusing to overwrite frozen test belief evaluation: {args.output_dir}"
            )
    elif args.frozen_test:
        raise ValueError("--frozen-test is valid only with --split test")
    rows = _read(args.val_jsonl, args.split)
    updater = PairedToolBeliefUpdater.from_checkpoint(args.checkpoint, device=args.device)
    report, details = evaluate_sequences(
        rows,
        updater,
        minimum_macro_f1_gain=args.minimum_macro_f1_gain,
        max_false_edit_increase=args.max_false_edit_increase,
        max_missed_edit_increase=args.max_missed_edit_increase,
        max_noop_probability_drift=args.max_noop_probability_drift,
        max_noop_confidence_drift=args.max_noop_confidence_drift,
        max_noop_geometry_drift=args.max_noop_geometry_drift,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "details.jsonl").open("w", encoding="utf-8") as handle:
        for row in details:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
