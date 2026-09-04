#!/usr/bin/env python3
"""Audit a V6 Evidence Value policy on immutable V5 selector states.

The audit evaluates only action selection against the frozen per-candidate
counterfactual registry in a declared split. It does not rerun the updater or
write a map, so it cannot establish an executable map-update claim. Its role
is to decide whether a separately registered full writeback evaluation is
warranted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from activemap.selector_records import SelectorSample

OPERATIONS = ("KEEP", "ADD", "DELETE", "RESHAPE")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def target_operation(sample: SelectorSample) -> str:
    target = str(sample.metadata.get("gt_edit", sample.edit_type.value)).upper()
    if target not in OPERATIONS:
        raise ValueError(f"{sample.sample_id}: unsupported target operation {target!r}")
    return target


def audit_row(sample: SelectorSample, scores: np.ndarray, stop_margin: float) -> dict[str, Any]:
    if scores.shape != (len(sample.evidence_ids),):
        raise ValueError(f"{sample.sample_id}: score shape does not match candidates")
    if not np.isfinite(scores).all():
        raise ValueError(f"{sample.sample_id}: non-finite candidate score")
    outcomes = sample.metadata.get("executable_outcomes")
    if not isinstance(outcomes, dict):
        raise ValueError(f"{sample.sample_id}: missing executable outcomes")
    selected_index = int(np.argmax(scores))
    selected_id = sample.evidence_ids[selected_index]
    outcome = outcomes.get(selected_id)
    if not isinstance(outcome, dict):
        raise ValueError(f"{sample.sample_id}: missing outcome for selected candidate")
    candidate_utility = float(sample.oracle_utilities[selected_index])
    stop_utility = float(sample.stop_utility)
    oracle_index = int(np.argmax(np.asarray(sample.oracle_utilities + [stop_utility])))
    oracle_acquire = oracle_index < len(sample.evidence_ids)
    acquire = bool(scores[selected_index] > stop_margin)
    chosen_utility = candidate_utility if acquire else stop_utility
    harmful = bool(acquire and candidate_utility <= stop_utility)
    false_or_wrong = bool(outcome.get("false_edit", False) or outcome.get("wrong_edit", False))
    return {
        "sample_id": sample.sample_id,
        "source_episode": str(sample.metadata.get("source_episode", sample.sample_id)),
        "aoi_id": str(sample.metadata.get("aoi_id", "unknown")),
        "target_operation": target_operation(sample),
        "selected_evidence_id": selected_id,
        "selected_score": float(scores[selected_index]),
        "stop_margin": float(stop_margin),
        "selected_margin": float(scores[selected_index] - stop_margin),
        "acquire": acquire,
        "oracle_acquire": oracle_acquire,
        "exact_oracle_candidate": bool(acquire and selected_index == oracle_index),
        "candidate_utility": candidate_utility,
        "stop_utility": stop_utility,
        "chosen_utility": chosen_utility,
        "oracle_utility": float(max(max(sample.oracle_utilities), stop_utility)),
        "harmful_call": harmful,
        "false_or_wrong_call": bool(acquire and false_or_wrong),
        "missed_edit_after_selected_call": bool(acquire and outcome.get("missed_edit", False)),
        "selected_final_raster_iou": float(outcome["final_raster_iou"]),
        "selected_false_edit": bool(outcome.get("false_edit", False)),
        "selected_missed_edit": bool(outcome.get("missed_edit", False)),
        "selected_wrong_edit": bool(outcome.get("wrong_edit", False)),
        "test_assets_read": False,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot summarize no policy rows")
    calls = [row for row in rows if row["acquire"]]
    target_calls = [row for row in rows if row["oracle_acquire"]]
    return {
        "state_count": float(len(rows)),
        "acquire_rate": float(len(calls) / len(rows)),
        "oracle_acquire_rate": float(len(target_calls) / len(rows)),
        "acquire_recall": float(
            sum(row["acquire"] and row["oracle_acquire"] for row in rows)
            / max(len(target_calls), 1)
        ),
        "false_call_rate": float(
            sum(row["acquire"] and not row["oracle_acquire"] for row in rows)
            / len(rows)
        ),
        "harmful_call_fraction": float(
            sum(row["harmful_call"] for row in rows) / max(len(calls), 1)
        ),
        "false_or_wrong_call_fraction": float(
            sum(row["false_or_wrong_call"] for row in rows) / max(len(calls), 1)
        ),
        "missed_edit_call_fraction": float(
            sum(row["missed_edit_after_selected_call"] for row in rows)
            / max(len(calls), 1)
        ),
        "exact_oracle_candidate_rate": float(
            sum(row["exact_oracle_candidate"] for row in rows) / max(len(target_calls), 1)
        ),
        "mean_chosen_utility": float(np.mean([row["chosen_utility"] for row in rows])),
        "mean_oracle_utility": float(np.mean([row["oracle_utility"] for row in rows])),
        "mean_regret": float(
            np.mean([row["oracle_utility"] - row["chosen_utility"] for row in rows])
        ),
        "mean_selected_final_raster_iou": float(
            np.mean([row["selected_final_raster_iou"] for row in rows if row["acquire"]])
            if calls
            else 0.0
        ),
    }


def load_states(path: Path, *, split: str) -> list[SelectorSample]:
    samples = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        sample = SelectorSample.model_validate_json(line)
        if sample.split != split or int(sample.metadata.get("oracle_step", -1)) != 0:
            continue
        if sample.split == "test" or sample.metadata.get("test_assets_read") is True:
            raise ValueError(f"{path}:{line_number}: test provenance is forbidden")
        samples.append(sample)
    if not samples:
        raise ValueError(f"no initial {split} states in {path}")
    return samples


def main() -> None:
    from activemap.agent.evidence_value_head import load_evidence_value_predictor

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("states", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite V6 policy audit: {args.output_dir}")
    samples = load_states(args.states, split=args.split)
    predictor = load_evidence_value_predictor(str(args.checkpoint), device=args.device)
    rows = [
        audit_row(sample, predictor.score_sample(sample), predictor.safety_margin)
        for sample in samples
    ]
    by_operation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_operation[str(row["target_operation"])].append(row)
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    result = {
        "schema_version": "sn7-v6-evidence-value-policy-audit-v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "states": str(args.states.resolve()),
        "states_sha256": sha256(args.states),
        "split": args.split,
        "stop_margin": predictor.safety_margin,
        "test_assets_read": False,
        "interpretation": (
            "Policy-only audit against frozen candidate outcomes; it is not an "
            "executable vector-writeback result."
        ),
        "overall": summarize(rows),
        "operations": {
            operation: summarize(by_operation[operation])
            for operation in OPERATIONS
            if by_operation[operation]
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
