#!/usr/bin/env python3
"""Replace coarse frame utility with executable-map utility labels."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from activemap.data.argotweak_native import ArgoTweakNativeEpisode
from activemap.data.structured_map import derive_structured_atomic_edits
from activemap.models import EditOperation


def _confidence(proposal: dict[str, Any]) -> float:
    confidence = proposal.get("confidence") or {}
    joint = confidence.get("joint")
    if joint is None:
        joint = float(confidence.get("object", 0.0)) * float(confidence.get("change", 0.0))
    return min(1.0, max(0.0, float(joint)))


def _map_metrics(
    episode: ArgoTweakNativeEpisode,
    evidence: Any,
    gate: float,
    false_discovery_weight: float,
    missed_edit_weight: float,
) -> dict[str, float]:
    truth, _ = derive_structured_atomic_edits(
        Path(episode.prior_map_path), Path(episode.target_map_path), include_keep=True
    )
    truth_by_object = {row.object_id: row.operation for row in truth}
    truth_changed = {
        object_id: operation
        for object_id, operation in truth_by_object.items()
        if operation != EditOperation.KEEP
    }
    changed: dict[str, tuple[EditOperation, float]] = {}
    for proposal in evidence.proposals:
        operation = EditOperation(str(proposal["operation"]))
        confidence = _confidence(proposal)
        object_id = proposal.get("object_id")
        if operation == EditOperation.KEEP or confidence < gate or not object_id:
            continue
        key = str(object_id)
        if key not in changed or confidence > changed[key][1]:
            changed[key] = (operation, confidence)
    tp = sum(
        truth_changed.get(object_id) == operation
        for object_id, (operation, _) in changed.items()
    )
    fp = len(changed) - tp
    fn = len(truth_changed) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_edit_rate = fp / max(1, len(set(truth_by_object) | set(changed)))
    false_edit_discovery_rate = fp / max(1, tp + fp)
    missed_edit_rate = fn / max(1, len(truth_changed))
    utility = (
        f1
        - false_discovery_weight * false_edit_discovery_rate
        - missed_edit_weight * missed_edit_rate
    )
    return {
        "utility": utility,
        "map_edit_f1": f1,
        "map_edit_precision": precision,
        "map_edit_recall": recall,
        "map_false_edit_rate": false_edit_rate,
        "map_false_edit_discovery_rate": false_edit_discovery_rate,
        "map_missed_edit_rate": missed_edit_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--commit-confidence", type=float, required=True)
    parser.add_argument("--false-discovery-weight", type=float, default=0.5)
    parser.add_argument("--missed-edit-weight", type=float, default=0.5)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    episodes = [
        ArgoTweakNativeEpisode.model_validate_json(line)
        for line in args.episodes.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    evidence_index = {
        (episode.episode_id, evidence.evidence_id): (episode, evidence)
        for episode in episodes
        for evidence in episode.evidence
    }
    records = [
        json.loads(line)
        for line in (args.features / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    output_records = []
    for row in records:
        key = (str(row["episode_id"]), str(row["evidence_id"]))
        episode, evidence = evidence_index[key]
        metrics = _map_metrics(
            episode,
            evidence,
            args.commit_confidence,
            args.false_discovery_weight,
            args.missed_edit_weight,
        )
        output_records.append({**row, "operation_set_utility": row["utility"], **metrics})
    args.output_dir.mkdir(parents=True)
    shutil.copy2(args.features / "features.npy", args.output_dir / "features.npy")
    (args.output_dir / "records.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in output_records),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "activemap-argotweak-map-utility-features-v1",
        "episodes": len(episodes),
        "evidence_frames": len(output_records),
        "commit_confidence": args.commit_confidence,
        "false_discovery_weight": args.false_discovery_weight,
        "missed_edit_weight": args.missed_edit_weight,
        "input_features_unchanged": True,
        "labels": "executable edit F1 minus false-discovery and missed-edit penalties",
        "test_assets_read": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
