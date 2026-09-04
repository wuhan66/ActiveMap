#!/usr/bin/env python3
"""Evaluate ArgoTweak evidence policies after executable vector-map writeback."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from activemap.data.argotweak_native import (
    ArgoTweakNativeEpisode,
    proposal_city_geometry,
)
from activemap.data.structured_map import (
    StructuredMapObservation,
    StructuredMapSample,
    _load_feature_map,
    derive_structured_atomic_edits,
    write_structured_map_jsonl,
)
from activemap.evaluation.structured_map import evaluate_structured_map_predictions
from activemap.integrations.baselines.contracts import (
    StructuredMapPrediction,
    write_structured_prediction_jsonl,
)
from activemap.models import EditOperation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rankings", type=Path)
    parser.add_argument("--budgets", default="1,3,5,10")
    parser.add_argument("--commit-confidence", type=float, default=0.5)
    parser.add_argument("--false-edit-weight", type=float, default=0.5)
    parser.add_argument("--missed-edit-weight", type=float, default=0.5)
    parser.add_argument("--cost-weight", type=float, default=0.05)
    return parser.parse_args()


def _read_episodes(path: Path) -> list[ArgoTweakNativeEpisode]:
    rows = [
        ArgoTweakNativeEpisode.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"no native episodes in {path}")
    return rows


def _read_rankings(path: Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = {str(row["episode_id"]): [str(value) for value in row["ranked_evidence_ids"]] for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate episode IDs in ranking file")
    return result


def _joint_confidence(proposal: dict[str, Any]) -> float:
    confidence = proposal.get("confidence") or {}
    joint = confidence.get("joint")
    if joint is None:
        joint = float(confidence.get("object", 0.0)) * float(confidence.get("change", 0.0))
    return min(1.0, max(0.0, float(joint)))


def _observable_score(evidence: Any) -> tuple[float, float, str]:
    changed = [row for row in evidence.proposals if row.get("operation") != EditOperation.KEEP.value]
    confidence = max((_joint_confidence(row) for row in changed), default=0.0)
    ready_fraction = len(evidence.commit_ready_proposal_ids) / max(1, len(evidence.proposals))
    return (confidence, ready_fraction, evidence.evidence_id)


def _selection(
    episode: ArgoTweakNativeEpisode,
    policy: str,
    budget: int,
    learned_rankings: dict[str, list[str]],
    commit_confidence: float,
) -> list[Any]:
    evidence_by_id = {row.evidence_id: row for row in episode.evidence}
    if policy == "prior":
        return []
    if policy == "direct_center":
        return [episode.evidence[len(episode.evidence) // 2]]
    if policy == "acquire_all":
        return list(episode.evidence)
    if policy == "observable":
        return sorted(episode.evidence, key=_observable_score, reverse=True)[:budget]
    if policy == "operation_oracle":
        return sorted(
            episode.evidence,
            key=lambda row: (row.utility, row.evidence_id),
            reverse=True,
        )[:budget]
    if policy == "map_oracle":
        truth, _ = derive_structured_atomic_edits(
            Path(episode.prior_map_path), Path(episode.target_map_path), include_keep=True
        )
        truth_by_object = {row.object_id: row.operation for row in truth}
        all_object_count = len(truth_by_object)
        truth_changed = {
            object_id: operation
            for object_id, operation in truth_by_object.items()
            if operation != EditOperation.KEEP
        }
        selected: list[Any] = []
        remaining = list(episode.evidence)
        while remaining and len(selected) < budget:
            def score(candidate: Any) -> tuple[float, float, float, str]:
                changed = _aggregate_changed([*selected, candidate], commit_confidence)
                tp = sum(
                    truth_changed.get(object_id) == operation
                    for object_id, (operation, _) in changed.items()
                )
                fp = len(changed) - tp
                fn = len(truth_changed) - tp
                precision = tp / (tp + fp) if tp + fp else 0.0
                recall = tp / (tp + fn) if tp + fn else 0.0
                f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
                false_edit_rate = fp / max(1, all_object_count)
                missed_edit_rate = fn / max(1, len(truth_changed))
                utility = f1 - 0.5 * false_edit_rate - 0.5 * missed_edit_rate
                return (utility, f1, -false_edit_rate, candidate.evidence_id)

            best = max(remaining, key=score)
            selected.append(best)
            remaining.remove(best)
        return selected
    if policy == "learned":
        ranked = learned_rankings.get(episode.episode_id)
        if ranked is None:
            raise KeyError(f"missing learned ranking for {episode.episode_id}")
        return [evidence_by_id[evidence_id] for evidence_id in ranked[:budget]]
    raise ValueError(policy)


def _aggregate_changed(
    selected: list[Any], commit_confidence: float
) -> dict[str, tuple[EditOperation, float]]:
    changed: dict[str, tuple[EditOperation, float]] = {}
    for evidence in selected:
        for proposal in evidence.proposals:
            operation = EditOperation(str(proposal["operation"]))
            confidence = _joint_confidence(proposal)
            object_id = proposal.get("object_id")
            if (
                operation == EditOperation.KEEP
                or confidence < commit_confidence
                or not object_id
            ):
                continue
            key = str(object_id)
            if key not in changed or confidence > changed[key][1]:
                changed[key] = (operation, confidence)
    return changed


def _build_samples(episodes: list[ArgoTweakNativeEpisode], path: Path) -> list[StructuredMapSample]:
    samples: list[StructuredMapSample] = []
    for episode in episodes:
        edits, counts = derive_structured_atomic_edits(
            Path(episode.prior_map_path), Path(episode.target_map_path), include_keep=False
        )
        samples.append(
            StructuredMapSample(
                schema_version="activemap-structured-map-sample-v1",
                sample_id=f"argotweak:{episode.segment_id}",
                dataset="argotweak",
                split=episode.split,
                aoi_id=episode.segment_id,
                prior_map_path=episode.prior_map_path,
                target_map_path=episode.target_map_path,
                observations=[
                    StructuredMapObservation(
                        observation_id=row.evidence_id,
                        timestamp=row.timestamp,
                        modality="camera_bundle",
                        path=row.camera_bundle_path,
                        cost=row.cost,
                        metadata={},
                    )
                    for row in episode.evidence
                ],
                atomic_edits=edits,
                native_sample_id=episode.segment_id,
                test_assets_read=False,
                metadata={f"count_{key.lower()}": value for key, value in counts.items()},
            )
        )
    write_structured_map_jsonl(samples, path)
    return samples


def _policy_predictions(
    episodes: list[ArgoTweakNativeEpisode],
    *,
    policy: str,
    budget: int,
    rankings: dict[str, list[str]],
    commit_confidence: float,
    source_artifact: str,
) -> tuple[list[StructuredMapPrediction], dict[str, float | int]]:
    predictions: list[StructuredMapPrediction] = []
    acquired = proposed = committed = blocked_unassigned = blocked_geometry = 0
    for episode in episodes:
        selected = _selection(
            episode, policy, budget, rankings, commit_confidence
        )
        acquired += len(selected)
        best_by_object: dict[str, tuple[float, dict[str, Any], Any]] = {}
        for evidence in selected:
            for proposal in evidence.proposals:
                operation = EditOperation(str(proposal["operation"]))
                if operation == EditOperation.KEEP:
                    continue
                proposed += 1
                confidence = _joint_confidence(proposal)
                if confidence < commit_confidence:
                    continue
                object_id = proposal.get("object_id")
                if not object_id:
                    blocked_unassigned += 1
                    continue
                key = str(object_id)
                candidate = (confidence, proposal, evidence)
                if key not in best_by_object or candidate[0] > best_by_object[key][0]:
                    best_by_object[key] = candidate

        sample_id = f"argotweak:{episode.segment_id}"
        prior = _load_feature_map(Path(episode.prior_map_path))
        changed: dict[str, StructuredMapPrediction] = {}
        for object_id, (confidence, proposal, evidence) in best_by_object.items():
            operation = EditOperation(str(proposal["operation"]))
            geometry = None
            if operation in {EditOperation.ADD, EditOperation.RESHAPE}:
                try:
                    geometry = proposal_city_geometry(proposal, evidence.city_se3_egovehicle)
                except (TypeError, ValueError):
                    blocked_geometry += 1
                    continue
            changed[object_id] = StructuredMapPrediction(
                schema_version="activemap-structured-map-prediction-v1",
                sample_id=sample_id,
                split=episode.split,
                dataset="argotweak",
                baseline=f"{policy}-top{budget}",
                object_id=object_id,
                operation=operation,
                confidence=confidence,
                geometry=geometry,
                source_artifact=source_artifact,
                test_assets_read=False,
            )
            committed += 1
        predictions.extend(changed.values())
        for object_id in sorted(set(prior) - set(changed)):
            predictions.append(
                StructuredMapPrediction(
                    schema_version="activemap-structured-map-prediction-v1",
                    sample_id=sample_id,
                    split=episode.split,
                    dataset="argotweak",
                    baseline=f"{policy}-top{budget}",
                    object_id=object_id,
                    operation=EditOperation.KEEP,
                    confidence=1.0,
                    source_artifact=source_artifact,
                    test_assets_read=False,
                )
            )
    count = len(episodes)
    return predictions, {
        "mean_acquired_evidence": acquired / count,
        "proposal_count": proposed,
        "committed_edit_count": committed,
        "blocked_unassigned_count": blocked_unassigned,
        "blocked_geometry_count": blocked_geometry,
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if not 0.0 <= args.commit_confidence <= 1.0:
        raise ValueError("commit confidence must be between zero and one")
    args.output_dir.mkdir(parents=True)
    episodes = _read_episodes(args.episodes)
    rankings = _read_rankings(args.rankings)
    samples_path = args.output_dir / "samples.jsonl"
    _build_samples(episodes, samples_path)

    budgets = sorted({int(value) for value in args.budgets.split(",") if value.strip()})
    policies: list[tuple[str, int]] = [("prior", 0), ("direct_center", 1), ("acquire_all", 0)]
    for policy in ("observable", "operation_oracle", "map_oracle"):
        policies.extend((policy, budget) for budget in budgets)
    if rankings:
        policies.extend(("learned", budget) for budget in budgets)

    summary: list[dict[str, Any]] = []
    for policy, budget in policies:
        label = policy if policy in {"prior", "direct_center", "acquire_all"} else f"{policy}_top{budget}"
        output = args.output_dir / label
        predictions, accounting = _policy_predictions(
            episodes,
            policy=policy,
            budget=budget,
            rankings=rankings,
            commit_confidence=args.commit_confidence,
            source_artifact=str(args.episodes.resolve()),
        )
        output.mkdir()
        prediction_path = output / "predictions.jsonl"
        write_structured_prediction_jsonl(predictions, prediction_path)
        metrics = evaluate_structured_map_predictions(samples_path, prediction_path)
        edit = metrics["atomic_edit"]
        quality = float(edit["f1"])
        false_edit = float(metrics["false_edit_rate"])
        missed_edit = 1.0 - float(edit["recall"])
        cost = float(accounting["mean_acquired_evidence"])
        balanced_utility = (
            quality
            - args.false_edit_weight * false_edit
            - args.missed_edit_weight * missed_edit
            - args.cost_weight * cost
        )
        result = {
            "policy": label,
            "commit_confidence": args.commit_confidence,
            **accounting,
            "atomic_edit_f1": quality,
            "atomic_edit_precision": edit["precision"],
            "atomic_edit_recall": edit["recall"],
            "false_edit_rate": false_edit,
            "false_edit_discovery_rate": metrics["false_edit_discovery_rate"],
            "missed_edit_rate": missed_edit,
            "unchanged_preservation": metrics["unchanged_preservation"],
            "geometry_chamfer": metrics["geometry_chamfer"],
            "topology_f1": metrics["topology"]["f1"],
            "balanced_utility": balanced_utility,
        }
        (output / "metrics.json").write_text(json.dumps({**metrics, "accounting": accounting, "balanced_utility": balanced_utility}, indent=2) + "\n", encoding="utf-8")
        summary.append(result)

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "summary.jsonl").open("x", encoding="utf-8") as handle:
        for row in summary:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
