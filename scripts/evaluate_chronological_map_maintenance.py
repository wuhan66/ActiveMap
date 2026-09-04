#!/usr/bin/env python3
"""Evaluate persistent editable-map state over real chronological SN7 chains."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry

from activemap.agent.identifiers import public_task_id
from activemap.agent.writeback import geometry_iou
from activemap.models import EpisodeRecord


def _geometry(value: Any) -> BaseGeometry | None:
    if value is None:
        return None
    payload = value.model_dump() if hasattr(value, "model_dump") else value
    result = shape(payload)
    if result.is_empty:
        return None
    return result if result.is_valid else result.buffer(0)


def _apply_delta(
    prior: BaseGeometry | None,
    add_payload: dict[str, Any] | None,
    remove_payload: dict[str, Any] | None,
) -> BaseGeometry | None:
    current = prior
    added = _geometry(add_payload)
    removed = _geometry(remove_payload)
    if added is not None:
        current = added if current is None else current.union(added)
    if current is not None and removed is not None:
        current = current.difference(removed)
    if current is None or current.is_empty:
        return None
    return current if current.is_valid else current.buffer(0)


def _same_state(left: BaseGeometry | None, right: BaseGeometry | None, tolerance: float) -> bool:
    return geometry_iou(left, right) >= 1.0 - tolerance


def load_inputs(
    episodes_path: Path,
    writeback_path: Path,
    *,
    split: str,
    budget: float,
) -> list[tuple[EpisodeRecord, dict[str, Any]]]:
    episodes = {}
    for line in episodes_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        episode = EpisodeRecord.model_validate_json(line)
        if episode.split == split:
            episodes[public_task_id(episode.episode_id)] = episode
    rows = []
    identities = set()
    for line in writeback_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split") != split or abs(float(row["budget"]) - budget) > 1e-8:
            continue
        if bool(row.get("test_assets_read")) != (split == "test"):
            raise ValueError("writeback test-access flag disagrees with split")
        task_id = str(row["task_id"])
        if task_id not in episodes:
            raise ValueError(f"writeback task is absent from episode file: {task_id}")
        if task_id in identities:
            raise ValueError(f"duplicate writeback task at budget {budget}: {task_id}")
        identities.add(task_id)
        rows.append((episodes[task_id], row))
    if not rows:
        raise ValueError(f"no {split} writebacks at budget {budget:g}")
    return rows


def build_contiguous_chains(
    inputs: list[tuple[EpisodeRecord, dict[str, Any]]],
    *,
    continuity_tolerance: float,
    minimum_length: int,
) -> list[list[tuple[EpisodeRecord, dict[str, Any]]]]:
    grouped: dict[tuple[str, str], list[tuple[EpisodeRecord, dict[str, Any]]]] = defaultdict(list)
    for episode, row in inputs:
        timestamp = episode.anchor_timestamp
        object_id = episode.hypothesis.object_id
        if timestamp is None or episode.aoi_id is None or not object_id:
            continue
        grouped[(episode.aoi_id, object_id)].append((episode, row))

    chains = []
    for values in grouped.values():
        values.sort(key=lambda item: (str(item[0].anchor_timestamp), item[0].episode_id))
        current = []
        previous_target = None
        previous_timestamp = None
        for item in values:
            episode = item[0]
            prior = _geometry(episode.prior_geometry)
            timestamp = str(episode.anchor_timestamp)
            continuous = (
                current
                and timestamp > str(previous_timestamp)
                and _same_state(previous_target, prior, continuity_tolerance)
            )
            if not continuous:
                if len(current) >= minimum_length:
                    chains.append(current)
                current = []
            current.append(item)
            previous_target = _geometry(episode.target_geometry)
            previous_timestamp = timestamp
        if len(current) >= minimum_length:
            chains.append(current)
    return chains


def evaluate(
    chains: list[list[tuple[EpisodeRecord, dict[str, Any]]]],
    *,
    confidence_threshold: float,
    replay_iou_threshold: float,
) -> list[dict[str, Any]]:
    records = []
    for chain_index, chain in enumerate(chains):
        carry_state = _geometry(chain[0][0].prior_geometry)
        gated_state = carry_state
        for step, (episode, row) in enumerate(chain):
            oracle_prior = _geometry(episode.prior_geometry)
            target = _geometry(episode.target_geometry)
            add_payload = row.get("predicted_add_geometry")
            remove_payload = row.get("predicted_remove_geometry")
            independent_state = _apply_delta(oracle_prior, add_payload, remove_payload)
            carry_state = _apply_delta(carry_state, add_payload, remove_payload)
            requests_edit = bool(row.get("writeback_changed"))
            gate_passed = (
                not requests_edit
                or (
                    float(row.get("fused_confidence", 0.0)) >= confidence_threshold
                    and float(row.get("vector_replay_iou", 0.0)) >= replay_iou_threshold
                    and bool(row.get("vector_delta_topology_valid"))
                )
            )
            if gate_passed:
                gated_state = _apply_delta(gated_state, add_payload, remove_payload)
            records.append(
                {
                    "chain_id": f"chain-{chain_index:05d}",
                    "step": step,
                    "task_id": row["task_id"],
                    "aoi_id": episode.aoi_id,
                    "object_id": episode.hypothesis.object_id,
                    "timestamp": episode.anchor_timestamp,
                    "target_edit": episode.gt_edit.op.value,
                    "budget": float(row["budget"]),
                    "independent_iou": geometry_iou(independent_state, target),
                    "carry_iou": geometry_iou(carry_state, target),
                    "risk_gated_iou": geometry_iou(gated_state, target),
                    "prior_iou": geometry_iou(oracle_prior, target),
                    "carry_error_propagation": (
                        geometry_iou(carry_state, target)
                        - geometry_iou(independent_state, target)
                    ),
                    "risk_gated_error_propagation": (
                        geometry_iou(gated_state, target)
                        - geometry_iou(independent_state, target)
                    ),
                    "requested_intervention": requests_edit,
                    "risk_gate_passed": gate_passed,
                    "risk_gate_intervened": requests_edit and gate_passed,
                    "fused_confidence": float(row.get("fused_confidence", 0.0)),
                    "vector_replay_iou": float(row.get("vector_replay_iou", 0.0)),
                    "topology_valid": bool(row.get("vector_delta_topology_valid")),
                    "geometries": {
                        "target": mapping(target) if target is not None else None,
                        "independent": (
                            mapping(independent_state)
                            if independent_state is not None
                            else None
                        ),
                        "carry": mapping(carry_state) if carry_state is not None else None,
                        "risk_gated": (
                            mapping(gated_state) if gated_state is not None else None
                        ),
                    },
                    "test_assets_read": bool(row.get("test_assets_read")),
                }
            )
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, float | int]:
    if not records:
        raise ValueError("no chronological records")
    final = {}
    for row in records:
        final[row["chain_id"]] = row
    final_rows = list(final.values())
    return {
        "chain_count": len(final_rows),
        "transition_count": len(records),
        "aoi_count": len({row["aoi_id"] for row in records}),
        "mean_chain_length": float(len(records) / len(final_rows)),
        "mean_independent_iou": float(np.mean([row["independent_iou"] for row in records])),
        "mean_carry_iou": float(np.mean([row["carry_iou"] for row in records])),
        "mean_risk_gated_iou": float(np.mean([row["risk_gated_iou"] for row in records])),
        "final_independent_iou": float(np.mean([row["independent_iou"] for row in final_rows])),
        "final_carry_iou": float(np.mean([row["carry_iou"] for row in final_rows])),
        "final_risk_gated_iou": float(np.mean([row["risk_gated_iou"] for row in final_rows])),
        "mean_carry_error_propagation": float(
            np.mean([row["carry_error_propagation"] for row in records])
        ),
        "mean_risk_gated_error_propagation": float(
            np.mean([row["risk_gated_error_propagation"] for row in records])
        ),
        "requested_interventions": sum(row["requested_intervention"] for row in records),
        "risk_gated_interventions": sum(row["risk_gate_intervened"] for row in records),
        "risk_gate_rejections": sum(
            row["requested_intervention"] and not row["risk_gate_passed"] for row in records
        ),
    }


def grouped_bootstrap(
    records: list[dict[str, Any]], repetitions: int, seed: int
) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[str(row["aoi_id"])].append(row)
    group_ids = sorted(groups)
    rng = random.Random(seed)
    samples = {"risk_gated_minus_carry_iou": [], "carry_minus_independent_iou": []}
    for _ in range(repetitions):
        selected = [
            row
            for group_id in rng.choices(group_ids, k=len(group_ids))
            for row in groups[group_id]
        ]
        samples["risk_gated_minus_carry_iou"].append(
            float(np.mean([row["risk_gated_iou"] - row["carry_iou"] for row in selected]))
        )
        samples["carry_minus_independent_iou"].append(
            float(np.mean([row["carry_iou"] - row["independent_iou"] for row in selected]))
        )
    return {
        name: {
            "mean": float(np.mean(values)),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
        }
        for name, values in samples.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=Path)
    parser.add_argument("writeback", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--budget", type=float, required=True)
    parser.add_argument("--minimum-chain-length", type=int, default=2)
    parser.add_argument("--continuity-tolerance", type=float, default=1e-6)
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument("--replay-iou-threshold", type=float, default=0.99)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    if args.split == "test":
        from activemap.frozen_test import assert_frozen_test_access

        assert_frozen_test_access()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    inputs = load_inputs(
        args.episodes, args.writeback, split=args.split, budget=args.budget
    )
    chains = build_contiguous_chains(
        inputs,
        continuity_tolerance=args.continuity_tolerance,
        minimum_length=args.minimum_chain_length,
    )
    if not chains:
        raise ValueError("no real contiguous object chains on selected support")
    records = evaluate(
        chains,
        confidence_threshold=args.confidence_threshold,
        replay_iou_threshold=args.replay_iou_threshold,
    )
    args.output_dir.mkdir(parents=True)
    trace_path = args.output_dir / "chronological_traces.jsonl"
    trace_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in records),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "activemap-chronological-maintenance-v1",
        "split": args.split,
        "test_assets_read": args.split == "test",
        "budget": args.budget,
        "protocol": {
            "state_unit": "AOI-object persistent editable geometry",
            "ordering": "episode.anchor_timestamp",
            "continuity": "previous target geometry equals next prior geometry",
            "branches": ["independent_reset", "unconditional_carry", "risk_gated_carry"],
            "risk_gate_inputs": [
                "updater fused confidence",
                "vector replay consistency",
                "topology validity",
            ],
            "confidence_threshold": args.confidence_threshold,
            "replay_iou_threshold": args.replay_iou_threshold,
            "uses_outcome_labels_for_gate": False,
            "online_weight_update": False,
        },
        "metrics": summarize(records),
        "aoi_bootstrap": (
            grouped_bootstrap(records, args.bootstrap_repetitions, args.seed)
            if args.bootstrap_repetitions > 0
            else None
        ),
        "sources": {
            "episodes": str(args.episodes.resolve()),
            "writeback": str(args.writeback.resolve()),
            "trace": str(trace_path.resolve()),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
