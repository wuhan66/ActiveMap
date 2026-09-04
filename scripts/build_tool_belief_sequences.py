#!/usr/bin/env python3
"""Recompose grounded tool records into cumulative three-step belief sequences."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

from activemap.agent.identifiers import public_evidence_id, public_task_id
from activemap.agent.tool_belief_data import (
    ToolBeliefExample,
    ToolBeliefSequenceExample,
    ToolBeliefSequenceStep,
)
from activemap.agent.tools import CounterfactualBeliefUpdater
from activemap.geo_tools.records import GeoToolName
from activemap.models import EpisodeRecord
from activemap.selector_records import SelectorSample
from scripts.build_muno21_tool_belief_data import _initial_samples, _read_jsonl


def _probability_l1(before: list[float], after: list[float]) -> float:
    return math.fsum(abs(left - right) for left, right in zip(before, after, strict=True))


def build_sequences(
    rows: list[ToolBeliefExample],
    episodes: list[EpisodeRecord],
    initial_samples: dict[str, SelectorSample],
) -> tuple[list[ToolBeliefSequenceExample], dict[str, Any]]:
    splits = {episode.split for episode in episodes}
    if not splits or not splits <= {"train", "val", "test"}:
        raise ValueError(f"invalid episode splits: {sorted(splits)}")
    test_assets_read = "test" in splits
    if test_assets_read:
        if splits != {"test"}:
            raise ValueError("belief sequences must not mix test with train/val")
        from activemap.frozen_test import assert_frozen_test_access

        assert_frozen_test_access()
    public_episodes = {public_task_id(episode.episode_id): episode for episode in episodes}
    if len(public_episodes) != len(episodes):
        raise ValueError("public episode identifier collision")
    grouped: OrderedDict[
        str, OrderedDict[str, dict[GeoToolName, ToolBeliefExample]]
    ] = OrderedDict()
    for row in rows:
        by_evidence = grouped.setdefault(row.episode_id, OrderedDict()).setdefault(
            row.evidence_id, {}
        )
        if row.tool_result.tool in by_evidence:
            raise ValueError(
                f"duplicate tool record for {row.episode_id}/{row.evidence_id}: "
                f"{row.tool_result.tool.value}"
            )
        by_evidence[row.tool_result.tool] = row

    sequences = []
    counts: Counter[str] = Counter()
    cumulative_deltas = []
    for public_episode_id, evidence_groups in grouped.items():
        episode = public_episodes.get(public_episode_id)
        if episode is None:
            raise ValueError(f"grounded record has no source episode: {public_episode_id}")
        sample = initial_samples.get(episode.episode_id)
        if sample is None:
            raise ValueError(f"missing initial selector state for {episode.episode_id}")
        if sample.split != episode.split:
            raise ValueError(f"split mismatch for {episode.episode_id}")
        initial_id = str(sample.metadata["initial_evidence_id"])
        updater = CounterfactualBeliefUpdater(sample)
        initial_belief = updater.fuse([initial_id])
        public_to_raw = {
            public_evidence_id(item.evidence_id): item.evidence_id
            for item in episode.evidence_catalog
            if item.evidence_id != initial_id
        }
        if len(public_to_raw) != len(episode.evidence_catalog) - 1:
            raise ValueError(f"public evidence identifier collision for {episode.episode_id}")
        selected = [initial_id]
        steps = []
        required = {GeoToolName.IMAGE_QUALITY, GeoToolName.TEMPORAL_CHANGE}
        for public_evidence, tool_rows in evidence_groups.items():
            if set(tool_rows) != required:
                raise ValueError(
                    f"incomplete tools for {public_episode_id}/{public_evidence}: "
                    f"{sorted(item.value for item in tool_rows)}"
                )
            raw_evidence = public_to_raw.get(public_evidence)
            if raw_evidence is None:
                raise ValueError(f"unknown public evidence: {public_episode_id}/{public_evidence}")
            quality = tool_rows[GeoToolName.IMAGE_QUALITY]
            temporal = tool_rows[GeoToolName.TEMPORAL_CHANGE]
            if quality.prior_belief != initial_belief or temporal.prior_belief != initial_belief:
                raise ValueError(f"flat-record prior mismatch for {public_episode_id}")
            selected.append(raw_evidence)
            cumulative_target = updater.fuse(selected)
            steps.append(
                ToolBeliefSequenceStep(
                    evidence_id=public_evidence,
                    quality_result=quality.tool_result,
                    temporal_result=temporal.tool_result,
                    individual_target_belief=temporal.target_belief,
                    cumulative_target_belief=cumulative_target,
                )
            )
            cumulative_deltas.append(
                _probability_l1(
                    initial_belief.edit_probabilities,
                    cumulative_target.edit_probabilities,
                )
            )
        sequence_key = f"{episode.episode_id}|recurrent-tool-belief-v1"
        sequences.append(
            ToolBeliefSequenceExample(
                sequence_id=hashlib.sha256(sequence_key.encode()).hexdigest()[:24],
                episode_id=public_episode_id,
                split=episode.split,
                initial_belief=initial_belief,
                steps=steps,
                gt_edit=episode.gt_edit.op,
                metadata={
                    "teacher": "frozen_counterfactual_updater_cumulative_fusion",
                    "step_count": len(steps),
                    "test_assets_read": test_assets_read,
                },
            )
        )
        counts[f"split:{episode.split}"] += 1
        counts[f"gt:{episode.gt_edit.op.value}"] += 1

    summary = {
        "schema_version": "tool-belief-sequence-v1",
        "sequence_count": len(sequences),
        "unique_sequence_count": len({item.sequence_id for item in sequences}),
        "step_count": sum(len(item.steps) for item in sequences),
        "counts": dict(sorted(counts.items())),
        "cumulative_probability_l1": {
            "minimum": min(cumulative_deltas),
            "maximum": max(cumulative_deltas),
            "mean": math.fsum(cumulative_deltas) / len(cumulative_deltas),
        },
        "test_assets_read": test_assets_read,
    }
    return sequences, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("flat_data_root", type=Path)
    parser.add_argument("episodes_jsonl", type=Path)
    parser.add_argument("selector_states_jsonl", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--frozen-test", action="store_true")
    args = parser.parse_args()
    requested_splits = {
        value.strip() for value in args.splits.split(",") if value.strip()
    }
    if not requested_splits or not requested_splits <= {"train", "val", "test"}:
        raise ValueError("--splits must contain train, val, or test")
    if "test" in requested_splits:
        if requested_splits != {"test"} or not args.frozen_test:
            raise PermissionError("test sequences require test-only --frozen-test")
        from activemap.frozen_test import assert_frozen_test_access

        assert_frozen_test_access()
        if args.output_root.exists():
            raise FileExistsError(
                f"refusing to overwrite frozen test sequences: {args.output_root}"
            )
    elif args.frozen_test:
        raise ValueError("--frozen-test is valid only for test-only sequences")
    rows = []
    for split in sorted(requested_splits):
        rows.extend(_read_jsonl(args.flat_data_root / f"{split}.jsonl", ToolBeliefExample))
    episodes = [
        row
        for row in _read_jsonl(args.episodes_jsonl, EpisodeRecord)
        if row.split in requested_splits
    ]
    samples = [
        row
        for row in _read_jsonl(args.selector_states_jsonl, SelectorSample)
        if row.split in requested_splits
    ]
    sequences, summary = build_sequences(rows, episodes, _initial_samples(samples))
    args.output_root.mkdir(parents=True, exist_ok=True)
    handles = {
        split: (args.output_root / f"{split}.jsonl").open("w", encoding="utf-8")
        for split in sorted(requested_splits)
    }
    try:
        for sequence in sequences:
            handles[sequence.split].write(sequence.model_dump_json() + "\n")
    finally:
        for handle in handles.values():
            handle.close()
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
