#!/usr/bin/env python3
"""Filter selector supervision to evidence observable at the decision timestamp."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from activemap.models import EpisodeRecord
from activemap.selector_records import SelectorSample

TEMPORAL_CONTRACT = "causal-evidence-at-anchor-v1"


def month_index(timestamp: str) -> int:
    year, month = (int(value) for value in timestamp.replace("-", "_").split("_")[:2])
    return year * 12 + month


def load_episode_index(paths: list[Path]) -> dict[str, EpisodeRecord]:
    episodes: dict[str, EpisodeRecord] = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    episode = EpisodeRecord.model_validate_json(line)
                except Exception as exc:
                    raise ValueError(f"invalid episode at {path}:{line_number}") from exc
                if episode.split == "test":
                    raise ValueError("causal selector construction forbids test episodes")
                if episode.episode_id in episodes:
                    raise ValueError(f"duplicate episode id: {episode.episode_id}")
                episodes[episode.episode_id] = episode
    if not episodes:
        raise ValueError("episode index is empty")
    return episodes


def filter_sample(
    sample: SelectorSample,
    episode: EpisodeRecord,
) -> tuple[SelectorSample | None, dict[str, Any]]:
    if sample.split == "test" or sample.metadata.get("test_assets_read") is True:
        raise ValueError("causal selector construction forbids test records")
    anchor = episode.anchor_timestamp or episode.evidence_catalog[-1].timestamp
    anchor_month = month_index(anchor)
    catalog = {item.evidence_id: item for item in episode.evidence_catalog}
    unknown = [evidence_id for evidence_id in sample.evidence_ids if evidence_id not in catalog]
    if unknown:
        raise ValueError(f"{sample.sample_id} has unknown evidence ids: {unknown[:3]}")
    indices = [
        index
        for index, evidence_id in enumerate(sample.evidence_ids)
        if month_index(catalog[evidence_id].timestamp) <= anchor_month
    ]
    before_target = sample.target_index()
    before_action = "STOP" if before_target == len(sample.evidence_ids) else "ACQUIRE"
    if not indices:
        return None, {
            "before_action": before_action,
            "after_action": "DROPPED_TERMINAL_ONLY",
            "candidate_count_before": len(sample.evidence_ids),
            "candidate_count_after": 0,
        }
    metadata = dict(sample.metadata)
    metadata["candidate_temporal_contract"] = {
        "version": TEMPORAL_CONTRACT,
        "anchor_timestamp": str(anchor),
        "candidate_count_before": len(sample.evidence_ids),
        "candidate_count_after": len(indices),
        "future_candidate_count": len(sample.evidence_ids) - len(indices),
    }
    metadata["runtime_temporally_causal"] = True
    filtered = sample.model_copy(
        update={
            "evidence_ids": [sample.evidence_ids[index] for index in indices],
            "evidence_features": [sample.evidence_features[index] for index in indices],
            "evidence_costs": [sample.evidence_costs[index] for index in indices],
            "false_edit_risks": [sample.false_edit_risks[index] for index in indices],
            "oracle_utilities": [sample.oracle_utilities[index] for index in indices],
            "metadata": metadata,
        }
    )
    after_target = filtered.target_index()
    after_action = "STOP" if after_target == len(filtered.evidence_ids) else "ACQUIRE"
    return filtered, {
        "before_action": before_action,
        "after_action": after_action,
        "candidate_count_before": len(sample.evidence_ids),
        "candidate_count_after": len(filtered.evidence_ids),
    }


def build_manifest(
    source: Path,
    episodes: list[Path],
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    episode_index = load_episode_index(episodes)
    counts: Counter[str] = Counter()
    before_counts: list[int] = []
    after_counts: list[int] = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open(encoding="utf-8") as handle, output.open("x", encoding="utf-8") as out:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                sample = SelectorSample.model_validate_json(line)
            except Exception as exc:
                raise ValueError(f"invalid selector sample at line {line_number}") from exc
            source_episode = str(sample.metadata["source_episode"])
            episode = episode_index.get(source_episode)
            if episode is None:
                raise ValueError(f"missing episode: {source_episode}")
            filtered, audit = filter_sample(sample, episode)
            counts[f"before:{audit['before_action']}"] += 1
            counts[f"after:{audit['after_action']}"] += 1
            before_counts.append(int(audit["candidate_count_before"]))
            after_counts.append(int(audit["candidate_count_after"]))
            if filtered is None:
                continue
            counts[f"split:{filtered.split}"] += 1
            out.write(filtered.model_dump_json() + "\n")
    retained = counts["split:train"] + counts["split:val"]
    if retained == 0:
        output.unlink(missing_ok=True)
        raise ValueError("causal selector manifest retained no train/validation samples")
    summary = {
        "schema_version": "activemap-causal-selector-manifest-v1",
        "temporal_contract": TEMPORAL_CONTRACT,
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "record_count": retained,
        "counts": dict(sorted(counts.items())),
        "candidate_count": {
            "mean_before": sum(before_counts) / len(before_counts),
            "mean_after": sum(after_counts) / len(after_counts),
            "dropped_terminal_only": counts["after:DROPPED_TERMINAL_ONLY"],
        },
        "test_assets_read": False,
    }
    output.with_suffix(output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--episodes", action="append", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_manifest(args.source, args.episodes, args.output), indent=2))


if __name__ == "__main__":
    main()
