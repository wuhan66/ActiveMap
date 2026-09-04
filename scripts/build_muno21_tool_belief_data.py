#!/usr/bin/env python3
"""Execute real MUNO21 tools and build grounded belief-update supervision."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm

from activemap.agent.identifiers import public_evidence_id, public_task_id
from activemap.agent.tool_belief_data import ToolBeliefExample
from activemap.agent.tool_features import TOOL_RESULT_FEATURE_NAMES, encode_tool_result
from activemap.agent.tools import CounterfactualBeliefUpdater
from activemap.geo_tools.raster import ImageQualityTool, TemporalChangeTool
from activemap.geo_tools.records import GeoToolCall, GeoToolName, GeoToolResult
from activemap.geo_tools.registry import GeoToolRegistry
from activemap.models import EpisodeRecord
from activemap.selector_records import SelectorSample


def _read_jsonl(path: Path, model: type[Any]) -> list[Any]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(model.model_validate_json(line))
                except Exception as exc:
                    raise ValueError(f"invalid {path}:{line_number}: {exc}") from exc
    return rows


def _initial_samples(samples: list[SelectorSample]) -> dict[str, SelectorSample]:
    selected: dict[str, SelectorSample] = {}
    signatures: dict[str, str] = {}
    for sample in samples:
        metadata = sample.metadata
        episode_id = str(metadata.get("source_episode", ""))
        initial_id = metadata.get("initial_evidence_id")
        selected_ids = metadata.get("selected_evidence_ids")
        if not episode_id or not isinstance(initial_id, str):
            continue
        if selected_ids != [initial_id] or int(metadata.get("oracle_step", -1)) != 0:
            continue
        signature = json.dumps(
            {
                "split": sample.split,
                "gt_edit": metadata.get("gt_edit"),
                "initial_evidence_id": initial_id,
                "evidence_predictions": metadata.get("evidence_predictions"),
            },
            sort_keys=True,
        )
        if episode_id in signatures and signatures[episode_id] != signature:
            raise ValueError(f"inconsistent initial selector states for {episode_id}")
        signatures[episode_id] = signature
        selected.setdefault(episode_id, sample)
    return selected


def _call_id(episode_id: str, evidence_id: str, tool: GeoToolName) -> str:
    payload = f"{episode_id}|{evidence_id}|{tool.value}".encode()
    return f"tb-{hashlib.sha256(payload).hexdigest()[:20]}"


def _window(region: tuple[int, int, int, int], out_size: int) -> dict[str, object]:
    x_min, y_min, x_max, y_max = region
    return {
        "pixel_window": [x_min, y_min, x_max - x_min, y_max - y_min],
        "out_size": [out_size, out_size],
    }


def _execute(
    registry: GeoToolRegistry,
    *,
    episode: EpisodeRecord,
    evidence_id: str,
    initial_id: str,
    tool: GeoToolName,
    out_size: int,
) -> GeoToolResult:
    catalog = {item.evidence_id: item for item in episode.evidence_catalog}
    evidence = catalog[evidence_id]
    initial = catalog[initial_id]
    inputs: dict[str, object]
    if tool == GeoToolName.IMAGE_QUALITY:
        inputs = {"image_path": evidence.image_path}
    elif tool == GeoToolName.TEMPORAL_CHANGE:
        inputs = {
            "before_path": evidence.image_path,
            "after_path": initial.image_path,
        }
    else:  # pragma: no cover - fixed builder contract
        raise ValueError(f"unsupported grounded tool: {tool.value}")
    call = GeoToolCall(
        call_id=_call_id(episode.episode_id, evidence_id, tool),
        tool=tool,
        inputs=inputs,
        parameters=_window(evidence.region, out_size),
    )
    return registry.execute(call)


def build_examples(
    episodes: list[EpisodeRecord],
    initial_samples: dict[str, SelectorSample],
    *,
    artifact_root: Path,
    out_size: int,
) -> tuple[list[ToolBeliefExample], dict[str, object]]:
    splits = {episode.split for episode in episodes}
    if not splits or not splits <= {"train", "val", "test"}:
        raise ValueError(f"invalid episode splits: {sorted(splits)}")
    test_assets_read = "test" in splits
    if test_assets_read:
        if splits != {"test"}:
            raise ValueError("grounded tool construction must not mix test with train/val")
        from activemap.frozen_test import assert_frozen_test_access

        assert_frozen_test_access()
    registry = GeoToolRegistry()
    registry.register(ImageQualityTool())
    registry.register(TemporalChangeTool(artifact_root / "temporal_change"))
    examples: list[ToolBeliefExample] = []
    counts: Counter[str] = Counter()
    feature_values: dict[str, list[float]] = {
        name: [] for name in TOOL_RESULT_FEATURE_NAMES
    }
    teacher_deltas: dict[str, list[float]] = {
        tool.value: [] for tool in (GeoToolName.IMAGE_QUALITY, GeoToolName.TEMPORAL_CHANGE)
    }

    for episode in tqdm(episodes, desc="Grounded MUNO21 tools", unit="episode"):
        sample = initial_samples.get(episode.episode_id)
        if sample is None:
            raise ValueError(f"missing initial selector state for {episode.episode_id}")
        if sample.split != episode.split:
            raise ValueError(f"split mismatch for {episode.episode_id}")
        initial_id = str(sample.metadata["initial_evidence_id"])
        updater = CounterfactualBeliefUpdater(sample)
        prior_belief = updater.fuse([initial_id])

        for evidence in episode.evidence_catalog:
            if evidence.evidence_id == initial_id:
                continue
            target_belief = updater.fuse([initial_id, evidence.evidence_id])
            for tool in (GeoToolName.IMAGE_QUALITY, GeoToolName.TEMPORAL_CHANGE):
                result = _execute(
                    registry,
                    episode=episode,
                    evidence_id=evidence.evidence_id,
                    initial_id=initial_id,
                    tool=tool,
                    out_size=out_size,
                )
                carries_semantic_update = tool == GeoToolName.TEMPORAL_CHANGE
                effective_target = (
                    target_belief
                    if result.success and carries_semantic_update
                    else prior_belief
                )
                record_key = f"{episode.episode_id}|{evidence.evidence_id}|{tool.value}"
                examples.append(
                    ToolBeliefExample(
                        record_id=hashlib.sha256(record_key.encode()).hexdigest()[:24],
                        episode_id=public_task_id(episode.episode_id),
                        split=episode.split,
                        evidence_id=public_evidence_id(evidence.evidence_id),
                        prior_belief=prior_belief,
                        tool_result=result.model_copy(update={"artifacts": []}),
                        target_belief=effective_target,
                        gt_edit=episode.gt_edit.op,
                        metadata={
                            "teacher": "frozen_counterfactual_updater_fusion",
                            "target_kind": (
                                "teacher_belief_update"
                                if carries_semantic_update
                                else "observational_noop"
                            ),
                            "artifact_call_id": result.call_id,
                            "test_assets_read": test_assets_read,
                        },
                    )
                )
                counts[f"split:{episode.split}"] += 1
                counts[f"tool:{tool.value}"] += 1
                counts[f"tool_success:{tool.value}:{result.success}"] += 1
                counts[f"gt:{episode.gt_edit.op.value}"] += 1
                encoded = encode_tool_result(result)
                for name, value in zip(TOOL_RESULT_FEATURE_NAMES, encoded, strict=True):
                    feature_values[name].append(value)
                teacher_deltas[tool.value].append(
                    sum(
                        abs(after - before)
                        for before, after in zip(
                            prior_belief.edit_probabilities,
                            effective_target.edit_probabilities,
                            strict=True,
                        )
                    )
                )

    feature_summary = {}
    for name, values in feature_values.items():
        if values and any(value != 0.0 for value in values):
            feature_summary[name] = {
                "minimum": min(values),
                "maximum": max(values),
                "mean": math.fsum(values) / len(values),
                "unique_count": len(set(values)),
            }
    delta_summary = {
        tool: {
            "minimum": min(values),
            "maximum": max(values),
            "mean": math.fsum(values) / len(values),
        }
        for tool, values in teacher_deltas.items()
        if values
    }

    summary: dict[str, object] = {
        "schema_version": "tool-belief-v1",
        "episode_count": len(episodes),
        "record_count": len(examples),
        "unique_record_count": len({example.record_id for example in examples}),
        "out_size": out_size,
        "counts": dict(sorted(counts.items())),
        "feature_summary": feature_summary,
        "teacher_probability_l1_delta": delta_summary,
        "test_assets_read": test_assets_read,
    }
    return examples, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes_jsonl", type=Path)
    parser.add_argument("selector_states_jsonl", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--out-size", type=int, default=256)
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument("--frozen-test", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--limit-per-class", type=int)
    args = parser.parse_args()
    if args.out_size <= 0:
        raise ValueError("out-size must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    if args.limit_per_class is not None and args.limit_per_class <= 0:
        raise ValueError("limit-per-class must be positive")
    if args.limit is not None and args.limit_per_class is not None:
        raise ValueError("limit and limit-per-class are mutually exclusive")

    if args.split == "test":
        if not args.frozen_test:
            raise PermissionError("test tool data requires --frozen-test")
        from activemap.frozen_test import assert_frozen_test_access

        assert_frozen_test_access()
        if args.output_root.exists():
            raise FileExistsError(
                f"refusing to overwrite frozen test tool data: {args.output_root}"
            )
    elif args.frozen_test:
        raise ValueError("--frozen-test is valid only with --split test")
    episodes = _read_jsonl(args.episodes_jsonl, EpisodeRecord)
    if args.split is not None:
        episodes = [episode for episode in episodes if episode.split == args.split]
    if args.limit_per_class is not None:
        per_class: Counter[str] = Counter()
        selected_episodes = []
        for episode in episodes:
            operation = episode.gt_edit.op.value
            if per_class[operation] < args.limit_per_class:
                selected_episodes.append(episode)
                per_class[operation] += 1
        episodes = selected_episodes
    if args.limit is not None:
        episodes = episodes[: args.limit]
    samples = _read_jsonl(args.selector_states_jsonl, SelectorSample)
    examples, summary = build_examples(
        episodes,
        _initial_samples(samples),
        artifact_root=args.output_root / "artifacts",
        out_size=args.out_size,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    output_splits = sorted({example.split for example in examples})
    paths = {split: args.output_root / f"{split}.jsonl" for split in output_splits}
    handles = {split: path.open("w", encoding="utf-8") for split, path in paths.items()}
    try:
        for example in examples:
            handles[example.split].write(example.model_dump_json() + "\n")
    finally:
        for handle in handles.values():
            handle.close()
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
