#!/usr/bin/env python3
"""Execute grounded tools on model-selected Active-Catalog evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm

from activemap.agent.active_catalog_joint import ActiveCatalogJointTransition
from activemap.agent.identifiers import public_evidence_id, public_task_id
from activemap.agent.records import AgentBelief
from activemap.agent.tool_belief_data import PostAcquisitionToolPairExample
from activemap.agent.tool_features import TOOL_RESULT_FEATURE_NAMES, encode_tool_result
from activemap.geo_tools.records import GeoToolCall, GeoToolName, GeoToolResult
from activemap.geo_tools.registry import GeoToolRegistry
from activemap.models import EditOperation, EpisodeRecord


def _read(path: Path, model: type[Any]) -> list[Any]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(model.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"invalid {path}:{line_number}: {exc}") from exc
    if not rows:
        raise ValueError(f"empty input: {path}")
    return rows


def parse_asset_root_maps(values: list[str]) -> tuple[tuple[Path, Path], ...]:
    mappings = []
    for value in values:
        if "=" not in value:
            raise ValueError("asset root maps must use SOURCE=TARGET")
        source_text, target_text = value.split("=", 1)
        source, target = Path(source_text), Path(target_text)
        if not source.is_absolute() or not target.is_absolute():
            raise ValueError("asset root maps must contain absolute paths")
        mappings.append((source, target))
    return tuple(mappings)


def remap_asset_path(
    path: str | Path, mappings: tuple[tuple[Path, Path], ...]
) -> Path:
    original = Path(path)
    for source, target in mappings:
        try:
            return target / original.relative_to(source)
        except ValueError:
            continue
    return original


def _target_belief(
    prior: AgentBelief, target: EditOperation, smoothing: float
) -> AgentBelief:
    probabilities = [smoothing / 3.0] * 4
    probabilities[list(EditOperation).index(target)] = 1.0 - smoothing
    entropy = -math.fsum(
        value * math.log(max(value, 1e-12)) for value in probabilities
    ) / math.log(len(probabilities))
    return AgentBelief(
        edit_probabilities=probabilities,
        confidence=prior.confidence,
        geometry_delta=prior.geometry_delta,
        uncertainty=entropy,
        recommended_edit=target,
    )


def _call_id(
    source_episode: str, evidence_id: str, tool: GeoToolName, out_size: int
) -> str:
    payload = f"{source_episode}|{evidence_id}|{tool.value}|{out_size}".encode()
    return f"ac-{hashlib.sha256(payload).hexdigest()[:20]}"


def _window(region: tuple[int, int, int, int], out_size: int) -> dict[str, object]:
    x_min, y_min, x_max, y_max = region
    return {
        "pixel_window": [x_min, y_min, x_max - x_min, y_max - y_min],
        "out_size": [out_size, out_size],
    }


def _execute_pair(
    registry: GeoToolRegistry,
    *,
    source_episode: str,
    evidence_id: str,
    evidence_path: Path,
    anchor_path: Path,
    region: tuple[int, int, int, int],
    out_size: int,
) -> tuple[GeoToolResult, GeoToolResult]:
    parameters = _window(region, out_size)
    quality = registry.execute(
        GeoToolCall(
            call_id=_call_id(
                source_episode, evidence_id, GeoToolName.IMAGE_QUALITY, out_size
            ),
            tool=GeoToolName.IMAGE_QUALITY,
            inputs={"image_path": str(evidence_path)},
            parameters=parameters,
        )
    )
    temporal = registry.execute(
        GeoToolCall(
            call_id=_call_id(
                source_episode, evidence_id, GeoToolName.TEMPORAL_CHANGE, out_size
            ),
            tool=GeoToolName.TEMPORAL_CHANGE,
            inputs={
                "before_path": str(evidence_path),
                "after_path": str(anchor_path),
            },
            parameters=parameters,
        )
    )
    def sanitize(result: GeoToolResult) -> GeoToolResult:
        return result.model_copy(
            update={
                "outputs": {**result.outputs, "evidence_id": evidence_id},
                "artifacts": [],
            }
        )
    return sanitize(quality), sanitize(temporal)


def build_grounded_examples(
    transitions: list[ActiveCatalogJointTransition],
    episodes: list[EpisodeRecord],
    *,
    artifact_root: Path,
    out_size: int,
    label_smoothing: float,
    asset_root_maps: tuple[tuple[Path, Path], ...] = (),
    registry: GeoToolRegistry | None = None,
) -> tuple[list[PostAcquisitionToolPairExample], dict[str, Any]]:
    if out_size <= 0:
        raise ValueError("out_size must be positive")
    if not 0.0 < label_smoothing < 0.25:
        raise ValueError("label_smoothing must be between zero and 0.25")
    splits = {row.split for row in transitions}
    if len(splits) != 1:
        raise ValueError("grounded tool construction requires one split")
    split = next(iter(splits))
    episode_index = {row.episode_id: row for row in episodes if row.split == split}
    if len(episode_index) != len([row for row in episodes if row.split == split]):
        raise ValueError("duplicate source episodes")
    required_episodes = {row.source_episode for row in transitions}
    if not required_episodes <= set(episode_index):
        raise ValueError("joint transitions lack source episode assets")

    if registry is None:
        from activemap.geo_tools.raster import ImageQualityTool, TemporalChangeTool

        registry = GeoToolRegistry()
        registry.register(ImageQualityTool())
        registry.register(TemporalChangeTool(artifact_root / "temporal_change"))
    cache: dict[tuple[str, str], tuple[GeoToolResult, GeoToolResult]] = {}
    output = []
    counts: Counter[str] = Counter()
    failure_reasons: Counter[str] = Counter()
    failed_transition_ids: list[str] = []
    failed_cache_keys: set[tuple[str, str]] = set()
    feature_values = {name: [] for name in TOOL_RESULT_FEATURE_NAMES}
    for row in tqdm(transitions, desc=f"Active-Catalog tools {split}", unit="transition"):
        counts[f"attempted_target:{row.target_edit.value}"] += 1
        episode = episode_index[row.source_episode]
        public_catalog = {
            public_evidence_id(item.evidence_id): item for item in episode.evidence_catalog
        }
        if len(public_catalog) != len(episode.evidence_catalog):
            raise ValueError(f"public evidence collision: {episode.episode_id}")
        evidence = public_catalog.get(row.evidence_id)
        if evidence is None:
            raise ValueError(
                f"unknown acquired evidence: {row.source_episode}/{row.evidence_id}"
            )
        anchor_id = row.prior_observation.selected_evidence_ids[0]
        anchor = public_catalog.get(anchor_id)
        if anchor is None:
            raise ValueError(f"unknown anchor evidence: {row.source_episode}/{anchor_id}")
        key = (row.source_episode, row.evidence_id)
        if key not in cache:
            cache[key] = _execute_pair(
                registry,
                source_episode=row.source_episode,
                evidence_id=row.evidence_id,
                evidence_path=remap_asset_path(evidence.image_path, asset_root_maps),
                anchor_path=remap_asset_path(anchor.image_path, asset_root_maps),
                region=evidence.region,
                out_size=out_size,
            )
        quality, temporal = cache[key]
        if not quality.success or not temporal.success:
            failed_cache_keys.add(key)
            failed_transition_ids.append(row.transition_id)
            counts["skipped_failed_tool_pair"] += 1
            for result in (quality, temporal):
                if not result.success:
                    failure_reasons[
                        f"{result.tool.value}:{result.error or 'unknown error'}"
                    ] += 1
            continue
        prior = row.post_acquisition_observation.belief
        selected_candidate = next(
            (
                candidate
                for candidate in row.prior_observation.candidates
                if candidate.evidence_id == row.evidence_id
            ),
            None,
        )
        if selected_candidate is None:
            raise ValueError("selected evidence is absent from pre-tool candidates")
        identity = f"{row.transition_id}|grounded-tool-pair-v1"
        output.append(
            PostAcquisitionToolPairExample(
                example_id=hashlib.sha256(identity.encode()).hexdigest()[:24],
                task_id=public_task_id(row.source_episode),
                split=row.split,
                evidence_id=row.evidence_id,
                post_acquisition_belief=prior,
                quality_result=quality,
                temporal_result=temporal,
                target_belief=_target_belief(
                    prior, row.target_edit, label_smoothing
                ),
                gt_edit=row.target_edit,
                evidence_cost=row.evidence_cost,
                tool_cost=quality.cost + temporal.cost,
                metadata={
                    "source_transition_id": row.transition_id,
                    "source_policy_snapshot": row.policy_snapshot,
                    "transition_source": "model_selected_executed_rollout",
                    "selected_by_model": True,
                    "oracle_next_state_replay": False,
                    "oracle_action_exported": False,
                    "operation_update_threshold": row.operation_update_threshold,
                    "observable_candidate_features": selected_candidate.features,
                    "pre_acquisition_initial_budget": (
                        row.prior_observation.initial_budget
                    ),
                    "pre_acquisition_remaining_budget": (
                        row.prior_observation.remaining_budget
                    ),
                    "pre_acquisition_spent_cost": row.prior_observation.spent_cost,
                    "pre_acquisition_step": row.prior_observation.step,
                    "label_smoothing": label_smoothing,
                    "test_assets_read": False,
                },
            )
        )
        counts[f"target:{row.target_edit.value}"] += 1
        for result in (quality, temporal):
            counts[f"tool:{result.tool.value}"] += 1
            for name, value in zip(
                TOOL_RESULT_FEATURE_NAMES, encode_tool_result(result), strict=True
            ):
                feature_values[name].append(value)
    if not output:
        raise ValueError("all grounded tool pairs failed")
    failed_id_payload = "\n".join(sorted(failed_transition_ids)).encode()
    summary = {
        "schema_version": "active-catalog-grounded-tool-pair-dataset-v1",
        "split": split,
        "transitions": len(transitions),
        "examples": len(output),
        "skipped_failed_transitions": len(failed_transition_ids),
        "failed_transition_rate": len(failed_transition_ids) / len(transitions),
        "failed_transition_ids_sha256": hashlib.sha256(failed_id_payload).hexdigest(),
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "episodes": len(required_episodes),
        "aois": len({row.aoi_id for row in transitions}),
        "unique_grounded_tool_executions": len(cache),
        "unique_failed_tool_executions": len(failed_cache_keys),
        "tool_calls": 2 * len(cache),
        "cache_reuse_count": len(transitions) - len(cache),
        "counts": dict(sorted(counts.items())),
        "feature_summary": {
            name: {
                "minimum": min(values),
                "maximum": max(values),
                "mean": math.fsum(values) / len(values),
                "unique_count": len(set(values)),
            }
            for name, values in feature_values.items()
            if values
        },
        "out_size": out_size,
        "label_smoothing": label_smoothing,
        "model_selected_state_transitions": True,
        "oracle_next_state_replay": False,
        "oracle_action_exported": False,
        "explicit_geospatial_tool_calls": True,
        "test_assets_read": False,
    }
    return output, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("transitions", type=Path)
    parser.add_argument("episodes", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--split", required=True, choices=("train", "val"))
    parser.add_argument("--out-size", type=int, default=256)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--asset-root-map", action="append", default=[], metavar="SOURCE=TARGET"
    )
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    transitions = [
        row
        for row in _read(args.transitions, ActiveCatalogJointTransition)
        if row.split == args.split
    ]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        transitions = transitions[: args.limit]
    if not transitions:
        raise ValueError(f"no {args.split} transitions")
    episodes = _read(args.episodes, EpisodeRecord)
    examples, summary = build_grounded_examples(
        transitions,
        episodes,
        artifact_root=args.output_root / "artifacts",
        out_size=args.out_size,
        label_smoothing=args.label_smoothing,
        asset_root_maps=parse_asset_root_maps(args.asset_root_map),
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / f"{args.split}.jsonl"
    output.write_text(
        "".join(row.model_dump_json() + "\n" for row in examples), encoding="utf-8"
    )
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
