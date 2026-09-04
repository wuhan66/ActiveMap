#!/usr/bin/env python3
"""Build active evidence-selection VLM SFT from counterfactual selector states."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter
from shapely.geometry import shape

from activemap.agent.identifiers import public_evidence_id, public_task_id
from activemap.agent.active_catalog import (
    ACTIVE_CATALOG_SYSTEM_PROMPT,
    candidate_payload as _candidate_payload,
    observable_shortlist_indices as _shortlist_indices,
)
from activemap.agent.sequential_controller import (
    ControllerStage,
    SelectionDecision,
    SequentialControllerAction,
)
from activemap.models import EpisodeRecord
from activemap.selector_records import SelectorSample

SYSTEM_PROMPT = ACTIVE_CATALOG_SYSTEM_PROMPT

FORBIDDEN_PROMPT_KEYS = {
    "gt_edit",
    "oracle_utilities",
    "target_operation",
    "policy_relative_advantage",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path, model: type[Any]) -> list[Any]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(model.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"invalid record at {path}:{line_number}: {exc}") from exc
    if not rows:
        raise ValueError(f"empty input: {path}")
    return rows


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _state(
    sample: SelectorSample,
    episode: EpisodeRecord,
    policy_snapshot: str,
    candidate_indices: list[int] | None = None,
) -> dict[str, Any]:
    indices = candidate_indices or list(range(len(sample.evidence_ids)))
    candidates = [
        _candidate_payload(
            sample.evidence_ids[index],
            sample.evidence_features[index],
            sample.evidence_costs[index],
            episode,
        )
        for index in indices
    ]
    selected = [
        public_evidence_id(value)
        for value in sample.metadata.get("selected_evidence_ids", [])
    ]
    state = {
        "controller_stage": ControllerStage.SELECT.value,
        "policy_snapshot": policy_snapshot,
        "direct_draft": {
            "edit": sample.edit_type.value,
            "confidence": round(float(sample.hypothesis_features[13]), 6),
        },
        "belief": {
            "edit_probabilities": [round(float(value), 6) for value in sample.state_features[2:6]],
            "uncertainty": round(float(sample.hypothesis_features[12]), 6),
        },
        "budget": {
            "remaining": round(float(sample.metadata["budget"]), 6),
            "initial_evidence_cost_excluded": bool(
                sample.metadata.get("initial_evidence_cost_excluded", False)
            ),
        },
        "selected_evidence_ids": selected,
        "candidate_evidence": candidates,
        "safety": {"false_edit_risk_limit": 0.02},
    }
    if FORBIDDEN_PROMPT_KEYS.intersection(state):
        raise ValueError("selector state leaks target metadata")
    return state


def _target(
    sample: SelectorSample, candidate_indices: list[int] | None = None
) -> tuple[SequentialControllerAction, float]:
    indices = candidate_indices or list(range(len(sample.evidence_ids)))
    index = max(indices, key=sample.oracle_utilities.__getitem__)
    best_gain = sample.oracle_utilities[index] - sample.stop_utility
    if best_gain <= 0.0:
        return (
            SequentialControllerAction(
                stage=ControllerStage.SELECT,
                selection=SelectionDecision.STOP,
            ),
            best_gain,
        )
    return (
        SequentialControllerAction(
            stage=ControllerStage.SELECT,
            selection=SelectionDecision.ACQUIRE,
            evidence_id=public_evidence_id(sample.evidence_ids[index]),
        ),
        best_gain,
    )


def _messages(
    image_path: Path,
    state: dict[str, Any],
    action: SequentialControllerAction,
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": json.dumps(state, separators=(",", ":"))},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": action.model_dump_json(exclude_none=True)}
            ],
        },
    ]


def _evaluation_index_row(
    sample: SelectorSample,
    *,
    example_id: str,
    task_id: str,
    candidate_indices: list[int],
    action: SequentialControllerAction,
) -> dict[str, Any]:
    candidates = [
        {
            "evidence_id": public_evidence_id(sample.evidence_ids[index]),
            "utility": float(sample.oracle_utilities[index]),
            "cost": float(sample.evidence_costs[index]),
        }
        for index in candidate_indices
    ]
    utility_by_id = {item["evidence_id"]: item["utility"] for item in candidates}
    target_utility = (
        utility_by_id[str(action.evidence_id)]
        if action.selection == SelectionDecision.ACQUIRE
        else float(sample.stop_utility)
    )
    return {
        "example_id": example_id,
        "task_id": task_id,
        "split": sample.split,
        "source_episode": str(sample.metadata["source_episode"]),
        "aoi_id": str(sample.metadata.get("aoi_id", "UNKNOWN")),
        "gt_edit": str(sample.metadata.get("gt_edit", "UNKNOWN")),
        "draft_edit": sample.edit_type.value,
        "budget": float(sample.metadata["budget"]),
        "oracle_step": int(sample.metadata["oracle_step"]),
        "stop_utility": float(sample.stop_utility),
        "target_selection": action.selection.value,
        "target_evidence_id": action.evidence_id,
        "target_utility": target_utility,
        "candidates": candidates,
        "model_visible": False,
        "test_assets_read": False,
    }


def render_initial_state(episode: EpisodeRecord, evidence_id: str, output: Path) -> None:
    from activemap.oracle.updater_counterfactual import _read_candidate

    evidence = next(item for item in episode.evidence_catalog if item.evidence_id == evidence_id)
    context_evidence = max(
        (item for item in episode.evidence_catalog if item.timestamp == evidence.timestamp),
        key=lambda item: (item.scale, item.region),
    )
    prior_geometry = shape(episode.prior_geometry.model_dump()) if episode.prior_geometry else None
    image, _, _, _, _ = _read_candidate(
        evidence,
        prior_geometry=prior_geometry,
        target_geometry=None,
        image_size=384,
        image_channels=3,
    )
    context_image, prior, _, _, _ = _read_candidate(
        context_evidence,
        prior_geometry=prior_geometry,
        target_geometry=None,
        image_size=384,
        image_channels=3,
    )
    rgb = np.moveaxis(image[:3], 0, -1)
    rgb = np.asarray(np.round(np.clip(rgb, 0.0, 1.0) * 255.0), dtype=np.uint8)
    context_rgb = np.moveaxis(context_image[:3], 0, -1)
    context_rgb = np.asarray(
        np.round(np.clip(context_rgb, 0.0, 1.0) * 255.0), dtype=np.uint8
    )
    overlay = context_rgb.astype(np.float32)
    mask = prior >= 0.5
    overlay[mask] = 0.82 * overlay[mask] + 0.18 * np.asarray([0.0, 220.0, 220.0])
    mask_image = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255)
    dilated = np.asarray(mask_image.filter(ImageFilter.MaxFilter(5))) > 0
    eroded = np.asarray(mask_image.filter(ImageFilter.MinFilter(5))) > 0
    boundary = np.logical_and(dilated, np.logical_not(eroded))
    overlay[boundary] = np.asarray([0.0, 255.0, 255.0])
    canvas = Image.new("RGB", (768, 384))
    canvas.paste(Image.fromarray(rgb), (0, 0))
    canvas.paste(Image.fromarray(np.asarray(overlay, dtype=np.uint8)), (384, 0))
    canvas.paste((255, 255, 255), (382, 0, 386, 384))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92, subsampling=0)


def build_dataset(
    states_path: Path,
    episodes_path: Path,
    output_root: Path,
    *,
    policy_snapshot: str,
    stats_only: bool,
    max_candidates: int,
    asset_root_maps: tuple[tuple[Path, Path], ...] = (),
) -> dict[str, Any]:
    from activemap.oracle.updater_counterfactual import remap_episode_assets

    samples = _read_jsonl(states_path, SelectorSample)
    episodes = {
        row.episode_id: row
        for row in remap_episode_assets(
            _read_jsonl(episodes_path, EpisodeRecord), asset_root_maps
        )
    }
    if output_root.exists() and not stats_only:
        raise FileExistsError(f"refusing to overwrite {output_root}")
    counts: Counter[str] = Counter()
    rendered: set[str] = set()
    global_positive_states = 0
    global_positive_in_shortlist = 0
    retrieval_regrets: list[float] = []
    retrieval_by_split = {
        split: {"positive": 0, "included": 0, "regrets": []}
        for split in ("train", "val")
    }
    handles = {}
    evaluation_handles = {}
    if not stats_only:
        output_root.mkdir(parents=True)
        handles = {
            split: (output_root / f"{split}.jsonl").open("x", encoding="utf-8")
            for split in ("train", "val")
        }
        evaluation_handles = {
            split: (output_root / f"{split}_evaluation_index.jsonl").open(
                "x", encoding="utf-8"
            )
            for split in ("train", "val")
        }
        _write_progress(
            output_root / "build.progress.json",
            {
                "status": "running",
                "states_processed": 0,
                "states_total": len(samples),
                "episodes_rendered": 0,
                "test_assets_read": False,
            },
        )
    try:
        iterator = samples
        if not stats_only:
            from tqdm.auto import tqdm

            iterator = tqdm(samples, desc="Active-catalog SFT", unit="state")
        for processed, sample in enumerate(iterator, start=1):
            if sample.split not in {"train", "val"}:
                raise ValueError(f"forbidden selector split: {sample.split}")
            episode_id = str(sample.metadata["source_episode"])
            episode = episodes.get(episode_id)
            if episode is None or episode.split != sample.split:
                raise ValueError(f"missing or split-mismatched episode: {episode_id}")
            candidate_indices = _shortlist_indices(sample, max_candidates)
            global_index = sample.target_index(allow_stop=True)
            global_gain = max(sample.oracle_utilities) - sample.stop_utility
            local_best_gain = (
                max(sample.oracle_utilities[index] for index in candidate_indices)
                - sample.stop_utility
            )
            if global_index != len(sample.evidence_ids):
                global_positive_states += 1
                global_positive_in_shortlist += int(global_index in candidate_indices)
                regret = max(global_gain - max(local_best_gain, 0.0), 0.0)
                retrieval_regrets.append(regret)
                retrieval_by_split[sample.split]["positive"] += 1
                retrieval_by_split[sample.split]["included"] += int(
                    global_index in candidate_indices
                )
                retrieval_by_split[sample.split]["regrets"].append(regret)
            action, advantage = _target(sample, candidate_indices)
            action_name = action.selection.value
            counts[f"split:{sample.split}"] += 1
            counts[f"split:{sample.split}:action:{action_name}"] += 1
            counts[f"action:{action_name}"] += 1
            counts[f"gt:{sample.metadata.get('gt_edit', 'UNKNOWN')}:{action_name}"] += 1
            counts[f"draft:{sample.edit_type.value}:{action_name}"] += 1
            if stats_only:
                continue
            task_id = public_task_id(episode_id)
            image_path = output_root / "images" / f"{task_id}.jpg"
            if episode_id not in rendered:
                render_initial_state(
                    episode, str(sample.metadata["initial_evidence_id"]), image_path
                )
                rendered.add(episode_id)
            state = _state(sample, episode, policy_snapshot, candidate_indices)
            portable_image_path = Path("images") / image_path.name
            example_id = hashlib.sha256(sample.sample_id.encode()).hexdigest()[:24]
            row = {
                "messages": _messages(portable_image_path, state, action),
                "example_id": example_id,
                "task_id": task_id,
                "split": sample.split,
                "stage": ControllerStage.SELECT.value,
                "selected_tool": action.selection == SelectionDecision.ACQUIRE,
                "policy_relative_advantage": advantage,
                "protocol": "active-catalog-sequential-selector-v1",
            }
            handles[sample.split].write(json.dumps(row, separators=(",", ":")) + "\n")
            evaluation_row = _evaluation_index_row(
                sample,
                example_id=example_id,
                task_id=task_id,
                candidate_indices=candidate_indices,
                action=action,
            )
            evaluation_handles[sample.split].write(
                json.dumps(evaluation_row, separators=(",", ":")) + "\n"
            )
            if processed == 1 or processed % 100 == 0 or processed == len(samples):
                _write_progress(
                    output_root / "build.progress.json",
                    {
                        "status": "running" if processed < len(samples) else "data_complete",
                        "states_processed": processed,
                        "states_total": len(samples),
                        "episodes_rendered": len(rendered),
                        "counts": dict(sorted(counts.items())),
                        "test_assets_read": False,
                    },
                )
    finally:
        for handle in [*handles.values(), *evaluation_handles.values()]:
            handle.close()

    summary = {
        "schema_version": "active-catalog-sequential-selector-sft-v2",
        "states": len(samples),
        "episodes": len({str(row.metadata["source_episode"]) for row in samples}),
        "policy_snapshot": policy_snapshot,
        "max_candidates": max_candidates,
        "counts": dict(sorted(counts.items())),
        "stats_only": stats_only,
        "prompt_target_metadata_exposed": False,
        "portable_relative_image_paths": True,
        "visual_encoding": "direct_crop_then_same_timestamp_wide_context_prior_outline",
        "candidate_fields": [
            "evidence_id",
            "timestamp",
            "scale",
            "clear_fraction",
            "temporal_offset_normalized",
            "cost",
        ],
        "test_assets_read": False,
        "retrieval": {
            "global_positive_states": global_positive_states,
            "global_positive_target_in_shortlist": global_positive_in_shortlist,
            "global_positive_target_recall": (
                global_positive_in_shortlist / global_positive_states
                if global_positive_states
                else 1.0
            ),
            "mean_positive_utility_regret": (
                float(np.mean(retrieval_regrets)) if retrieval_regrets else 0.0
            ),
            "by_split": {
                split: {
                    "global_positive_states": int(values["positive"]),
                    "global_positive_target_in_shortlist": int(values["included"]),
                    "global_positive_target_recall": (
                        float(values["included"]) / float(values["positive"])
                        if values["positive"]
                        else 1.0
                    ),
                    "mean_positive_utility_regret": (
                        float(np.mean(values["regrets"])) if values["regrets"] else 0.0
                    ),
                }
                for split, values in retrieval_by_split.items()
            },
        },
        "sources": {
            "states": {"path": str(states_path.resolve()), "sha256": _sha256(states_path)},
            "episodes": {
                "path": str(episodes_path.resolve()),
                "sha256": _sha256(episodes_path),
            },
        },
        "asset_root_maps": [
            {"source": str(source), "target": str(target)}
            for source, target in asset_root_maps
        ],
    }
    if not stats_only:
        summary["outputs"] = {
            split: {
                "path": str((output_root / f"{split}.jsonl").resolve()),
                "sha256": _sha256(output_root / f"{split}.jsonl"),
            }
            for split in ("train", "val")
        }
        summary["evaluation_indices"] = {
            split: {
                "path": str((output_root / f"{split}_evaluation_index.jsonl").resolve()),
                "sha256": _sha256(output_root / f"{split}_evaluation_index.jsonl"),
                "model_visible": False,
            }
            for split in ("train", "val")
        }
        (output_root / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        _write_progress(
            output_root / "build.progress.json",
            {
                "status": "complete",
                "states_processed": len(samples),
                "states_total": len(samples),
                "episodes_rendered": len(rendered),
                "summary_sha256": _sha256(output_root / "summary.json"),
                "test_assets_read": False,
            },
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("selector_states", type=Path)
    parser.add_argument("episodes_jsonl", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--policy-snapshot", required=True)
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument(
        "--asset-root-map",
        action="append",
        default=[],
        metavar="SOURCE=TARGET",
    )
    args = parser.parse_args()
    asset_root_maps = []
    for value in args.asset_root_map:
        if "=" not in value:
            parser.error("--asset-root-map must use SOURCE=TARGET")
        source, target = value.split("=", 1)
        asset_root_maps.append((Path(source), Path(target)))
    print(
        json.dumps(
            build_dataset(
                args.selector_states,
                args.episodes_jsonl,
                args.output_root,
                policy_snapshot=args.policy_snapshot,
                stats_only=args.stats_only,
                max_candidates=args.max_candidates,
                asset_root_maps=tuple(asset_root_maps),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
