#!/usr/bin/env python3
"""Render identical validation examples for multiple closed-loop controllers."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from activemap.agent.identifiers import public_task_id
from scripts.render_active_catalog_closed_loop_examples import (
    _parse_asset_root_maps,
    _remap_episode_assets,
    render_example,
)


Key = tuple[str, float]


def _parse_method(value: str) -> tuple[str, Path, Path]:
    name, separator, paths = value.partition("=")
    trace, second, writeback = paths.partition(",")
    if (
        not separator
        or not second
        or not re.fullmatch(r"[A-Za-z0-9_-]+", name)
        or not trace
        or not writeback
    ):
        raise argparse.ArgumentTypeError(
            "method must be NAME=/path/to/traces.jsonl,/path/to/writeback.jsonl"
        )
    return name, Path(trace), Path(writeback)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty visual input: {path}")
    return rows


def _trace_index(path: Path) -> dict[Key, dict[str, Any]]:
    result = {}
    for row in _read_jsonl(path):
        if row.get("split") != "val" or row.get("test_assets_read") is not False:
            raise ValueError(f"visual trace is not validation-only: {path}")
        key = (public_task_id(str(row["source_episode"])), float(row["budget"]))
        if key in result:
            raise ValueError(f"duplicate visual trace key: {key}")
        result[key] = row
    return result


def _writeback_index(path: Path) -> dict[Key, dict[str, Any]]:
    result = {}
    for row in _read_jsonl(path):
        if row.get("test_assets_read") is not False:
            raise ValueError(f"visual writeback is not validation-only: {path}")
        key = (str(row["task_id"]), float(row["budget"]))
        if key in result:
            raise ValueError(f"duplicate visual writeback key: {key}")
        result[key] = row
    return result


def _visible_fraction(writeback: dict[str, Any]) -> float:
    artifact = np.load(str(writeback["mask_artifact"]))
    prior = np.asarray(artifact["prior_mask"], dtype=bool)
    target = np.asarray(artifact["target_mask"], dtype=bool)
    return max(float(prior.mean()), float(np.logical_xor(prior, target).mean()))


def select_paired_examples(
    traces: dict[str, dict[Key, dict[str, Any]]],
    writebacks: dict[str, dict[Key, dict[str, Any]]],
    *,
    candidate: str,
    per_edit_success: int,
    per_edit_failure: int,
    per_edit_disagreement: int,
    visible_fraction: dict[Key, float] | None = None,
    min_visual_fraction: float = 0.01,
) -> list[tuple[Key, str]]:
    if candidate not in traces or candidate not in writebacks:
        raise ValueError(f"missing candidate method: {candidate}")
    support = set(traces[candidate])
    collections = [*traces.values(), *writebacks.values()]
    if any(set(values) != support for values in collections):
        raise ValueError("paired visual methods require identical task-budget support")
    selected: list[tuple[Key, str]] = []
    used: set[Key] = set()
    visible_fraction = visible_fraction or {key: 1.0 for key in support}
    for edit in ("KEEP", "ADD", "DELETE", "RESHAPE"):
        keys = [
            key
            for key in support
            if traces[candidate][key]["target_edit"] == edit
            and visible_fraction.get(key, 0.0) >= min_visual_fraction
        ]
        successes = sorted(
            (
                key
                for key in keys
                if traces[candidate][key]["predicted_edit"] == edit
            ),
            key=lambda key: float(
                writebacks[candidate][key]["raster_iou_gain"]
            ),
            reverse=True,
        )
        failures = sorted(
            keys,
            key=lambda key: (
                traces[candidate][key]["predicted_edit"] == edit,
                float(writebacks[candidate][key]["raster_iou_gain"]),
            ),
        )
        disagreements = sorted(
            keys,
            key=lambda key: (
                max(
                    float(values[key]["raster_iou_gain"])
                    for values in writebacks.values()
                )
                - min(
                    float(values[key]["raster_iou_gain"])
                    for values in writebacks.values()
                )
            ),
            reverse=True,
        )
        for category, ranked, count in (
            ("success", successes, per_edit_success),
            ("failure", failures, per_edit_failure),
            ("disagreement", disagreements, per_edit_disagreement),
        ):
            added = 0
            for key in ranked:
                if key in used:
                    continue
                selected.append((key, category))
                used.add(key)
                added += 1
                if added >= count:
                    break
    return selected


def render(
    episodes_path: Path,
    methods: list[tuple[str, Path, Path]],
    output_dir: Path,
    *,
    candidate: str,
    per_edit_success: int,
    per_edit_failure: int,
    per_edit_disagreement: int,
    min_visual_fraction: float,
    asset_root_maps: tuple[tuple[Path, Path], ...] = (),
) -> dict[str, Any]:
    from activemap.oracle.updater_counterfactual import load_episodes

    if output_dir.exists():
        raise FileExistsError(output_dir)
    traces = {name: _trace_index(trace) for name, trace, _ in methods}
    writebacks = {
        name: _writeback_index(writeback) for name, _, writeback in methods
    }
    support = set(traces[candidate])
    fractions = {
        key: _visible_fraction(writebacks[candidate][key]) for key in support
    }
    selected = select_paired_examples(
        traces,
        writebacks,
        candidate=candidate,
        per_edit_success=per_edit_success,
        per_edit_failure=per_edit_failure,
        per_edit_disagreement=per_edit_disagreement,
        visible_fraction=fractions,
        min_visual_fraction=min_visual_fraction,
    )
    episodes = {
        episode.episode_id: _remap_episode_assets(episode, asset_root_maps)
        for episode in load_episodes(episodes_path, split="val")
    }
    output_dir.mkdir(parents=True)
    index = []
    for rank, (key, category) in enumerate(selected, 1):
        task_id, budget = key
        candidate_trace = traces[candidate][key]
        source_episode = str(candidate_trace["source_episode"])
        folder = output_dir / (
            f"{rank:03d}_{candidate_trace['target_edit'].lower()}_{category}_{task_id[:8]}"
        )
        folder.mkdir()
        method_metadata = {}
        for name, _, _ in methods:
            method_metadata[name] = render_example(
                episodes[source_episode],
                traces[name][key],
                writebacks[name][key],
                folder / name,
                category,
                include_composite=False,
            )
        index.append(
            {
                "rank": rank,
                "folder": folder.name,
                "task_id": task_id,
                "source_episode": source_episode,
                "budget": budget,
                "target_edit": candidate_trace["target_edit"],
                "category": category,
                "methods": {
                    name: {
                        "predicted_edit": traces[name][key]["predicted_edit"],
                        "raster_iou_gain": writebacks[name][key]["raster_iou_gain"],
                    }
                    for name, _, _ in methods
                },
            }
        )
    with (output_dir / "selection.jsonl").open("x", encoding="utf-8") as handle:
        for row in index:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {
        "schema_version": "active-catalog-paired-qualitative-v1",
        "candidate": candidate,
        "methods": [name for name, _, _ in methods],
        "support_count": len(support),
        "selected_count": len(index),
        "minimum_visual_fraction": min_visual_fraction,
        "identical_sample_ids_across_methods": True,
        "individual_unlabeled_panels": True,
        "composites_rendered": False,
        "selection_uses_validation_only": True,
        "test_assets_read": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--method", action="append", type=_parse_method, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--per-edit-success", type=int, default=2)
    parser.add_argument("--per-edit-failure", type=int, default=1)
    parser.add_argument("--per-edit-disagreement", type=int, default=1)
    parser.add_argument("--min-visual-fraction", type=float, default=0.01)
    parser.add_argument("--asset-root-map", action="append", default=[])
    args = parser.parse_args()
    names = [name for name, _, _ in args.method]
    if len(names) != len(set(names)):
        raise ValueError("duplicate visual method names")
    if min(
        args.per_edit_success,
        args.per_edit_failure,
        args.per_edit_disagreement,
    ) < 0:
        raise ValueError("visual selection counts must be nonnegative")
    if not 0.0 <= args.min_visual_fraction < 1.0:
        raise ValueError("minimum visual fraction must be in [0, 1)")
    print(
        json.dumps(
            render(
                args.episodes,
                args.method,
                args.output_dir,
                candidate=args.candidate,
                per_edit_success=args.per_edit_success,
                per_edit_failure=args.per_edit_failure,
                per_edit_disagreement=args.per_edit_disagreement,
                min_visual_fraction=args.min_visual_fraction,
                asset_root_maps=_parse_asset_root_maps(args.asset_root_map),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
