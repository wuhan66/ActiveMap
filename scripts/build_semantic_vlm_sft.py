#!/usr/bin/env python3
"""Build consensus-grounded multimodal SFT for semantic tool use."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from activemap.agent.identifiers import public_task_id
from activemap.agent.tool_belief_data import PostAcquisitionToolPairExample
from activemap.agent.tool_sft import terminal_action, terminal_reward
from activemap.geo_tools.records import GeoToolCall, GeoToolName
from activemap.models import EditOperation
from activemap.updater_records import UpdaterSample, load_updater_samples

SYSTEM_PROMPT = (
    "You are the ActiveMap visual tool controller. Inspect the registered current remote-sensing "
    "image and editable map prior. Return exactly one executable JSON action. Use RASTER_SEGMENT "
    "only when its expected map-quality or safety gain exceeds cost. Otherwise COMMIT ADD, "
    "DELETE, or RESHAPE, or REJECT to preserve the map. Return JSON only."
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> list[PostAcquisitionToolPairExample]:
    rows = [
        PostAcquisitionToolPairExample.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty semantic dataset: {path}")
    return rows


def _index_rows(
    paths: list[Path],
) -> tuple[list[str], list[dict[str, PostAcquisitionToolPairExample]]]:
    indices = []
    order = []
    for position, path in enumerate(paths):
        rows = _read(path)
        index = {row.example_id: row for row in rows}
        if len(index) != len(rows):
            raise ValueError(f"duplicate example_id in {path}")
        if position == 0:
            order = [row.example_id for row in rows]
        elif set(index) != set(indices[0]):
            raise ValueError("semantic seed datasets do not share example IDs")
        indices.append(index)
    return order, indices


def consensus_tool_opportunity(
    rows: list[PostAcquisitionToolPairExample], *, minimum_votes: int
) -> dict[str, Any]:
    if not rows or not 1 <= minimum_votes <= len(rows):
        raise ValueError("minimum_votes must be within the semantic seed count")
    reference = rows[0]
    if any(
        row.task_id != reference.task_id
        or row.evidence_id != reference.evidence_id
        or row.gt_edit != reference.gt_edit
        or row.split != reference.split
        for row in rows[1:]
    ):
        raise ValueError("semantic seed rows disagree on state identity or target")
    baseline = list(EditOperation)[
        int(np.argmax(reference.post_acquisition_belief.edit_probabilities))
    ]
    baseline_reward = terminal_reward(reference.gt_edit, baseline)
    gains = []
    operations = []
    for row in rows:
        if row.semantic_result is None:
            raise ValueError(f"missing semantic result: {row.example_id}")
        operation = EditOperation(str(row.semantic_result.outputs.get("gated_edit")))
        operations.append(operation)
        gains.append(
            terminal_reward(row.gt_edit, operation) - row.semantic_result.cost - baseline_reward
        )
    votes = sum(gain > 0.0 for gain in gains)
    mean_gain = float(np.mean(gains))
    return {
        "use_tool": votes >= minimum_votes and mean_gain > 0.0,
        "beneficial_votes": votes,
        "mean_utility_gain": mean_gain,
        "per_seed_utility_gain": gains,
        "baseline_operation": baseline,
        "semantic_operations": operations,
    }


def _remap(path: str | Path, mappings: tuple[tuple[Path, Path], ...]) -> Path:
    original = Path(path)
    for source, target in mappings:
        try:
            return target / original.relative_to(source)
        except ValueError:
            continue
    return original


def _parse_mappings(values: list[str]) -> tuple[tuple[Path, Path], ...]:
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


def _samples(path: Path) -> dict[str, UpdaterSample]:
    rows = load_updater_samples(path)
    result = {public_task_id(f"{row.sample_id}__temporal"): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("public updater sample collision")
    return result


def _rgb(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim != 3:
        raise ValueError("image NPY must have three dimensions")
    if values.shape[0] in {1, 3, 4}:
        values = np.moveaxis(values[:3], 0, -1)
    elif values.shape[-1] in {1, 3, 4}:
        values = values[..., :3]
    else:
        raise ValueError(f"cannot infer image channel axis: {values.shape}")
    values = values.astype(np.float32)
    output = np.empty_like(values)
    for channel in range(values.shape[-1]):
        plane = values[..., channel]
        finite = plane[np.isfinite(plane)]
        if finite.size == 0:
            raise ValueError("image channel has no finite pixels")
        low, high = np.percentile(finite, [2.0, 98.0])
        output[..., channel] = np.clip((plane - low) / max(high - low, 1e-6), 0.0, 1.0)
    if output.shape[-1] == 1:
        output = np.repeat(output, 3, axis=-1)
    return np.asarray(np.round(output * 255.0), dtype=np.uint8)


def render_registered_state(
    image_path: Path, prior_path: Path, output_path: Path, *, panel_size: int = 384
) -> None:
    rgb = _rgb(np.load(image_path))
    prior = np.asarray(np.load(prior_path)).squeeze() > 0
    if prior.shape != rgb.shape[:2]:
        raise ValueError("image and prior are not registered")
    overlay = rgb.astype(np.float32)
    cyan = np.asarray([0.0, 220.0, 220.0], dtype=np.float32)
    overlay[prior] = 0.45 * overlay[prior] + 0.55 * cyan
    left = Image.fromarray(rgb).resize((panel_size, panel_size), Image.Resampling.BILINEAR)
    right = Image.fromarray(np.asarray(overlay, dtype=np.uint8)).resize(
        (panel_size, panel_size), Image.Resampling.BILINEAR
    )
    canvas = Image.new("RGB", (panel_size * 2, panel_size))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (panel_size, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92, subsampling=0)


def _observation(row: PostAcquisitionToolPairExample, *, post_tool: bool) -> dict[str, Any]:
    observation = {
        "task_id": row.task_id,
        "evidence_id": row.evidence_id,
        "belief": row.post_acquisition_belief.model_dump(mode="json"),
        "available_tools": [GeoToolName.RASTER_SEGMENT.value],
        "semantic_tool_cost": float(row.semantic_result.cost) if row.semantic_result else None,
        "stage": "POST_TOOL" if post_tool else "PRE_TOOL",
    }
    if post_tool:
        assert row.semantic_result is not None
        observation["tool_result"] = {
            key: row.semantic_result.outputs.get(key)
            for key in (
                "edit_probabilities",
                "gated_edit",
                "confidence",
                "uncertainty",
                "changed_fraction",
                "add_fraction",
                "remove_fraction",
                "prior_iou",
            )
        }
    return observation


def _messages(image_path: Path, observation: dict[str, Any], action: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": json.dumps(observation, separators=(",", ":"))},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": action}]},
    ]


def build_dataset(
    semantic_paths: list[Path],
    updater_manifest: Path,
    output_dir: Path,
    *,
    minimum_votes: int,
    asset_root_maps: tuple[tuple[Path, Path], ...],
    stats_only: bool,
) -> dict[str, Any]:
    order, indices = _index_rows(semantic_paths)
    split = indices[0][order[0]].split
    if split not in {"train", "val"} or any(
        row.split != split for index in indices for row in index.values()
    ):
        raise ValueError("SFT input must contain one train or validation split")
    samples = {} if stats_only else _samples(updater_manifest)
    action_counts: Counter[str] = Counter()
    vote_counts: Counter[int] = Counter()
    positive_tasks = set()
    rendered = set()
    records = []
    for example_id in order:
        rows = [index[example_id] for index in indices]
        decision = consensus_tool_opportunity(rows, minimum_votes=minimum_votes)
        row = rows[0]
        vote_counts[int(decision["beneficial_votes"])] += 1
        use_tool = bool(decision["use_tool"])
        action_counts["USE_TOOL" if use_tool else "TERMINAL"] += 1
        if use_tool:
            positive_tasks.add(row.task_id)
        if stats_only:
            continue
        sample = samples.get(row.task_id)
        if sample is None:
            raise ValueError(f"missing updater sample for {row.task_id}")
        image_path = _remap(sample.image_path, asset_root_maps)
        prior_path = _remap(sample.prior_mask_path, asset_root_maps)
        visual_path = output_dir / "images" / f"{row.task_id}.jpg"
        if row.task_id not in rendered:
            render_registered_state(image_path, prior_path, visual_path)
            rendered.add(row.task_id)
        if use_tool:
            tool_action = {
                "action": "USE_TOOL",
                "tool_call": GeoToolCall(
                    call_id=f"vlm-{example_id[:12]}",
                    tool=GeoToolName.RASTER_SEGMENT,
                    inputs={"evidence_id": row.evidence_id},
                ).model_dump(mode="json", exclude_none=True),
            }
            records.append(
                {
                    "messages": _messages(
                        visual_path,
                        _observation(row, post_tool=False),
                        json.dumps(tool_action, separators=(",", ":")),
                    ),
                    "example_id": example_id,
                    "task_id": row.task_id,
                    "stage": "PRE_TOOL",
                    "oracle_use_tool": True,
                    "consensus": decision,
                    "split": split,
                }
            )
            terminal = terminal_action(row.gt_edit).model_dump_json(exclude_none=True)
            records.append(
                {
                    "messages": _messages(visual_path, _observation(row, post_tool=True), terminal),
                    "example_id": example_id,
                    "task_id": row.task_id,
                    "stage": "POST_TOOL",
                    "oracle_use_tool": True,
                    "consensus": decision,
                    "split": split,
                }
            )
        else:
            terminal = terminal_action(row.gt_edit).model_dump_json(exclude_none=True)
            records.append(
                {
                    "messages": _messages(
                        visual_path, _observation(row, post_tool=False), terminal
                    ),
                    "example_id": example_id,
                    "task_id": row.task_id,
                    "stage": "PRE_TOOL",
                    "oracle_use_tool": False,
                    "consensus": decision,
                    "split": split,
                }
            )
    summary = {
        "schema_version": "semantic-vlm-sft-consensus-v1",
        "split": split,
        "semantic_seed_count": len(indices),
        "minimum_beneficial_votes": minimum_votes,
        "source_example_count": len(order),
        "sft_record_count": len(records) if not stats_only else None,
        "task_count": len({indices[0][example_id].task_id for example_id in order}),
        "tool_positive_example_count": action_counts["USE_TOOL"],
        "tool_positive_task_count": len(positive_tasks),
        "tool_positive_rate": action_counts["USE_TOOL"] / len(order),
        "action_counts": dict(action_counts),
        "beneficial_vote_counts": {str(key): value for key, value in sorted(vote_counts.items())},
        "visual_protocol": "registered-new-image-plus-editable-prior-overlay",
        "test_assets_read": False,
        "sources": [
            {"path": str(path.resolve()), "sha256": _sha256(path)} for path in semantic_paths
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    if not stats_only:
        output_path = output_dir / "sft.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        summary["sft_sha256"] = _sha256(output_path)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("updater_manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--semantic", type=Path, action="append", required=True)
    parser.add_argument("--minimum-votes", type=int, default=2)
    parser.add_argument("--asset-root-map", action="append", default=[])
    parser.add_argument("--stats-only", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    summary = build_dataset(
        args.semantic,
        args.updater_manifest,
        args.output_dir,
        minimum_votes=args.minimum_votes,
        asset_root_maps=_parse_mappings(args.asset_root_map),
        stats_only=args.stats_only,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
