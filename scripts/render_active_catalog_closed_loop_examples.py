#!/usr/bin/env python3
"""Render paper-ready recurrent acquisition and executable writeback examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from activemap.agent.identifiers import public_task_id, resolve_evidence_id
from activemap.models import EpisodeRecord

EDIT_COLORS = {
    "prior": np.asarray([255, 190, 0], dtype=np.float32),
    "committed": np.asarray([0, 205, 205], dtype=np.float32),
    "target": np.asarray([40, 200, 100], dtype=np.float32),
    "false_positive": np.asarray([245, 70, 70], dtype=np.float32),
    "false_negative": np.asarray([170, 80, 220], dtype=np.float32),
}
OPERATION_NAMES = ("KEEP", "ADD", "DELETE", "RESHAPE")


def event_action(event: dict[str, Any]) -> str:
    for key in ("executed_action", "action", "selected_action", "model_action"):
        value = event.get(key)
        if isinstance(value, dict):
            action = value.get("action") or value.get("type") or value.get("name")
            if action:
                return str(action)
        elif isinstance(value, str) and value:
            return value
    return str(event.get("event_type") or event.get("type") or "UNKNOWN")


def boundary(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    padded = np.pad(mask, 1, mode="edge")
    neighbors = [
        padded[1 + dy : 1 + dy + mask.shape[0], 1 + dx : 1 + dx + mask.shape[1]]
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
    ]
    eroded = np.logical_and.reduce(neighbors)
    return mask & ~eroded


def overlay_masks(
    image: np.ndarray,
    masks: list[tuple[np.ndarray, np.ndarray, float]],
) -> np.ndarray:
    result = np.asarray(image, dtype=np.float32).copy()
    for mask, color, alpha in masks:
        selected = np.asarray(mask, dtype=bool)
        result[selected] = (1.0 - alpha) * result[selected] + alpha * color
    return np.asarray(np.clip(result, 0, 255), dtype=np.uint8)


def error_overlay(
    image: np.ndarray, committed: np.ndarray, target: np.ndarray
) -> np.ndarray:
    committed = np.asarray(committed, dtype=bool)
    target = np.asarray(target, dtype=bool)
    true_positive = committed & target
    false_positive = committed & ~target
    false_negative = ~committed & target
    return overlay_masks(
        image,
        [
            (true_positive, EDIT_COLORS["target"], 0.38),
            (false_positive, EDIT_COLORS["false_positive"], 0.70),
            (false_negative, EDIT_COLORS["false_negative"], 0.70),
        ],
    )


def error_mask(committed: np.ndarray, target: np.ndarray) -> np.ndarray:
    committed = np.asarray(committed, dtype=bool)
    target = np.asarray(target, dtype=bool)
    result = np.zeros((*committed.shape, 3), dtype=np.uint8)
    result[committed & target] = EDIT_COLORS["target"].astype(np.uint8)
    result[committed & ~target] = EDIT_COLORS["false_positive"].astype(np.uint8)
    result[~committed & target] = EDIT_COLORS["false_negative"].astype(np.uint8)
    return result


def belief_history(trace: dict[str, Any]) -> np.ndarray:
    rows = []
    for event in trace.get("events", []):
        probabilities = event.get("observable_state", {}).get("belief", {}).get(
            "edit_probabilities"
        )
        if probabilities is not None:
            rows.append([float(value) for value in probabilities])
    if not rows:
        return np.empty((0, 4), dtype=np.float32)
    values = np.asarray(rows, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("belief history must contain four operation probabilities")
    return values


def select_examples(
    traces: list[dict[str, Any]],
    writebacks: list[dict[str, Any]],
    count: int,
) -> list[tuple[dict[str, Any], dict[str, Any], str]]:
    trace_by_key = {
        (public_task_id(str(row["source_episode"])), float(row["budget"])): row
        for row in traces
    }
    pairs = [
        (trace_by_key[(str(row["task_id"]), float(row["budget"]))], row)
        for row in writebacks
        if (str(row["task_id"]), float(row["budget"])) in trace_by_key
    ]
    if not pairs:
        raise ValueError("closed-loop traces and writebacks have no shared keys")
    ranked: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    best = max(pairs, key=lambda pair: float(pair[1]["raster_iou_gain"]))
    worst = min(pairs, key=lambda pair: float(pair[1]["raster_iou_gain"]))
    ranked.append((*best, "largest map-quality gain"))
    if worst != best:
        ranked.append((*worst, "failure case"))
    safe = [
        pair
        for pair in pairs
        if pair[0]["target_edit"] == "KEEP" and pair[0]["predicted_edit"] == "KEEP"
    ]
    if safe:
        ranked.append((*min(safe, key=lambda pair: float(pair[0]["spent_cost"])), "safe stop"))
    efficient = max(
        pairs,
        key=lambda pair: float(pair[1]["raster_iou_gain"])
        / max(float(pair[0]["spent_cost"]), 0.25),
    )
    ranked.append((*efficient, "cost-efficient update"))
    unique = []
    identities = set()
    for item in ranked:
        identity = (item[0]["source_episode"], float(item[0]["budget"]))
        if identity not in identities:
            identities.add(identity)
            unique.append(item)
        if len(unique) >= count:
            break
    return unique


def _rgb_for_evidence(episode: EpisodeRecord, evidence_id: str, image_size: int) -> np.ndarray:
    from activemap.oracle.updater_counterfactual import _geometry, _read_candidate

    evidence = next(item for item in episode.evidence_catalog if item.evidence_id == evidence_id)
    image, _, _, _, _ = _read_candidate(
        evidence,
        prior_geometry=_geometry(episode.prior_geometry),
        target_geometry=None,
        image_size=image_size,
        image_channels=3,
    )
    rgb = np.moveaxis(image[:3], 0, -1)
    return np.asarray(np.round(np.clip(rgb, 0.0, 1.0) * 255.0), dtype=np.uint8)


def render_panel_bundle(
    *,
    initial_rgb: np.ndarray,
    acquired_rgb: np.ndarray,
    prior_panel: np.ndarray,
    committed_panel: np.ndarray,
    target_panel: np.ndarray,
    errors: np.ndarray,
    beliefs: np.ndarray,
    output_dir: Path,
    header: str,
    acquired: bool,
    include_composite: bool = False,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    acquisition_title = "Acquired evidence" if acquired else "Stopped without acquisition"
    files = {
        "initial_evidence": "01_initial_evidence.png",
        "acquired_evidence": "02_acquired_evidence.png",
        "prior_mask": "04_prior_mask.png",
        "committed_mask": "05_committed_mask.png",
        "target_mask": "06_target_mask.png",
        "error_mask": "07_error_mask.png",
    }
    panel_payloads = (
        (files["initial_evidence"], initial_rgb),
        (files["acquired_evidence"], acquired_rgb),
        (files["prior_mask"], prior_panel),
        (files["committed_mask"], committed_panel),
        (files["target_mask"], target_panel),
        (files["error_mask"], errors),
    )
    for filename, panel in panel_payloads:
        Image.fromarray(np.asarray(panel, dtype=np.uint8)).save(output_dir / filename)

    if len(beliefs):
        files["belief_revision"] = "03_belief_revision.png"
        belief_figure, belief_axis = plt.subplots(figsize=(4.0, 4.0))
        for index, name in enumerate(OPERATION_NAMES):
            belief_axis.plot(
                range(len(beliefs)), beliefs[:, index], marker="o", linewidth=2
            )
        belief_axis.set_ylim(0.0, 1.0)
        belief_axis.set_xlim(0, max(len(beliefs) - 1, 1))
        belief_axis.axis("off")
        belief_figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
        belief_figure.savefig(
            output_dir / files["belief_revision"],
            dpi=220,
            bbox_inches="tight",
            pad_inches=0,
            transparent=True,
        )
        plt.close(belief_figure)

    if include_composite:
        files["composite"] = "00_composite.png"
        figure, axes = plt.subplots(2, 4, figsize=(15.5, 7.8), constrained_layout=True)
        for axis in axes.flat:
            axis.set_xticks([])
            axis.set_yticks([])
        axes[0, 0].imshow(initial_rgb)
        axes[0, 0].set_title("Initial evidence")
        axes[0, 1].imshow(acquired_rgb)
        axes[0, 1].set_title(acquisition_title)
        belief_axis = axes[0, 2]
        if len(beliefs):
            for index, name in enumerate(OPERATION_NAMES):
                belief_axis.plot(
                    range(len(beliefs)), beliefs[:, index], marker="o", label=name
                )
            belief_axis.set_ylim(0.0, 1.0)
            belief_axis.set_xticks(range(len(beliefs)))
            belief_axis.set_xlabel("Decision step")
            belief_axis.set_ylabel("Belief probability")
            belief_axis.set_title("Belief revision")
            belief_axis.legend(frameon=False, ncol=2, fontsize=8)
        else:
            belief_axis.axis("off")
        axes[0, 3].axis("off")
        axes[1, 0].imshow(prior_panel, cmap="gray", vmin=0, vmax=255)
        axes[1, 0].set_title("Prior mask")
        axes[1, 1].imshow(committed_panel, cmap="gray", vmin=0, vmax=255)
        axes[1, 1].set_title("Committed mask")
        axes[1, 2].imshow(target_panel, cmap="gray", vmin=0, vmax=255)
        axes[1, 2].set_title("Target mask")
        axes[1, 3].imshow(errors)
        axes[1, 3].set_title("Error: FP red / FN purple / TP green")
        figure.suptitle(header, fontsize=12)
        figure.savefig(
            output_dir / files["composite"], dpi=220, bbox_inches="tight"
        )
        plt.close(figure)
    return files


def render_example(
    episode: EpisodeRecord,
    trace: dict[str, Any],
    writeback: dict[str, Any],
    output_dir: Path,
    label: str,
    *,
    include_composite: bool = False,
) -> dict[str, Any]:
    artifact = np.load(str(writeback["mask_artifact"]))
    prior = np.asarray(artifact["prior_mask"], dtype=bool)
    committed = np.asarray(artifact["committed_mask"], dtype=bool)
    target = np.asarray(artifact["target_mask"], dtype=bool)
    image_size = int(prior.shape[0])
    selected = [
        resolve_evidence_id(str(item), [row.evidence_id for row in episode.evidence_catalog])
        for item in trace["selected_evidence_ids"]
    ]
    initial_rgb = _rgb_for_evidence(episode, selected[0], image_size)
    acquired_rgb = _rgb_for_evidence(episode, selected[-1], image_size)
    prior_panel = np.asarray(prior, dtype=np.uint8) * 255
    committed_panel = np.asarray(committed, dtype=np.uint8) * 255
    target_panel = np.asarray(target, dtype=np.uint8) * 255
    errors = error_mask(committed, target)
    beliefs = belief_history(trace)

    actions = " -> ".join(event_action(event) for event in trace.get("events", []))
    header = (
        f"{label} | {episode.aoi_id} | target={trace['target_edit']} "
        f"pred={trace['predicted_edit']} | cost={trace['spent_cost']:.2f} | "
        f"raster IoU gain={writeback['raster_iou_gain']:.3f}\n{actions}"
    )
    files = render_panel_bundle(
        initial_rgb=initial_rgb,
        acquired_rgb=acquired_rgb,
        prior_panel=prior_panel,
        committed_panel=committed_panel,
        target_panel=target_panel,
        errors=errors,
        beliefs=beliefs,
        output_dir=output_dir,
        header=header,
        acquired=len(selected) > 1,
        include_composite=include_composite,
    )
    metadata = {
        "schema_version": "active-catalog-closed-loop-example-v1",
        "selection_label": label,
        "source_episode": trace["source_episode"],
        "aoi_id": str(episode.aoi_id),
        "budget": float(trace["budget"]),
        "target_edit": trace["target_edit"],
        "predicted_edit": trace["predicted_edit"],
        "spent_cost": float(trace["spent_cost"]),
        "selected_evidence_ids": trace["selected_evidence_ids"],
        "action_sequence": actions,
        "raster_iou": float(writeback["raster_iou"]),
        "prior_raster_iou": float(writeback["prior_raster_iou"]),
        "raster_iou_gain": float(writeback["raster_iou_gain"]),
        "vector_replay_iou": float(writeback["vector_replay_iou"]),
        "vector_delta_topology_valid": bool(
            writeback["vector_delta_topology_valid"]
        ),
        "colors": {
            "prior": "yellow",
            "committed": "cyan",
            "target_or_true_positive": "green",
            "false_positive": "red",
            "false_negative": "purple",
        },
        "files": files,
        "belief_history_available": bool(len(beliefs)),
        "mask_encoding": {
            "prior_committed_target": "uint8 binary; background=0 foreground=255",
            "error_background": "black",
        },
        "test_assets_read": False,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def _parse_asset_root_maps(values: list[str]) -> tuple[tuple[Path, Path], ...]:
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


def _remap_path(value: str | None, mappings: tuple[tuple[Path, Path], ...]) -> str | None:
    if value is None:
        return None
    original = Path(value)
    for source, target in mappings:
        try:
            return str(target / original.relative_to(source))
        except ValueError:
            continue
    return value


def _remap_episode_assets(episode, mappings):
    if not mappings:
        return episode
    catalog = [
        item.model_copy(
            update={
                "image_path": _remap_path(item.image_path, mappings),
                "udm_path": _remap_path(item.udm_path, mappings),
            }
        )
        for item in episode.evidence_catalog
    ]
    return episode.model_copy(update={"evidence_catalog": catalog})


def main() -> None:
    from activemap.oracle.updater_counterfactual import load_episodes

    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=Path)
    parser.add_argument("traces", type=Path)
    parser.add_argument("writebacks", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--include-composite", action="store_true")
    parser.add_argument("--asset-root-map", action="append", default=[])
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.count <= 0:
        raise ValueError("count must be positive")
    asset_root_maps = _parse_asset_root_maps(args.asset_root_map)
    episodes = {
        episode.episode_id: _remap_episode_assets(episode, asset_root_maps)
        for episode in load_episodes(args.episodes, split="val")
    }
    traces = [
        json.loads(line) for line in args.traces.read_text(encoding="utf-8").splitlines() if line
    ]
    writebacks = [
        json.loads(line)
        for line in args.writebacks.read_text(encoding="utf-8").splitlines()
        if line
    ]
    selected = select_examples(traces, writebacks, args.count)
    args.output_dir.mkdir(parents=True)
    index = []
    for position, (trace, writeback, label) in enumerate(selected, 1):
        episode = episodes[str(trace["source_episode"])]
        example_dir = args.output_dir / f"{position:02d}_{label.replace(' ', '_')}"
        metadata = render_example(
            episode,
            trace,
            writeback,
            example_dir,
            label,
            include_composite=args.include_composite,
        )
        index.append(
            {
                "rank": position,
                "label": label,
                "source_episode": trace["source_episode"],
                "budget": trace["budget"],
                "directory": str(example_dir.resolve()),
                "panels": metadata["files"],
            }
        )
    (args.output_dir / "index.json").write_text(
        json.dumps(
            {
                "schema_version": "active-catalog-closed-loop-figure-v1",
                "examples": index,
                "selection_uses_validation_only": True,
                "test_assets_read": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
