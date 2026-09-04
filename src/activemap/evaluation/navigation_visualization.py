"""Map-native rendering for auditable indoor-navigation rollouts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from activemap.data.navigation_map import NavigationMapEpisode
from activemap.evaluation.navigation_rollout import (
    NavigationRolloutResult,
    NavigationRolloutStep,
    _load_occupancy,
    _merge_local_observation,
)

UNKNOWN = np.float32(0.5)
LEGEND_ITEMS = (
    ("Unknown", (166, 174, 184)),
    ("Correct free", (255, 255, 255)),
    ("Correct obstacle", (32, 38, 46)),
    ("False free", (220, 75, 64)),
    ("False obstacle", (232, 145, 50)),
)


def load_navigation_rollout_results(path: Path) -> list[NavigationRolloutResult]:
    """Load trace records written by the navigation evaluator."""
    records: list[NavigationRolloutResult] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            payload["steps"] = tuple(NavigationRolloutStep(**step) for step in payload["steps"])
            records.append(NavigationRolloutResult(**payload))
    return records


def reconstruct_committed_map(
    episode: NavigationMapEpisode,
    result: NavigationRolloutResult,
) -> np.ndarray:
    """Replay only trace-approved commits into the final occupancy map."""
    target = _load_occupancy(episode.target_map_path)
    committed = _load_occupancy(episode.initial_map_path, expected_shape=target.shape)
    evidence = {row.evidence_id: row for row in episode.evidence}
    for step in result.steps:
        if not step.committed:
            continue
        observation = _load_occupancy(evidence[step.evidence_id].path, expected_shape=target.shape)
        committed, _ = _merge_local_observation(committed, observation)
    return committed


def _occupancy_rgb(
    committed: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Render correct map content, unknowns, and directional occupancy errors."""
    image = np.full((*target.shape, 3), (246, 246, 246), dtype=np.uint8)
    image[valid & (committed == UNKNOWN)] = (166, 174, 184)
    image[valid & (committed == 0.0) & (target == 0.0)] = (255, 255, 255)
    image[valid & (committed == 1.0) & (target == 1.0)] = (32, 38, 46)
    image[valid & (committed == 0.0) & (target == 1.0)] = (220, 75, 64)
    image[valid & (committed == 1.0) & (target == 0.0)] = (232, 145, 50)
    return image


def _active_crop_slices(
    target: np.ndarray,
    valid: np.ndarray,
    *,
    padding: int = 8,
) -> tuple[slice, slice]:
    """Crop all comparison panels to the reference-supported map extent."""
    if padding < 0:
        raise ValueError("padding must be non-negative")
    active_rows, active_cols = np.nonzero(valid | (target != UNKNOWN))
    if not len(active_rows):
        return slice(0, target.shape[0]), slice(0, target.shape[1])
    row_start = max(0, int(active_rows.min()) - padding)
    row_stop = min(target.shape[0], int(active_rows.max()) + padding + 1)
    col_start = max(0, int(active_cols.min()) - padding)
    col_stop = min(target.shape[1], int(active_cols.max()) + padding + 1)
    return slice(row_start, row_stop), slice(col_start, col_stop)


def render_navigation_comparison(
    episode: NavigationMapEpisode,
    *,
    labeled_results: list[tuple[str, NavigationRolloutResult]],
    output_path: Path,
    panel_width: int = 280,
) -> None:
    """Render one target, one prior, and trace-replayed policy maps side by side."""
    if not labeled_results:
        raise ValueError("at least one rollout result is required")
    target = _load_occupancy(episode.target_map_path)
    initial = _load_occupancy(episode.initial_map_path, expected_shape=target.shape)
    valid_path = episode.metadata.get("valid_space_path")
    valid = (
        np.load(valid_path) > 0
        if isinstance(valid_path, str)
        else np.ones(target.shape, dtype=bool)
    )
    panels = [("Target occupancy", target), ("Initial committed map", initial)]
    panels.extend(
        (label, reconstruct_committed_map(episode, result)) for label, result in labeled_results
    )

    row_slice, col_slice = _active_crop_slices(target, valid)
    target = target[row_slice, col_slice]
    valid = valid[row_slice, col_slice]
    panels = [(label, committed[row_slice, col_slice]) for label, committed in panels]
    scale = panel_width / max(target.shape)
    panel_height = max(1, round(target.shape[0] * scale))
    margin, label_height, legend_height = 12, 38, 28
    canvas = Image.new(
        "RGB",
        (
            len(panels) * (panel_width + margin) + margin,
            panel_height + label_height + legend_height + 2 * margin,
        ),
        (250, 250, 250),
    )
    draw = ImageDraw.Draw(canvas)
    for index, (label, committed) in enumerate(panels):
        panel = Image.fromarray(_occupancy_rgb(committed, target, valid), mode="RGB")
        panel = panel.resize((panel_width, panel_height), Image.Resampling.NEAREST)
        origin = (margin + index * (panel_width + margin), label_height + margin)
        canvas.paste(panel, origin)
        draw.rectangle(
            (origin[0] - 1, origin[1] - 1, origin[0] + panel_width, origin[1] + panel_height),
            outline=(75, 82, 90),
            width=1,
        )
        draw.text((origin[0], margin), label, fill=(25, 31, 38))
    legend_x = margin
    legend_y = label_height + panel_height + margin + 6
    for label, color in LEGEND_ITEMS:
        draw.rectangle(
            (legend_x, legend_y, legend_x + 10, legend_y + 10), fill=color, outline=(75, 82, 90)
        )
        draw.text((legend_x + 15, legend_y - 2), label, fill=(25, 31, 38))
        legend_x += 15 + draw.textlength(label) + 18
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
