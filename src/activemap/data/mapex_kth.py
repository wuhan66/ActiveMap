"""MapEx KTH adapter for non-test indoor active-mapping development episodes.

The public KTH assets provide complete occupancy maps, not recorded robot
rollouts.  This adapter therefore constructs deterministic *local-observation*
episodes: a policy sees only a small initial observation and must pay to reveal
one of several candidate local occupancy observations.  It is deliberately a
development pilot, not evidence of dynamic or persistent map updating.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np

from activemap.data.navigation_map import (
    NavigationEvidence,
    NavigationMapEpisode,
    NavigationPose,
    write_navigation_map_jsonl,
)

MAPEX_KTH_SOURCE_COMMIT = "53636bd1c79153acc3c74a532837d78c926bae5e"
OCCUPANCY_UNKNOWN = np.float32(0.5)


def deterministic_mapex_kth_split(
    map_id: str, val_fraction: float = 0.2
) -> Literal["train", "val"]:
    """Assign whole KTH maps to a deterministic development split."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between zero and one")
    bucket = int(hashlib.sha256(map_id.encode("ascii")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < val_fraction else "train"


def list_mapex_kth_maps(source_root: Path) -> list[Path]:
    """Return KTH map directories with the paired source assets required here."""
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    maps = [
        path
        for path in sorted(source_root.iterdir())
        if path.is_dir()
        and (path / "occ_map.npy").is_file()
        and (path / "valid_space.npy").is_file()
    ]
    if not maps:
        raise ValueError(f"no KTH map directories found under {source_root}")
    return maps


def _stable_seed(map_id: str, seed: int) -> int:
    payload = f"{seed}:{map_id}".encode()
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def _normalized_occupancy(raw: np.ndarray) -> np.ndarray:
    values = set(np.unique(raw).tolist())
    if values != {0, 254}:
        raise ValueError(f"KTH occupancy must contain exactly {{0, 254}}, got {sorted(values)}")
    return np.where(raw == 0, 1.0, 0.0).astype(np.float32)


def _local_reveal(
    target: np.ndarray,
    valid_space: np.ndarray,
    center: tuple[int, int],
    radius: int,
    *,
    sensor_error_rate: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if not 0.0 <= sensor_error_rate < 1.0:
        raise ValueError("sensor_error_rate must be in [0, 1)")
    if sensor_error_rate > 0.0 and rng is None:
        raise ValueError("a random generator is required when adding sensor error")
    observation = np.full(target.shape, OCCUPANCY_UNKNOWN, dtype=np.float32)
    row, col = center
    row_start, row_stop = max(0, row - radius), min(target.shape[0], row + radius + 1)
    col_start, col_stop = max(0, col - radius), min(target.shape[1], col + radius + 1)
    observation[row_start:row_stop, col_start:col_stop] = target[
        row_start:row_stop, col_start:col_stop
    ]
    observation[valid_space <= 0] = OCCUPANCY_UNKNOWN
    if sensor_error_rate > 0.0:
        observed = observation != OCCUPANCY_UNKNOWN
        errors = observed & (rng.random(target.shape) < sensor_error_rate)
        observation[errors] = 1.0 - observation[errors]
    return observation


def _sample_free_positions(
    target: np.ndarray,
    valid_space: np.ndarray,
    count: int,
    seed: int,
) -> list[tuple[int, int]]:
    available = np.argwhere((target == 0.0) & (valid_space > 0))
    if len(available) < count:
        raise ValueError(f"only {len(available)} valid free cells available for {count} poses")

    rng = np.random.default_rng(seed)
    pool_size = min(len(available), max(count * 64, 1024))
    pool = available[rng.choice(len(available), size=pool_size, replace=False)]
    centroid = np.mean(available, axis=0)
    chosen = [pool[int(np.argmin(np.sum((pool - centroid) ** 2, axis=1)))]]
    while len(chosen) < count:
        chosen_array = np.asarray(chosen)
        nearest_sq_distance = np.min(
            np.sum((pool[:, None, :] - chosen_array[None, :, :]) ** 2, axis=2), axis=1
        )
        candidate = pool[int(np.argmax(nearest_sq_distance))]
        chosen.append(candidate)
    return [(int(row), int(col)) for row, col in chosen]


def _pose(row: int, col: int, pixels_per_meter: float) -> NavigationPose:
    return NavigationPose(x=col / pixels_per_meter, y=row / pixels_per_meter, yaw=0.0)


def build_mapex_kth_navigation_episodes(
    source_root: Path,
    output_root: Path,
    *,
    candidate_count: int = 8,
    observation_radius: int = 96,
    pixels_per_meter: float = 10.0,
    budget: float = 250.0,
    sensor_error_rate: float = 0.0,
    val_fraction: float = 0.2,
    seed: int = 20260829,
    max_maps: int | None = None,
) -> dict[str, object]:
    """Materialize deterministic train/validation MapEx KTH navigation episodes.

    ``source_root`` is the immutable ``kth_test_maps`` directory.  The output
    consists exclusively of normalized derived maps and manifests.  No split
    named ``test`` is generated, and source targets never appear in initial-map
    artifacts or evidence metadata.
    """
    if candidate_count < 1:
        raise ValueError("candidate_count must be at least one")
    if observation_radius < 1:
        raise ValueError("observation_radius must be positive")
    if pixels_per_meter <= 0.0 or budget <= 0.0:
        raise ValueError("pixels_per_meter and budget must be positive")
    if not 0.0 <= sensor_error_rate < 1.0:
        raise ValueError("sensor_error_rate must be in [0, 1)")
    if output_root.exists():
        raise FileExistsError(output_root)

    source_maps = list_mapex_kth_maps(source_root)
    if max_maps is not None:
        if max_maps < 1:
            raise ValueError("max_maps must be at least one")
        source_maps = source_maps[:max_maps]

    episodes: list[NavigationMapEpisode] = []
    output_root.mkdir(parents=True)
    for map_dir in source_maps:
        map_id = map_dir.name
        target = _normalized_occupancy(np.load(map_dir / "occ_map.npy"))
        valid_space = np.load(map_dir / "valid_space.npy")
        if target.shape != valid_space.shape:
            raise ValueError(f"KTH target/valid-space shape mismatch for {map_id}")
        positions = _sample_free_positions(
            target, valid_space, candidate_count + 1, _stable_seed(map_id, seed)
        )
        start = positions[0]
        split = deterministic_mapex_kth_split(map_id, val_fraction)
        episode_root = output_root / "episodes" / map_id
        evidence_root = episode_root / "evidence"
        evidence_root.mkdir(parents=True)

        initial_path = episode_root / "initial_occupancy.npy"
        target_path = episode_root / "target_occupancy.npy"
        valid_path = episode_root / "valid_space.npy"
        np.save(initial_path, _local_reveal(target, valid_space, start, observation_radius))
        np.save(target_path, target)
        np.save(valid_path, valid_space)

        start_pose = _pose(*start, pixels_per_meter)
        evidence: list[NavigationEvidence] = []
        for index, position in enumerate(positions[1:]):
            evidence_path = evidence_root / f"candidate_{index:02d}.npy"
            np.save(
                evidence_path,
                _local_reveal(
                    target,
                    valid_space,
                    position,
                    observation_radius,
                    sensor_error_rate=sensor_error_rate,
                    rng=np.random.default_rng(_stable_seed(f"{map_id}:{index}", seed)),
                ),
            )
            candidate_pose = _pose(*position, pixels_per_meter)
            distance = float(
                np.hypot(candidate_pose.x - start_pose.x, candidate_pose.y - start_pose.y)
            )
            evidence.append(
                NavigationEvidence(
                    evidence_id=f"{map_id}-candidate-{index:02d}",
                    modality="occupancy_crop",
                    path=str(evidence_path.resolve()),
                    cost=max(distance, 1.0),
                    timestamp=index,
                    pose=candidate_pose,
                    footprint_radius_pixels=observation_radius,
                )
            )

        episodes.append(
            NavigationMapEpisode(
                schema_version="activemap-navigation-map-episode-v1",
                episode_id=f"mapex-kth-{map_id}",
                dataset="mapex_kth",
                domain="indoor",
                split=split,
                map_representation="occupancy_grid",
                initial_map_path=str(initial_path.resolve()),
                target_map_path=str(target_path.resolve()),
                start_pose=start_pose,
                evidence=evidence,
                budget=budget,
                test_assets_read=False,
                metadata={
                    "source_commit": MAPEX_KTH_SOURCE_COMMIT,
                    "source_map_id": map_id,
                    "source_occ_path": str((map_dir / "occ_map.npy").resolve()),
                    "valid_space_path": str(valid_path.resolve()),
                    "observation_model": (
                        "square_local_noisy_reveal_v1"
                        if sensor_error_rate > 0.0
                        else "square_local_ground_truth_reveal_v1"
                    ),
                    "occupancy_encoding": "free=0,unknown=0.5,occupied=1",
                    "observation_radius_pixels": observation_radius,
                    "pixels_per_meter": pixels_per_meter,
                    "candidate_count": candidate_count,
                    "construction_seed": seed,
                    "candidate_sensor_error_rate": sensor_error_rate,
                    "development_only": True,
                },
            )
        )

    manifest_root = output_root / "manifests"
    train_episodes = [episode for episode in episodes if episode.split == "train"]
    val_episodes = [episode for episode in episodes if episode.split == "val"]
    if train_episodes:
        write_navigation_map_jsonl(train_episodes, manifest_root / "train.jsonl")
    if val_episodes:
        write_navigation_map_jsonl(val_episodes, manifest_root / "val.jsonl")
    summary = {
        "schema_version": "activemap-mapex-kth-navigation-pilot-v1",
        "source_commit": MAPEX_KTH_SOURCE_COMMIT,
        "source_root": str(source_root.resolve()),
        "output_root": str(output_root.resolve()),
        "evaluation_role": "development_only_cross_domain_pilot",
        "test_assets_read": False,
        "map_count": len(episodes),
        "split_counts": {"train": len(train_episodes), "val": len(val_episodes), "test": 0},
        "candidate_count_per_episode": candidate_count,
        "observation_radius_pixels": observation_radius,
        "pixels_per_meter": pixels_per_meter,
        "budget": budget,
        "seed": seed,
        "observation_model": (
            "square_local_noisy_reveal_v1"
            if sensor_error_rate > 0.0
            else "square_local_ground_truth_reveal_v1"
        ),
        "candidate_sensor_error_rate": sensor_error_rate,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
