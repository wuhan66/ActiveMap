"""Portable records for indoor and outdoor active-mapping navigation pilots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class NavigationPose(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: float
    y: float
    yaw: float


class NavigationEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    modality: Literal["camera", "lidar", "occupancy_crop", "map_prediction", "raycast"]
    path: str
    cost: float = Field(gt=0.0)
    timestamp: int = Field(ge=0)


class NavigationMapEpisode(BaseModel):
    """Domain-neutral active navigation episode with auditable map artifacts."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["activemap-navigation-map-episode-v1"]
    episode_id: str
    dataset: str
    domain: Literal["indoor", "outdoor"]
    split: Literal["train", "val", "test"]
    map_representation: Literal["occupancy_grid", "vector_hd_map"]
    initial_map_path: str
    target_map_path: str
    start_pose: NavigationPose
    evidence: list[NavigationEvidence]
    budget: float = Field(gt=0.0)
    test_assets_read: bool = False
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_episode(self) -> "NavigationMapEpisode":
        if not self.evidence:
            raise ValueError("navigation episode requires evidence candidates")
        ids = [row.evidence_id for row in self.evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("navigation evidence IDs must be unique")
        if self.test_assets_read != (self.split == "test"):
            raise ValueError("test_assets_read must match split")
        return self


def validate_navigation_map_jsonl(
    path: Path, *, allow_test: bool = False, check_paths: bool = False
) -> tuple[int, list[str]]:
    errors: list[str] = []
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            count += 1
            try:
                episode = NavigationMapEpisode.model_validate_json(line)
                if episode.split == "test" and not allow_test:
                    raise PermissionError("test navigation episode is locked")
                if check_paths:
                    paths = [
                        episode.initial_map_path,
                        episode.target_map_path,
                        *(row.path for row in episode.evidence),
                    ]
                    for raw_path in paths:
                        if not Path(raw_path).is_file():
                            raise FileNotFoundError(raw_path)
            except Exception as exc:
                errors.append(f"{path}:{line_number}: {exc}")
    if count == 0:
        errors.append(f"{path}: no navigation episodes")
    return count, errors


def write_navigation_map_jsonl(episodes: list[NavigationMapEpisode], path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for episode in sorted(episodes, key=lambda row: row.episode_id):
            handle.write(json.dumps(episode.model_dump(mode="json"), separators=(",", ":")) + "\n")
