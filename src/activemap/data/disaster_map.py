"""Portable records for active disaster-map updating and route evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DisasterEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    modality: Literal["pre_event_rgb", "post_event_rgb", "foundation_map", "flood_mask"]
    path: str
    cost: float = Field(gt=0.0)
    timestamp: int = Field(ge=0)


class DisasterRouteQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_id: str
    start_node: str
    goal_node: str
    reference_travel_time: float | None = Field(default=None, gt=0.0)


class DisasterMapEpisode(BaseModel):
    """A disaster episode with editable infrastructure and optional routing queries."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["activemap-disaster-map-episode-v1"]
    episode_id: str
    dataset: Literal["spacenet8"]
    split: Literal["train", "val", "test"]
    aoi_id: str
    prior_map_path: str
    target_map_path: str
    evidence: list[DisasterEvidence]
    target_layers: list[
        Literal["building", "road", "flooded_building", "flooded_road", "road_speed"]
    ]
    route_queries: list[DisasterRouteQuery] = Field(default_factory=list)
    budget: float = Field(gt=0.0)
    test_assets_read: bool = False
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_episode(self) -> DisasterMapEpisode:
        if not self.evidence:
            raise ValueError("disaster episode requires evidence candidates")
        evidence_ids = [row.evidence_id for row in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("disaster evidence IDs must be unique")
        if not self.target_layers:
            raise ValueError("disaster episode requires target layers")
        if len(self.target_layers) != len(set(self.target_layers)):
            raise ValueError("disaster target layers must be unique")
        if self.test_assets_read != (self.split == "test"):
            raise ValueError("test_assets_read must match split")
        return self


def validate_disaster_map_jsonl(
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
                episode = DisasterMapEpisode.model_validate_json(line)
                if episode.split == "test" and not allow_test:
                    raise PermissionError("test disaster episode is locked")
                if check_paths:
                    paths = [
                        episode.prior_map_path,
                        episode.target_map_path,
                        *(row.path for row in episode.evidence),
                    ]
                    for raw_path in paths:
                        if not Path(raw_path).is_file():
                            raise FileNotFoundError(raw_path)
            except Exception as exc:
                errors.append(f"{path}:{line_number}: {exc}")
    if count == 0:
        errors.append(f"{path}: no disaster episodes")
    return count, errors


def write_disaster_map_jsonl(episodes: list[DisasterMapEpisode], path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for episode in sorted(episodes, key=lambda row: row.episode_id):
            handle.write(json.dumps(episode.model_dump(mode="json"), separators=(",", ":")) + "\n")


def spacenet8_tile_id(path: Path) -> str:
    """Return the three-field tile suffix shared by annotations and imagery."""
    fields = path.stem.split("_")
    if len(fields) < 3:
        raise ValueError(f"invalid SpaceNet8 filename: {path.name}")
    return "_".join(fields[-3:])


def match_spacenet8_assets(dataset_root: Path) -> list[dict[str, object]]:
    """Match annotations to all PRE/POST observations using the official tile suffix."""
    annotations = sorted((dataset_root / "annotations").glob("*.geojson"))
    pre_images = sorted((dataset_root / "PRE-event").glob("*.tif"))
    post_images = sorted((dataset_root / "POST-event").glob("*.tif"))
    pre_by_tile: dict[str, list[Path]] = {}
    post_by_tile: dict[str, list[Path]] = {}
    for path in pre_images:
        pre_by_tile.setdefault(spacenet8_tile_id(path), []).append(path)
    for path in post_images:
        post_by_tile.setdefault(spacenet8_tile_id(path), []).append(path)

    rows: list[dict[str, object]] = []
    for annotation in annotations:
        tile_id = spacenet8_tile_id(annotation)
        pre = pre_by_tile.get(tile_id, [])
        post = post_by_tile.get(tile_id, [])
        if len(pre) != 1 or not post:
            raise ValueError(
                f"tile {tile_id} has {len(pre)} PRE and {len(post)} POST observations"
            )
        rows.append(
            {
                "tile_id": tile_id,
                "annotation": annotation,
                "pre": pre[0],
                "post": post,
            }
        )
    return rows


def spacenet8_target_counts(target_path: Path) -> dict[str, int]:
    payload = json.loads(target_path.read_text(encoding="utf-8"))
    counts = {"features": 0, "roads": 0, "buildings": 0, "flooded": 0}
    for feature in payload.get("features", []):
        counts["features"] += 1
        properties = feature.get("properties", {})
        if properties.get("highway") is not None:
            counts["roads"] += 1
        if properties.get("building") is not None:
            counts["buildings"] += 1
        flooded = properties.get("flooded")
        if isinstance(flooded, str) and flooded.lower() in {"yes", "true", "1"}:
            counts["flooded"] += 1
    return counts


def write_spacenet8_prior(target_path: Path, prior_path: Path) -> dict[str, int]:
    """Create a pre-disaster prior by hiding target flood-state attributes."""
    payload = json.loads(target_path.read_text(encoding="utf-8"))
    counts = spacenet8_target_counts(target_path)
    for feature in payload.get("features", []):
        properties = feature.setdefault("properties", {})
        properties["flooded"] = None
    prior_path.parent.mkdir(parents=True, exist_ok=True)
    prior_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return counts


def deterministic_disaster_split(
    tile_id: str, val_fraction: float = 0.2
) -> Literal["train", "val"]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between zero and one")
    bucket = int(hashlib.sha256(tile_id.encode("ascii")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < val_fraction else "train"
