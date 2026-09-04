"""Causal RGB-D value model for active occupancy-map evidence acquisition."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn

from activemap.data.navigation_map import NavigationEvidence
from activemap.evaluation.navigation_rollout import (
    NavigationState,
    _unknown_footprint_count,
    unknown_coverage_policy,
)


@dataclass(frozen=True)
class NavigationVisualValueConfig:
    image_size: int = 96
    candidate_dim: int = 6
    base_channels: int = 16
    hidden_dim: int = 128
    dropout: float = 0.10

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


class _SpatialEncoder(nn.Module):
    def __init__(self, channels: int, base: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, base, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(4, base),
            nn.GELU(),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, base * 2),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, base * 4),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class NavigationVisualValueNet(nn.Module):
    """Estimate an ACQUIRE value from previously committed RGB-D and map state.

    The model intentionally accepts *current* RGB-D only. A candidate's RGB-D
    is acquired after selection and never appears in this forward input.
    """

    def __init__(self, config: NavigationVisualValueConfig) -> None:
        super().__init__()
        if config.image_size < 32:
            raise ValueError("image_size must be at least 32")
        if config.candidate_dim != 6:
            raise ValueError("candidate_dim must be six")
        self.config = config
        base = config.base_channels
        self.visual_encoder = _SpatialEncoder(4, base, config.dropout)
        self.map_encoder = _SpatialEncoder(1, base, config.dropout)
        spatial_dim = base * 4
        self.candidate_encoder = nn.Sequential(
            nn.LayerNorm(config.candidate_dim),
            nn.Linear(config.candidate_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.context_projection = nn.Sequential(
            nn.Linear(spatial_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, current_rgbd: Tensor, committed_map: Tensor, candidate: Tensor) -> Tensor:
        if current_rgbd.ndim != 4 or current_rgbd.shape[1] != 4:
            raise ValueError("current_rgbd must have shape [N,4,H,W]")
        if committed_map.ndim != 4 or committed_map.shape[1] != 1:
            raise ValueError("committed_map must have shape [N,1,H,W]")
        if candidate.ndim != 2 or candidate.shape[1] != self.config.candidate_dim:
            raise ValueError(f"candidate must have shape [N,{self.config.candidate_dim}]")
        if not (current_rgbd.shape[0] == committed_map.shape[0] == candidate.shape[0]):
            raise ValueError("visual, map, and candidate batches must align")
        context = self.context_projection(
            torch.cat((self.visual_encoder(current_rgbd), self.map_encoder(committed_map)), dim=1)
        )
        encoded_candidate = self.candidate_encoder(candidate)
        return self.value_head(
            torch.cat((context, encoded_candidate, context * encoded_candidate), dim=1)
        ).squeeze(1)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def current_rgbd_array(
    rgb_path: str | None,
    depth_path: str | None,
    *,
    image_size: int,
    max_depth_m: float = 5.0,
) -> np.ndarray:
    """Load the previously acquired RGB-D state, using zeros only when absent."""
    if image_size < 1 or max_depth_m <= 0.0:
        raise ValueError("image_size and max_depth_m must be positive")
    result = np.zeros((4, image_size, image_size), dtype=np.float32)
    if rgb_path is not None:
        path = Path(rgb_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        image = Image.open(path).convert("RGB").resize(
            (image_size, image_size), Image.Resampling.BILINEAR
        )
        result[:3] = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    if depth_path is not None:
        path = Path(depth_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        depth = np.asarray(np.load(path), dtype=np.float32)
        if depth.ndim != 2:
            raise ValueError(f"depth artifact must be 2D: {path}")
        normalized = np.nan_to_num(depth, nan=0.0, posinf=max_depth_m, neginf=0.0)
        normalized = np.clip(normalized / max_depth_m, 0.0, 1.0)
        depth_image = Image.fromarray(normalized, mode="F").resize(
            (image_size, image_size), Image.Resampling.BILINEAR
        )
        result[3] = np.asarray(depth_image, dtype=np.float32)
    return result


def occupancy_array(committed_map: np.ndarray, *, image_size: int) -> np.ndarray:
    """Resize the current editable occupancy map without revealing target content."""
    if committed_map.ndim != 2:
        raise ValueError("committed_map must be 2D")
    image = Image.fromarray(np.asarray(committed_map, dtype=np.float32), mode="F").resize(
        (image_size, image_size), Image.Resampling.NEAREST
    )
    return np.array(image, dtype=np.float32, copy=True)[None, ...]


def candidate_features(state: NavigationState, evidence: NavigationEvidence) -> np.ndarray:
    """Build pre-acquisition geometry and map-coverage features for one view."""
    if evidence.pose is None:
        raise ValueError("visual value policy requires candidate pose metadata")
    dx = float(evidence.pose.x - state.pose.x)
    dy = float(evidence.pose.y - state.pose.y)
    distance = float(np.hypot(dx, dy))
    yaw_delta = float(evidence.pose.yaw - state.pose.yaw)
    return np.asarray(
        (
            dx,
            dy,
            distance,
            np.sin(yaw_delta),
            np.cos(yaw_delta),
            float(_unknown_footprint_count(state, evidence)) / max(evidence.cost, 1e-6),
        ),
        dtype=np.float32,
    )


class NavigationVisualValuePolicy:
    """Deployable pre-observation policy with an explicit cost-aware STOP gate."""

    def __init__(self, checkpoint_path: str | Path, *, device: str = "cpu") -> None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("protocol") != "navigation_visual_value_v1":
            raise ValueError("unsupported navigation visual value checkpoint")
        self.config = NavigationVisualValueConfig(**checkpoint["model_config"])
        self.model = NavigationVisualValueNet(self.config)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.target_mean = float(checkpoint["target_mean"])
        self.target_std = float(checkpoint["target_std"])
        self.cost_weight = float(checkpoint["cost_weight"])
        self.stop_margin = float(checkpoint["stop_margin"])

    @torch.no_grad()
    def score(
        self, state: NavigationState, candidates: tuple[NavigationEvidence, ...]
    ) -> np.ndarray:
        if not candidates:
            return np.empty(0, dtype=np.float32)
        rgbd = current_rgbd_array(
            state.visual_rgb_path, state.visual_depth_path, image_size=self.config.image_size
        )
        occupancy = occupancy_array(state.committed_map, image_size=self.config.image_size)
        candidate = np.stack([candidate_features(state, row) for row in candidates])
        count = len(candidates)
        outputs = self.model(
            torch.from_numpy(np.repeat(rgbd[None, ...], count, axis=0)).to(self.device),
            torch.from_numpy(np.repeat(occupancy[None, ...], count, axis=0)).to(self.device),
            torch.from_numpy(candidate).to(self.device),
        )
        gain = outputs.cpu().numpy() * self.target_std + self.target_mean
        costs = np.asarray([row.cost for row in candidates], dtype=np.float32)
        return gain.astype(np.float32) - self.cost_weight * costs

    def __call__(
        self, state: NavigationState, candidates: tuple[NavigationEvidence, ...]
    ) -> str | None:
        scores = self.score(state, candidates)
        if not len(scores):
            return None
        index = int(np.argmax(scores))
        return candidates[index].evidence_id if float(scores[index]) > self.stop_margin else None


def select_prior_anchored_candidate(
    candidates: tuple[NavigationEvidence, ...],
    scores: np.ndarray,
    *,
    baseline_evidence_id: str,
    override_margin: float,
) -> str:
    """Override a strong map prior only when learned value clears a margin."""
    if override_margin < 0.0:
        raise ValueError("override_margin must be non-negative")
    if len(scores) != len(candidates):
        raise ValueError("candidate scores must align with candidates")
    indices = {row.evidence_id: index for index, row in enumerate(candidates)}
    if baseline_evidence_id not in indices:
        raise ValueError("baseline candidate is not available")
    baseline_index = indices[baseline_evidence_id]
    visual_index = int(np.argmax(scores))
    advantage = float(scores[visual_index] - scores[baseline_index])
    if visual_index != baseline_index and advantage > override_margin:
        return candidates[visual_index].evidence_id
    return baseline_evidence_id


class NavigationPriorAnchoredVisualValuePolicy:
    """Permit visual overrides only when they beat the geometry prior by a margin."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
        override_margin: float = 0.0,
    ) -> None:
        if override_margin < 0.0:
            raise ValueError("override_margin must be non-negative")
        self.visual = NavigationVisualValuePolicy(checkpoint_path, device=device)
        self.override_margin = override_margin

    def __call__(
        self, state: NavigationState, candidates: tuple[NavigationEvidence, ...]
    ) -> str | None:
        baseline_evidence_id = unknown_coverage_policy(state, candidates)
        if baseline_evidence_id is None:
            return None
        return select_prior_anchored_candidate(
            candidates,
            self.visual.score(state, candidates),
            baseline_evidence_id=baseline_evidence_id,
            override_margin=self.override_margin,
        )


def make_navigation_visual_value_policy(
    checkpoint_path: str | Path, *, device: str = "cpu"
) -> Callable[[NavigationState, tuple[NavigationEvidence, ...]], str | None]:
    return NavigationVisualValuePolicy(checkpoint_path, device=device)
