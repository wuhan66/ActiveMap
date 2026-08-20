"""Observable candidate-utility ranking for Active-Catalog controllers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from activemap.models import EditOperation

EDIT_VALUES = tuple(item.value for item in EditOperation)
CANDIDATE_FEATURE_NAMES = (
    "clear_fraction",
    "temporal_offset_normalized",
    "cost",
    "clear_per_cost",
    "scale_1",
    "scale_2",
    "scale_4",
)
CONTEXT_FEATURE_NAMES = tuple(
    [f"draft_edit_{value.lower()}" for value in EDIT_VALUES]
    + [f"belief_edit_{value.lower()}" for value in EDIT_VALUES]
    + [
        "belief_uncertainty",
        "draft_confidence",
        "remaining_budget",
        "selected_fraction",
    ]
)
RANKER_FEATURE_NAMES = CONTEXT_FEATURE_NAMES + CANDIDATE_FEATURE_NAMES


def observable_ranker_features(
    state: dict[str, Any], candidate: dict[str, Any]
) -> np.ndarray:
    """Encode only fields already exposed to the VLM in the SELECT prompt."""

    edit = str(state["direct_draft"]["edit"])
    edit_one_hot = [float(edit == value) for value in EDIT_VALUES]
    probabilities = [float(value) for value in state["belief"]["edit_probabilities"]]
    if len(probabilities) != len(EDIT_VALUES):
        raise ValueError("belief edit probabilities have the wrong dimension")
    scale = int(candidate["scale"])
    cost = float(candidate["cost"])
    if cost <= 0.0:
        raise ValueError("candidate cost must be positive")
    values = np.asarray(
        edit_one_hot
        + probabilities
        + [
            float(state["belief"]["uncertainty"]),
            float(state["direct_draft"]["confidence"]),
            float(state["budget"]["remaining"]),
            min(len(state.get("selected_evidence_ids", [])) / 16.0, 1.0),
            float(candidate["clear_fraction"]),
            float(candidate["temporal_offset_normalized"]),
            cost,
            float(candidate["clear_fraction"]) / cost,
            float(scale == 1),
            float(scale == 2),
            float(scale == 4),
        ],
        dtype=np.float32,
    )
    if values.shape != (len(RANKER_FEATURE_NAMES),) or not np.all(np.isfinite(values)):
        raise ValueError("invalid observable ranker feature vector")
    return values


@dataclass(frozen=True)
class CandidateUtilityRankerConfig:
    input_dim: int = len(RANKER_FEATURE_NAMES)
    hidden_dim: int = 128
    dropout: float = 0.10
    fusion_type: str = "concat"
    fusion_dim: int = 32
    metadata_dim: int = len(RANKER_FEATURE_NAMES)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class CandidateUtilityRanker(nn.Module):
    """Predict candidate utility gain relative to the STOP action."""

    def __init__(self, config: CandidateUtilityRankerConfig) -> None:
        super().__init__()
        self.config = config
        if config.fusion_type not in {"concat", "low_rank"}:
            raise ValueError("unsupported candidate ranker fusion type")
        if config.fusion_type == "low_rank":
            state_dim = config.input_dim - config.metadata_dim
            if state_dim <= 0:
                raise ValueError("low-rank fusion requires visual state features")
            self.metadata_projection = nn.Sequential(
                nn.Linear(config.metadata_dim, config.fusion_dim),
                nn.LayerNorm(config.fusion_dim),
                nn.GELU(),
            )
            self.state_projection = nn.Sequential(
                nn.Linear(state_dim, config.fusion_dim),
                nn.LayerNorm(config.fusion_dim),
                nn.GELU(),
            )
            network_input_dim = 3 * config.fusion_dim
        else:
            network_input_dim = config.input_dim
        self.network = nn.Sequential(
            nn.Linear(network_input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        if features.shape[-1] != self.config.input_dim:
            raise ValueError("candidate ranker input dimension mismatch")
        if self.config.fusion_type == "low_rank":
            metadata = self.metadata_projection(features[..., : self.config.metadata_dim])
            state = self.state_projection(features[..., self.config.metadata_dim :])
            features = torch.cat((metadata, state, metadata * state), dim=-1)
        return self.network(features).squeeze(-1)


class CandidateUtilityRankerPredictor:
    def __init__(self, checkpoint_path: str, device: str = "cpu") -> None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.model = CandidateUtilityRanker(
            CandidateUtilityRankerConfig(**checkpoint["model_config"])
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(device).eval()
        self.device = torch.device(device)
        self.safety_margin = float(checkpoint.get("safety_margin", 0.0))
        self.feature_mean = np.asarray(
            checkpoint.get("feature_mean", np.zeros(self.model.config.input_dim)),
            dtype=np.float32,
        )
        self.feature_std = np.asarray(
            checkpoint.get("feature_std", np.ones(self.model.config.input_dim)),
            dtype=np.float32,
        )
        if self.feature_mean.shape != (self.model.config.input_dim,):
            raise ValueError("candidate ranker feature mean has the wrong dimension")
        if self.feature_std.shape != (self.model.config.input_dim,):
            raise ValueError("candidate ranker feature std has the wrong dimension")
        self.state_feature_dim = self.model.config.input_dim - len(RANKER_FEATURE_NAMES)
        if self.state_feature_dim < 0:
            raise ValueError("candidate ranker input is smaller than metadata features")

    @torch.no_grad()
    def score_state(
        self, state: dict[str, Any], state_embedding: np.ndarray | None = None
    ) -> dict[str, float]:
        candidates = list(state["candidate_evidence"])
        if not candidates:
            return {}
        raw_features = np.stack(
            [observable_ranker_features(state, candidate) for candidate in candidates]
        )
        if self.state_feature_dim:
            if state_embedding is None:
                raise ValueError("visual candidate ranker requires a state embedding")
            embedding = np.asarray(state_embedding, dtype=np.float32).reshape(-1)
            if embedding.shape != (self.state_feature_dim,):
                raise ValueError("candidate ranker state embedding has the wrong dimension")
            raw_features = np.concatenate(
                [raw_features, np.repeat(embedding[None], len(candidates), axis=0)],
                axis=1,
            )
        features = (raw_features - self.feature_mean) / self.feature_std
        scores = self.model(torch.from_numpy(features).to(self.device)).cpu().numpy()
        return {
            str(candidate["evidence_id"]): float(score)
            for candidate, score in zip(candidates, scores, strict=True)
        }

    def decide(self, state: dict[str, Any]) -> tuple[str, str | None, float]:
        scores = self.score_state(state)
        if not scores:
            return "STOP", None, 0.0
        evidence_id = max(scores, key=lambda value: (scores[value], value))
        score = scores[evidence_id]
        if score <= self.safety_margin:
            return "STOP", None, score
        return "ACQUIRE", evidence_id, score
