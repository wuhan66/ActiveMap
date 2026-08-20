"""Structured terminal-action adapter applied after Tool-Belief updates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from activemap.agent.records import AgentBelief
from activemap.agent.tool_belief_model import BELIEF_FEATURE_DIM, encode_belief
from activemap.agent.tool_features import (
    SEMANTIC_TOOL_FEATURE_DIM,
    TOOL_RESULT_FEATURE_DIM,
    encode_semantic_tool_result,
    encode_tool_result,
)
from activemap.geo_tools.records import GeoToolName, GeoToolResult
from activemap.models import EditOperation

UPDATE_OPERATIONS = tuple(
    operation for operation in EditOperation if operation != EditOperation.KEEP
)
TOOL_CONDITIONED_FEATURE_DIM = BELIEF_FEATURE_DIM + 2 * TOOL_RESULT_FEATURE_DIM
SEMANTIC_CONDITIONED_FEATURE_DIM = BELIEF_FEATURE_DIM + SEMANTIC_TOOL_FEATURE_DIM
SEMANTIC_ONLY_FEATURE_DIM = SEMANTIC_TOOL_FEATURE_DIM


def encode_post_tool_action_features(
    belief: AgentBelief,
    tool_history: list[GeoToolResult],
) -> list[float]:
    latest = {result.tool: result for result in tool_history}
    zeros = [0.0] * TOOL_RESULT_FEATURE_DIM
    quality = latest.get(GeoToolName.IMAGE_QUALITY)
    temporal = latest.get(GeoToolName.TEMPORAL_CHANGE)
    return (
        encode_belief(belief)
        + (encode_tool_result(quality) if quality is not None else zeros)
        + (encode_tool_result(temporal) if temporal is not None else zeros)
    )


def encode_semantic_action_features(
    belief: AgentBelief,
    semantic_result: GeoToolResult,
) -> list[float]:
    return encode_belief(belief) + encode_semantic_tool_result(semantic_result)


@dataclass(frozen=True)
class PostToolActionAdapterConfig:
    input_dim: int = BELIEF_FEATURE_DIM
    hidden_dim: int = 128
    dropout: float = 0.1
    include_tool_results: bool = False
    include_semantic_result: bool = False
    semantic_only: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class PostToolActionAdapter(nn.Module):
    """Hierarchical KEEP/UPDATE and edit-operation policy."""

    def __init__(self, config: PostToolActionAdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.update_head = nn.Linear(config.hidden_dim, 1)
        self.operation_head = nn.Linear(config.hidden_dim, len(UPDATE_OPERATIONS))

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        if features.shape[-1] != self.config.input_dim:
            raise ValueError("post-tool adapter feature dimension mismatch")
        hidden = self.encoder(features)
        return self.update_head(hidden).squeeze(-1), self.operation_head(hidden)


class PostToolActionAdapterPredictor:
    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if payload.get("protocol") != "post_tool_action_adapter_v1":
            raise ValueError("unsupported post-tool action adapter checkpoint")
        self.model = PostToolActionAdapter(
            PostToolActionAdapterConfig(**payload["model_config"])
        )
        self.model.load_state_dict(payload["state_dict"])
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.feature_mean = np.asarray(payload["feature_mean"], dtype=np.float32)
        self.feature_std = np.asarray(payload["feature_std"], dtype=np.float32)
        self.update_threshold = float(payload["update_threshold"])

    @torch.no_grad()
    def predict(
        self,
        belief: AgentBelief,
        tool_history: list[GeoToolResult] | None = None,
    ) -> EditOperation:
        if self.model.config.semantic_only:
            semantic = next(
                (
                    result
                    for result in reversed(tool_history or [])
                    if result.tool == GeoToolName.RASTER_SEGMENT
                ),
                None,
            )
            if semantic is None:
                raise ValueError("semantic adapter requires RASTER_SEGMENT history")
            features = np.asarray(
                encode_semantic_tool_result(semantic),
                dtype=np.float32,
            )
        elif self.model.config.include_tool_results:
            features = np.asarray(
                encode_post_tool_action_features(belief, tool_history or []),
                dtype=np.float32,
            )
        elif self.model.config.include_semantic_result:
            semantic = next(
                (
                    result
                    for result in reversed(tool_history or [])
                    if result.tool == GeoToolName.RASTER_SEGMENT
                ),
                None,
            )
            if semantic is None:
                raise ValueError("semantic adapter requires RASTER_SEGMENT history")
            features = np.asarray(
                encode_semantic_action_features(belief, semantic),
                dtype=np.float32,
            )
        else:
            features = np.asarray(encode_belief(belief), dtype=np.float32)
        features = (features - self.feature_mean) / self.feature_std
        update_logit, operation_logits = self.model(
            torch.from_numpy(features).to(self.device).unsqueeze(0)
        )
        if float(torch.sigmoid(update_logit).item()) < self.update_threshold:
            return EditOperation.KEEP
        operation_index = int(operation_logits.argmax(dim=-1).item())
        return UPDATE_OPERATIONS[operation_index]
