"""Learned residual fusion of tool outputs into the current map belief."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from activemap.agent.records import AgentBelief
from activemap.agent.tool_features import TOOL_RESULT_FEATURE_DIM, encode_tool_result
from activemap.geo_tools.records import GeoToolName, GeoToolResult
from activemap.models import EditOperation

BELIEF_FEATURE_DIM = 14
RECOMMENDATION_FEATURE_DIM = 5
ANCHORED_BELIEF_FEATURE_DIM = BELIEF_FEATURE_DIM + RECOMMENDATION_FEATURE_DIM


def encode_belief(belief: AgentBelief) -> list[float]:
    """Encode the complete public belief state used by the residual updater."""

    return [
        *belief.edit_probabilities,
        belief.confidence,
        belief.uncertainty,
        *belief.geometry_delta,
    ]


def encode_recommendation(belief: AgentBelief) -> list[float]:
    """Encode the deployed decision anchor separately from belief probabilities."""

    operations = list(EditOperation)
    predicted = belief.predicted_edit
    return [float(predicted == operation) for operation in operations] + [
        float(belief.recommended_edit is not None)
    ]


class ToolBeliefResidualNetwork(nn.Module):
    """Predict bounded corrections instead of replacing the current belief."""

    def __init__(self, *, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        input_dim = BELIEF_FEATURE_DIM + TOOL_RESULT_FEATURE_DIM
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.edit_head = nn.Linear(hidden_dim, 4)
        self.confidence_head = nn.Linear(hidden_dim, 1)
        self.geometry_head = nn.Linear(hidden_dim, 8)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, ...]:
        hidden = self.backbone(features)
        return (
            self.edit_head(hidden),
            self.confidence_head(hidden).squeeze(-1),
            self.geometry_head(hidden),
        )


class ToolPairBeliefResidualNetwork(nn.Module):
    """Fuse quality context and temporal evidence before changing public belief.

    ``reliability_gate`` makes the residual explicitly evidence-conditioned.  It is
    optional so existing checkpoints and the ungated ablation retain their exact
    architecture.  A conservative initial bias prevents untrained evidence from
    making a large change to the public belief.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        belief_feature_dim: int = BELIEF_FEATURE_DIM,
        reliability_gate: bool = False,
        gate_bias: float = -1.5,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if belief_feature_dim not in {BELIEF_FEATURE_DIM, ANCHORED_BELIEF_FEATURE_DIM}:
            raise ValueError("unsupported paired belief feature dimension")
        self.belief_feature_dim = belief_feature_dim
        self.reliability_gate = bool(reliability_gate)
        input_dim = belief_feature_dim + 2 * TOOL_RESULT_FEATURE_DIM
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.edit_head = nn.Linear(hidden_dim, 4)
        self.confidence_head = nn.Linear(hidden_dim, 1)
        self.geometry_head = nn.Linear(hidden_dim, 8)
        if self.reliability_gate:
            self.reliability_head = nn.Linear(hidden_dim, 1)
            nn.init.zeros_(self.reliability_head.weight)
            nn.init.constant_(self.reliability_head.bias, float(gate_bias))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, ...]:
        hidden = self.backbone(features)
        residuals = (
            self.edit_head(hidden),
            self.confidence_head(hidden).squeeze(-1),
            self.geometry_head(hidden),
        )
        if not self.reliability_gate:
            return residuals
        return (*residuals, torch.sigmoid(self.reliability_head(hidden)).squeeze(-1))


def apply_tool_belief_residual(
    belief_features: torch.Tensor,
    residuals: tuple[torch.Tensor, ...],
    *,
    max_logit_delta: float = 3.0,
    geometry_scale: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply bounded network outputs with the same math in training and inference."""

    if len(residuals) not in {3, 4}:
        raise ValueError("residuals must contain edit, confidence, geometry, and optional gate")
    edit_delta, confidence_delta, geometry_delta = residuals[:3]
    if len(residuals) == 4:
        reliability = residuals[3].clamp(0.0, 1.0)
        edit_delta = edit_delta * reliability.unsqueeze(-1)
        confidence_delta = confidence_delta * reliability
        geometry_delta = geometry_delta * reliability.unsqueeze(-1)
    probabilities = belief_features[..., :4].clamp_min(1e-7)
    logits = probabilities.log() + max_logit_delta * torch.tanh(edit_delta)
    updated_probabilities = torch.softmax(logits, dim=-1)
    confidence = belief_features[..., 4].clamp(1e-6, 1.0 - 1e-6)
    updated_confidence = torch.sigmoid(torch.logit(confidence) + confidence_delta)
    geometry = belief_features[..., 6:14]
    updated_geometry = geometry + geometry_scale * torch.tanh(geometry_delta)
    return updated_probabilities, updated_confidence, updated_geometry


class LearnedToolBeliefUpdater:
    """Apply a trained residual network after every tool result and replan."""

    def __init__(
        self,
        model: ToolBeliefResidualNetwork,
        *,
        device: str | torch.device | None = None,
        max_logit_delta: float = 3.0,
        geometry_scale: float = 0.25,
    ) -> None:
        if max_logit_delta <= 0.0:
            raise ValueError("max_logit_delta must be positive")
        if geometry_scale <= 0.0:
            raise ValueError("geometry_scale must be positive")
        self.device = (
            torch.device(device) if device is not None else next(model.parameters()).device
        )
        self.model = model.to(self.device).eval()
        self.max_logit_delta = float(max_logit_delta)
        self.geometry_scale = float(geometry_scale)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        device: str | torch.device = "cpu",
        hidden_dim: int | None = None,
        dropout: float | None = None,
    ) -> LearnedToolBeliefUpdater:
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        metadata = payload if isinstance(payload, dict) and "model_state_dict" in payload else {}
        if metadata:
            if metadata.get("belief_feature_dim", BELIEF_FEATURE_DIM) != BELIEF_FEATURE_DIM:
                raise ValueError("checkpoint belief feature dimension does not match runtime")
            checkpoint_tool_dim = metadata.get(
                "tool_result_feature_dim", TOOL_RESULT_FEATURE_DIM
            )
            if checkpoint_tool_dim != TOOL_RESULT_FEATURE_DIM:
                raise ValueError("checkpoint tool-result feature dimension does not match runtime")
        resolved_hidden_dim = int(hidden_dim or metadata.get("hidden_dim", 128))
        resolved_dropout = float(
            dropout if dropout is not None else metadata.get("dropout", 0.1)
        )
        model = ToolBeliefResidualNetwork(
            hidden_dim=resolved_hidden_dim, dropout=resolved_dropout
        )
        state_dict = metadata["model_state_dict"] if metadata else payload
        model.load_state_dict(state_dict)
        return cls(
            model,
            device=device,
            max_logit_delta=float(metadata.get("max_logit_delta", 3.0)),
            geometry_scale=float(metadata.get("geometry_scale", 0.25)),
        )

    def update(self, belief: AgentBelief, result: GeoToolResult) -> AgentBelief:
        features = torch.tensor(
            [encode_belief(belief) + encode_tool_result(result)],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.inference_mode():
            updated_probabilities, updated_confidence, updated_geometry = (
                apply_tool_belief_residual(
                    features[:, :BELIEF_FEATURE_DIM],
                    self.model(features),
                    max_logit_delta=self.max_logit_delta,
                    geometry_scale=self.geometry_scale,
                )
            )
        updated_probabilities = updated_probabilities[0]
        updated_confidence = updated_confidence[0]
        updated_geometry = updated_geometry[0]
        entropy = -torch.sum(updated_probabilities * updated_probabilities.clamp_min(1e-7).log())
        entropy /= math.log(updated_probabilities.numel())

        return AgentBelief(
            edit_probabilities=updated_probabilities.cpu().tolist(),
            confidence=float(updated_confidence.cpu()),
            geometry_delta=updated_geometry.cpu().tolist(),
            uncertainty=float(entropy.cpu()),
            recommended_edit=None,
        )


class PairedToolBeliefUpdater:
    """Apply one map update from quality context paired with temporal evidence."""

    def __init__(
        self,
        model: ToolPairBeliefResidualNetwork,
        *,
        device: str | torch.device | None = None,
        max_logit_delta: float = 1.5,
        geometry_scale: float = 0.15,
    ) -> None:
        if max_logit_delta <= 0.0:
            raise ValueError("max_logit_delta must be positive")
        if geometry_scale <= 0.0:
            raise ValueError("geometry_scale must be positive")
        self.device = (
            torch.device(device) if device is not None else next(model.parameters()).device
        )
        self.model = model.to(self.device).eval()
        self.max_logit_delta = float(max_logit_delta)
        self.geometry_scale = float(geometry_scale)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> PairedToolBeliefUpdater:
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        if not isinstance(payload, dict) or "model_state_dict" not in payload:
            raise ValueError("paired updater requires a metadata checkpoint")
        if payload.get("model_kind") not in {
            "paired_quality_temporal",
            "paired_quality_temporal_anchored",
            "post_acquisition_paired_quality_temporal",
        }:
            raise ValueError("checkpoint is not a paired quality-temporal model")
        if payload.get("belief_feature_dim") != BELIEF_FEATURE_DIM:
            raise ValueError("checkpoint belief feature dimension does not match runtime")
        if payload.get("tool_result_feature_dim") != TOOL_RESULT_FEATURE_DIM:
            raise ValueError("checkpoint tool-result feature dimension does not match runtime")
        model_belief_feature_dim = int(
            payload.get("model_belief_feature_dim", BELIEF_FEATURE_DIM)
        )
        model = ToolPairBeliefResidualNetwork(
            hidden_dim=int(payload.get("hidden_dim", 128)),
            dropout=float(payload.get("dropout", 0.1)),
            belief_feature_dim=model_belief_feature_dim,
            reliability_gate=bool(payload.get("reliability_gate", False)),
            gate_bias=float(payload.get("gate_bias", -1.5)),
        )
        model.load_state_dict(payload["model_state_dict"])
        return cls(
            model,
            device=device,
            max_logit_delta=float(payload.get("max_logit_delta", 1.5)),
            geometry_scale=float(payload.get("geometry_scale", 0.15)),
        )

    def update_pair(
        self,
        belief: AgentBelief,
        quality_result: GeoToolResult,
        temporal_result: GeoToolResult,
        *,
        use_quality: bool = True,
    ) -> AgentBelief:
        if quality_result.tool != GeoToolName.IMAGE_QUALITY:
            raise ValueError("quality_result must come from image_quality")
        if temporal_result.tool != GeoToolName.TEMPORAL_CHANGE:
            raise ValueError("temporal_result must come from temporal_change")
        quality_features = (
            encode_tool_result(quality_result)
            if use_quality
            else [0.0] * TOOL_RESULT_FEATURE_DIM
        )
        belief_values = encode_belief(belief)
        model_belief_values = list(belief_values)
        if self.model.belief_feature_dim == ANCHORED_BELIEF_FEATURE_DIM:
            model_belief_values.extend(encode_recommendation(belief))
        features = torch.tensor(
            [model_belief_values + quality_features + encode_tool_result(temporal_result)],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.inference_mode():
            updated_probabilities, updated_confidence, updated_geometry = (
                apply_tool_belief_residual(
                    torch.tensor(
                        [belief_values], dtype=torch.float32, device=self.device
                    ),
                    self.model(features),
                    max_logit_delta=self.max_logit_delta,
                    geometry_scale=self.geometry_scale,
                )
            )
        probabilities = updated_probabilities[0]
        confidence = updated_confidence[0]
        geometry = updated_geometry[0]
        entropy = -torch.sum(probabilities * probabilities.clamp_min(1e-7).log())
        entropy /= math.log(probabilities.numel())
        return AgentBelief(
            edit_probabilities=probabilities.cpu().tolist(),
            confidence=float(confidence.cpu()),
            geometry_delta=geometry.cpu().tolist(),
            uncertainty=float(entropy.cpu()),
            recommended_edit=None,
        )


class SequentialPairedToolBeliefUpdater:
    """Adapt paired quality/temporal fusion to sequential environment calls."""

    def __init__(self, updater: PairedToolBeliefUpdater, *, use_quality: bool = True) -> None:
        self.updater = updater
        self.use_quality = use_quality
        self.quality_results: dict[str, GeoToolResult] = {}
        self.matched_pairs = 0
        self.unmatched_temporal = 0
        self.redundant_quality = 0

    @staticmethod
    def _evidence_id(result: GeoToolResult) -> str | None:
        value = result.outputs.get("evidence_id")
        return str(value) if value is not None else None

    def update(self, belief: AgentBelief, result: GeoToolResult) -> AgentBelief:
        if not result.success:
            return belief
        evidence_id = self._evidence_id(result)
        if evidence_id is None:
            return belief
        if result.tool == GeoToolName.IMAGE_QUALITY:
            self.redundant_quality += int(evidence_id in self.quality_results)
            self.quality_results[evidence_id] = result
            return belief
        if result.tool != GeoToolName.TEMPORAL_CHANGE:
            return belief
        quality_result = self.quality_results.pop(evidence_id, None)
        if quality_result is None:
            self.unmatched_temporal += 1
            return belief
        self.matched_pairs += 1
        return self.updater.update_pair(
            belief,
            quality_result,
            result,
            use_quality=self.use_quality,
        )


class FrozenPriorSequentialPairedToolBeliefUpdater(SequentialPairedToolBeliefUpdater):
    """Ablation that fuses every evidence pair into the episode's initial belief."""

    def __init__(self, updater: PairedToolBeliefUpdater, *, use_quality: bool = True) -> None:
        super().__init__(updater, use_quality=use_quality)
        self.initial_belief: AgentBelief | None = None

    def update(self, belief: AgentBelief, result: GeoToolResult) -> AgentBelief:
        if self.initial_belief is None:
            self.initial_belief = belief.model_copy(deep=True)
        if not result.success:
            return belief
        evidence_id = self._evidence_id(result)
        if evidence_id is None:
            return belief
        if result.tool == GeoToolName.IMAGE_QUALITY:
            self.redundant_quality += int(evidence_id in self.quality_results)
            self.quality_results[evidence_id] = result
            return belief
        if result.tool != GeoToolName.TEMPORAL_CHANGE:
            return belief
        quality_result = self.quality_results.pop(evidence_id, None)
        if quality_result is None:
            self.unmatched_temporal += 1
            return belief
        self.matched_pairs += 1
        return self.updater.update_pair(
            self.initial_belief,
            quality_result,
            result,
            use_quality=self.use_quality,
        )


class IdentitySequentialPairedToolBeliefUpdater(SequentialPairedToolBeliefUpdater):
    """Execute and pair tools while leaving the controller belief unchanged."""

    def update(self, belief: AgentBelief, result: GeoToolResult) -> AgentBelief:
        if not result.success:
            return belief
        evidence_id = self._evidence_id(result)
        if evidence_id is None:
            return belief
        if result.tool == GeoToolName.IMAGE_QUALITY:
            self.redundant_quality += int(evidence_id in self.quality_results)
            self.quality_results[evidence_id] = result
            return belief
        if result.tool != GeoToolName.TEMPORAL_CHANGE:
            return belief
        quality_result = self.quality_results.pop(evidence_id, None)
        if quality_result is None:
            self.unmatched_temporal += 1
            return belief
        self.matched_pairs += 1
        return belief
