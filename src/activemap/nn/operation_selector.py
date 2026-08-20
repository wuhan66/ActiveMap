"""Independent typed-operation selector over frozen updater evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

from activemap.nn.updater import temporal_change_evidence_features


@dataclass(frozen=True)
class OperationSelectorConfig:
    context_dim: int
    spatial_channels: int = 3
    spatial_size: int = 64
    base_channels: int = 32
    hidden_dim: int = 256
    edit_classes: int = 4
    dropout: float = 0.20
    use_spatial: bool = True
    use_context: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class OperationSelector(nn.Module):
    """Classify KEEP/ADD/DELETE/RESHAPE without modifying the updater."""

    def __init__(self, config: OperationSelectorConfig) -> None:
        super().__init__()
        if config.context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if config.base_channels < 4 or config.base_channels % 4:
            raise ValueError("base_channels must be a multiple of four")
        base = config.base_channels
        self.config = config
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(config.spatial_channels, base, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(4, base),
            nn.GELU(),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, base * 2),
            nn.GELU(),
            nn.Conv2d(base * 2, base * 2, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, base * 2),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        spatial_dim = base * 2 * 4 * 4
        self.context_encoder = nn.Sequential(
            nn.LayerNorm(config.context_dim),
            nn.Linear(config.context_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(spatial_dim + config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.edit_classes),
        )

    def forward(self, spatial_evidence: Tensor, context: Tensor) -> Tensor:
        if spatial_evidence.ndim != 4:
            raise ValueError("spatial_evidence must have shape [N,C,H,W]")
        if context.ndim != 2 or context.shape[1] != self.config.context_dim:
            raise ValueError(f"context must have shape [N,{self.config.context_dim}]")
        spatial = self.spatial_encoder(spatial_evidence)
        encoded_context = self.context_encoder(context)
        if not self.config.use_spatial:
            spatial = torch.zeros_like(spatial)
        if not self.config.use_context:
            encoded_context = torch.zeros_like(encoded_context)
        return self.classifier(torch.cat((spatial, encoded_context), dim=1))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def operation_selector_inputs(
    outputs: dict[str, Tensor],
    prior_mask: Tensor,
    *,
    spatial_size: int,
) -> tuple[Tensor, Tensor]:
    """Build inference-only selector inputs from a frozen updater forward pass."""

    if "temporal_change_logits" not in outputs:
        raise ValueError("operation selection requires explicit temporal change logits")
    if "shared_features" not in outputs:
        raise ValueError("updater outputs must include shared_features")
    probabilities = torch.sigmoid(outputs["temporal_change_logits"])
    prior = prior_mask.float().clamp(0.0, 1.0)
    if prior.shape[-2:] != probabilities.shape[-2:]:
        prior = nn.functional.interpolate(prior, probabilities.shape[-2:], mode="area")
    spatial = torch.cat((probabilities, prior), dim=1)
    spatial = nn.functional.interpolate(spatial, size=(spatial_size, spatial_size), mode="area")
    temporal_statistics = temporal_change_evidence_features(
        outputs["temporal_change_logits"], prior_mask
    )
    pieces = [
        outputs["shared_features"],
        outputs["change_descriptor"],
        outputs["geometry_delta"],
        outputs["confidence_logits"].unsqueeze(1),
        outputs["edit_logits"],
        temporal_statistics,
    ]
    return spatial, torch.cat(pieces, dim=1)
