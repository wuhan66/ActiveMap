"""A compact prior-conditioned U-Net with edit and geometry heads."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class UpdaterConfig:
    image_channels: int = 3
    prior_channels: int = 1
    base_channels: int = 32
    geometry_dim: int = 8
    edit_classes: int = 4
    dropout: float = 0.10
    use_prior: bool = True
    hierarchical_edit: bool = False
    auxiliary_edit_head: bool = True
    geometry_head_mode: str = "shared"
    prior_guided_roi: bool = False
    segmentation_evidence: bool = False
    vector_change_encoder: bool = False
    vector_change_to_edit_head: bool = False
    temporal_change_head: bool = False
    temporal_change_to_edit_head: bool = False
    temporal_spatial_edit_head: bool = False
    temporal_pair_input: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.GELU(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class PriorConditionedUNet(nn.Module):
    def __init__(self, config: UpdaterConfig) -> None:
        super().__init__()
        if config.temporal_pair_input and config.image_channels != 6:
            raise ValueError(
                "temporal_pair_input requires six image channels ordered as RGB_(t-1)||RGB_t"
            )
        if not config.auxiliary_edit_head and not config.hierarchical_edit:
            raise ValueError("auxiliary_edit_head=false requires hierarchical_edit=true")
        if config.geometry_head_mode not in {"shared", "edit_specific"}:
            raise ValueError("geometry_head_mode must be shared or edit_specific")
        if config.segmentation_evidence and not config.hierarchical_edit:
            raise ValueError("segmentation_evidence requires hierarchical_edit=true")
        if config.vector_change_encoder and not config.hierarchical_edit:
            raise ValueError("vector_change_encoder requires hierarchical_edit=true")
        if config.vector_change_encoder and not config.use_prior:
            raise ValueError("vector_change_encoder requires use_prior=true")
        if config.vector_change_to_edit_head and not config.vector_change_encoder:
            raise ValueError("vector_change_to_edit_head requires vector_change_encoder=true")
        if config.vector_change_to_edit_head and not config.auxiliary_edit_head:
            raise ValueError("vector_change_to_edit_head requires auxiliary_edit_head=true")
        if config.temporal_change_head and not config.use_prior:
            raise ValueError("temporal_change_head requires use_prior=true")
        if config.temporal_change_to_edit_head and not config.temporal_change_head:
            raise ValueError("temporal_change_to_edit_head requires temporal_change_head=true")
        if config.temporal_change_to_edit_head and not config.auxiliary_edit_head:
            raise ValueError("temporal_change_to_edit_head requires auxiliary_edit_head=true")
        if config.temporal_spatial_edit_head and not config.temporal_change_head:
            raise ValueError("temporal_spatial_edit_head requires temporal_change_head=true")
        if config.temporal_spatial_edit_head and not config.auxiliary_edit_head:
            raise ValueError("temporal_spatial_edit_head requires auxiliary_edit_head=true")
        self.config = config
        base = config.base_channels
        self.encoder1 = ConvBlock(config.image_channels + config.prior_channels, base, 0.0)
        self.encoder2 = ConvBlock(base, base * 2, config.dropout)
        self.encoder3 = ConvBlock(base * 2, base * 4, config.dropout)
        self.bottleneck = ConvBlock(base * 4, base * 8, config.dropout)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.decoder3 = ConvBlock(base * 8, base * 4, config.dropout)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.decoder2 = ConvBlock(base * 4, base * 2, config.dropout)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.decoder1 = ConvBlock(base * 2, base, 0.0)
        self.segmentation_head = nn.Conv2d(base, 1, 1)
        if config.temporal_change_head:
            self.temporal_change_head = nn.Conv2d(base, 2, 1)
        if config.temporal_change_to_edit_head:
            self.temporal_change_evidence_head = nn.Linear(6, config.edit_classes, bias=False)
            nn.init.zeros_(self.temporal_change_evidence_head.weight)
        if config.temporal_spatial_edit_head:
            spatial_base = max(base // 2, 8)
            self.temporal_spatial_edit_encoder = nn.Sequential(
                ConvBlock(3, spatial_base, 0.0),
                nn.MaxPool2d(2),
                ConvBlock(spatial_base, base, config.dropout),
                nn.MaxPool2d(2),
                ConvBlock(base, base * 2, config.dropout),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(base * 2, config.edit_classes, bias=False),
            )
            nn.init.zeros_(self.temporal_spatial_edit_encoder[-1].weight)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.shared_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base * 8, base * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        if config.prior_guided_roi:
            self.roi_projection = nn.Sequential(
                nn.Linear(base * 8, base * 4, bias=False),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            nn.init.zeros_(self.roi_projection[0].weight)
        if config.auxiliary_edit_head:
            self.edit_head = nn.Linear(base * 4, config.edit_classes)
        if config.hierarchical_edit:
            self.presence_head = nn.Linear(base * 4, 1)
            self.change_head = nn.Linear(base * 4, 1)
        if config.segmentation_evidence:
            self.segmentation_evidence_head = nn.Linear(4, 2, bias=False)
            nn.init.zeros_(self.segmentation_evidence_head.weight)
        if config.vector_change_encoder:
            self.change_image_encoder = nn.Sequential(
                ConvBlock(config.image_channels, base, 0.0),
                self.pool,
                ConvBlock(base, base * 2, config.dropout),
                self.pool,
                ConvBlock(base * 2, base * 4, config.dropout),
            )
            self.change_prior_encoder = nn.Sequential(
                ConvBlock(config.prior_channels, base, 0.0),
                self.pool,
                ConvBlock(base, base * 2, config.dropout),
                self.pool,
                ConvBlock(base * 2, base * 4, config.dropout),
            )
            self.change_fusion = nn.Sequential(
                nn.Conv2d(base * 16, base * 4, 1, bias=False),
                nn.BatchNorm2d(base * 4),
                nn.GELU(),
                ConvBlock(base * 4, base * 4, config.dropout),
            )
            self.change_projection = nn.Sequential(
                nn.Linear(base * 8, base * 4, bias=False),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
        geometry_outputs = config.geometry_dim
        if config.geometry_head_mode == "edit_specific":
            geometry_outputs *= config.edit_classes
        self.geometry_head = nn.Linear(base * 4, geometry_outputs)
        self.confidence_head = nn.Linear(base * 4, 1)

    @staticmethod
    def _align(upsampled: Tensor, skip: Tensor) -> Tensor:
        if upsampled.shape[-2:] == skip.shape[-2:]:
            return upsampled
        return nn.functional.interpolate(
            upsampled, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )

    @staticmethod
    def _masked_pool(features: Tensor, mask: Tensor) -> Tensor:
        numerator = torch.sum(features * mask, dim=(-2, -1))
        denominator = torch.clamp(torch.sum(mask, dim=(-2, -1)), min=1e-6)
        return numerator / denominator

    def _prior_roi_features(self, features: Tensor, prior_mask: Tensor) -> Tensor:
        prior = nn.functional.interpolate(
            prior_mask.float(), size=features.shape[-2:], mode="area"
        ).clamp(0.0, 1.0)
        dilated = nn.functional.max_pool2d(prior, kernel_size=3, stride=1, padding=1)
        ring = (dilated - prior).clamp(0.0, 1.0)
        inside = self._masked_pool(features, prior)
        context = self._masked_pool(features, ring)
        return torch.cat((inside, context), dim=1)

    def forward(self, image: Tensor, prior_mask: Tensor) -> dict[str, Tensor]:
        if not self.config.use_prior:
            prior_mask = torch.zeros_like(prior_mask)
        inputs = torch.cat([image, prior_mask], dim=1)
        level1 = self.encoder1(inputs)
        level2 = self.encoder2(self.pool(level1))
        level3 = self.encoder3(self.pool(level2))
        bottleneck = self.bottleneck(self.pool(level3))

        decoded3 = self.decoder3(
            torch.cat([self._align(self.up3(bottleneck), level3), level3], dim=1)
        )
        decoded2 = self.decoder2(
            torch.cat([self._align(self.up2(decoded3), level2), level2], dim=1)
        )
        decoded1 = self.decoder1(
            torch.cat([self._align(self.up1(decoded2), level1), level1], dim=1)
        )
        shared = self.shared_head(self.global_pool(bottleneck))
        if self.config.prior_guided_roi:
            shared = shared + self.roi_projection(self._prior_roi_features(level3, prior_mask))
        direct_segmentation_logits = self.segmentation_head(decoded1)
        segmentation_logits = direct_segmentation_logits
        outputs = {
            "segmentation_logits": segmentation_logits,
            "confidence_logits": self.confidence_head(shared).squeeze(-1),
            "shared_features": shared,
        }
        if self.config.auxiliary_edit_head:
            outputs["edit_logits"] = self.edit_head(shared)
        if self.config.temporal_change_head:
            temporal_change_logits = self.temporal_change_head(decoded1)
            prior_occupied = prior_mask >= 0.5
            segmentation_logits = torch.where(
                prior_occupied,
                -temporal_change_logits[:, 1:2],
                temporal_change_logits[:, 0:1],
            )
            outputs["direct_segmentation_logits"] = direct_segmentation_logits
            outputs["temporal_change_logits"] = temporal_change_logits
            outputs["segmentation_logits"] = segmentation_logits
        if self.config.hierarchical_edit:
            hierarchical_shared = shared
            if self.config.vector_change_encoder:
                image_change = self.change_image_encoder(image)
                prior_change = self.change_prior_encoder(prior_mask.float())
                change_features = self.change_fusion(
                    torch.cat(
                        (
                            image_change,
                            prior_change,
                            torch.abs(image_change - prior_change),
                            image_change * prior_change,
                        ),
                        dim=1,
                    )
                )
                change_descriptor = self._prior_roi_features(change_features, prior_mask)
                change_residual = self.change_projection(change_descriptor)
                hierarchical_shared = hierarchical_shared + change_residual
                outputs["change_descriptor"] = change_descriptor
                outputs["change_residual"] = change_residual
                if self.config.vector_change_to_edit_head:
                    outputs["edit_logits"] = self.edit_head(hierarchical_shared)
            presence_logits = self.presence_head(hierarchical_shared).squeeze(-1)
            change_logits = self.change_head(hierarchical_shared).squeeze(-1)
            if self.config.segmentation_evidence:
                evidence = segmentation_evidence_features(
                    outputs["segmentation_logits"], prior_mask
                )
                evidence_residual = self.segmentation_evidence_head(evidence)
                outputs["segmentation_evidence"] = evidence
                outputs["evidence_residual"] = evidence_residual
                presence_logits = presence_logits + evidence_residual[:, 0]
                change_logits = change_logits + evidence_residual[:, 1]
            outputs["presence_logits"] = presence_logits
            outputs["change_logits"] = change_logits
        if self.config.temporal_change_to_edit_head:
            temporal_evidence = temporal_change_evidence_features(
                outputs["temporal_change_logits"], prior_mask
            )
            temporal_edit_residual = self.temporal_change_evidence_head(temporal_evidence)
            outputs["temporal_change_evidence"] = temporal_evidence
            outputs["temporal_edit_residual"] = temporal_edit_residual
            outputs["edit_logits"] = outputs["edit_logits"] + temporal_edit_residual
        if self.config.temporal_spatial_edit_head:
            temporal_probabilities = torch.sigmoid(outputs["temporal_change_logits"])
            spatial_prior = prior_mask.float().clamp(0.0, 1.0)
            if spatial_prior.shape[-2:] != temporal_probabilities.shape[-2:]:
                spatial_prior = nn.functional.interpolate(
                    spatial_prior,
                    size=temporal_probabilities.shape[-2:],
                    mode="area",
                )
            temporal_spatial_input = torch.cat((temporal_probabilities, spatial_prior), dim=1)
            spatial_edit_residual = self.temporal_spatial_edit_encoder(temporal_spatial_input)
            outputs["temporal_spatial_input"] = temporal_spatial_input
            outputs["spatial_edit_residual"] = spatial_edit_residual
            outputs["edit_logits"] = outputs["edit_logits"] + spatial_edit_residual
        geometry_delta = self.geometry_head(shared)
        if self.config.geometry_head_mode == "edit_specific":
            geometry_by_edit = geometry_delta.reshape(
                image.shape[0], self.config.edit_classes, self.config.geometry_dim
            )
            outputs["geometry_delta_by_edit"] = geometry_by_edit
            edit_probabilities = operation_probabilities(outputs, prior_mask)
            geometry_delta = torch.sum(
                geometry_by_edit * edit_probabilities.unsqueeze(-1), dim=1
            )
        outputs["geometry_delta"] = geometry_delta
        return outputs

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def segmentation_evidence_features(segmentation_logits: Tensor, prior_mask: Tensor) -> Tensor:
    """Summarize predicted target occupancy relative to an editable prior object."""

    probabilities = torch.sigmoid(segmentation_logits)
    prior = prior_mask.float().clamp(0.0, 1.0)
    if prior.shape[-2:] != probabilities.shape[-2:]:
        prior = nn.functional.interpolate(prior, size=probabilities.shape[-2:], mode="area")
    dilated = nn.functional.max_pool2d(prior, kernel_size=9, stride=1, padding=4)
    ring = (dilated - prior).clamp(0.0, 1.0)
    dimensions = (-2, -1)
    intersection = torch.sum(probabilities * prior, dim=dimensions)
    prior_mass = torch.clamp(torch.sum(prior, dim=dimensions), min=1e-6)
    ring_mass = torch.clamp(torch.sum(ring, dim=dimensions), min=1e-6)
    probability_mass = torch.sum(probabilities, dim=dimensions)
    inside_occupancy = intersection / prior_mass
    ring_occupancy = torch.sum(probabilities * ring, dim=dimensions) / ring_mass
    soft_iou = intersection / torch.clamp(probability_mass + prior_mass - intersection, min=1e-6)
    contrast = inside_occupancy - ring_occupancy
    return torch.cat((inside_occupancy, ring_occupancy, soft_iou, contrast), dim=1)


def temporal_change_evidence_features(temporal_change_logits: Tensor, prior_mask: Tensor) -> Tensor:
    """Summarize sparse ADD/REMOVE evidence for learned operation classification."""

    if temporal_change_logits.ndim != 4 or temporal_change_logits.shape[1] != 2:
        raise ValueError("temporal_change_logits must have shape [N,2,H,W]")
    probabilities = torch.sigmoid(temporal_change_logits)
    prior = prior_mask.float().clamp(0.0, 1.0)
    if prior.shape[-2:] != probabilities.shape[-2:]:
        prior = nn.functional.interpolate(prior, size=probabilities.shape[-2:], mode="area")

    def moments(values: Tensor, domain: Tensor) -> Tensor:
        count = torch.clamp(domain.sum(dim=(-2, -1)), min=1.0)
        mean = (values * domain).sum(dim=(-2, -1)) / count
        rms = torch.sqrt((values.square() * domain).sum(dim=(-2, -1)) / count + 1e-8)
        maximum = (values * domain).amax(dim=(-2, -1))
        return torch.stack((mean, rms, maximum), dim=-1)

    add = moments(probabilities[:, 0], 1.0 - prior[:, 0])
    remove = moments(probabilities[:, 1], prior[:, 0])
    return torch.cat((add, remove), dim=-1)


def _masked_sample_mean(values: Tensor, valid_mask: Tensor) -> Tensor:
    """Average pixels per sample before averaging the batch.

    This prevents crops with more valid pixels from dominating the objective.
    """

    dimensions = tuple(range(1, values.ndim))
    numerator = torch.sum(values * valid_mask, dim=dimensions)
    denominator = torch.clamp(torch.sum(valid_mask, dim=dimensions), min=1.0)
    return (numerator / denominator).mean()


def combine_confidence_targets(
    segmentation_quality: Tensor,
    edit_quality: Tensor,
    *,
    mode: str = "mean",
) -> Tensor:
    """Combine geometry and typed-edit quality into confidence supervision."""

    if mode == "mean":
        return 0.5 * (segmentation_quality + edit_quality)
    if mode == "product":
        return segmentation_quality * edit_quality
    raise ValueError(f"unsupported confidence_target_mode: {mode}")


def dice_loss(logits: Tensor, target: Tensor, valid_mask: Tensor, epsilon: float = 1e-6) -> Tensor:
    """Soft Dice for occupied targets and foreground suppression for empty targets."""

    probabilities = torch.sigmoid(logits) * valid_mask
    target = target * valid_mask
    dimensions = tuple(range(1, logits.ndim))
    intersection = torch.sum(probabilities * target, dim=dimensions)
    probability_mass = torch.sum(probabilities, dim=dimensions)
    target_mass = torch.sum(target, dim=dimensions)
    dice = 1.0 - (2.0 * intersection + epsilon) / (probability_mass + target_mass + epsilon)
    valid_pixels = torch.clamp(torch.sum(valid_mask, dim=dimensions), min=1.0)
    empty_target = probability_mass / valid_pixels
    return torch.where(target_mass > 0, dice, empty_target).mean()


def _soft_erode(mask: Tensor) -> Tensor:
    horizontal = -nn.functional.max_pool2d(-mask, kernel_size=(3, 1), stride=1, padding=(1, 0))
    vertical = -nn.functional.max_pool2d(-mask, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.minimum(horizontal, vertical)


def _soft_dilate(mask: Tensor) -> Tensor:
    return nn.functional.max_pool2d(mask, kernel_size=3, stride=1, padding=1)


def _soft_skeleton(mask: Tensor, iterations: int) -> Tensor:
    opened = _soft_dilate(_soft_erode(mask))
    skeleton = torch.relu(mask - opened)
    for _ in range(iterations):
        mask = _soft_erode(mask)
        opened = _soft_dilate(_soft_erode(mask))
        delta = torch.relu(mask - opened)
        skeleton = skeleton + torch.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice_loss(
    logits: Tensor,
    target: Tensor,
    valid_mask: Tensor,
    *,
    iterations: int = 5,
    epsilon: float = 1e-6,
) -> Tensor:
    """Differentiable centerline Dice loss for thin connected structures."""

    if iterations < 0:
        raise ValueError("clDice iterations must be non-negative")
    probabilities = torch.sigmoid(logits) * valid_mask
    target = target * valid_mask
    predicted_skeleton = _soft_skeleton(probabilities, iterations) * valid_mask
    target_skeleton = _soft_skeleton(target, iterations) * valid_mask
    dimensions = tuple(range(1, logits.ndim))
    topology_precision = (
        torch.sum(predicted_skeleton * target, dim=dimensions) + epsilon
    ) / (torch.sum(predicted_skeleton, dim=dimensions) + epsilon)
    topology_sensitivity = (
        torch.sum(target_skeleton * probabilities, dim=dimensions) + epsilon
    ) / (torch.sum(target_skeleton, dim=dimensions) + epsilon)
    score = (
        2.0 * topology_precision * topology_sensitivity + epsilon
    ) / (topology_precision + topology_sensitivity + epsilon)
    return (1.0 - score).mean()


def focal_loss(
    logits: Tensor,
    target: Tensor,
    valid_mask: Tensor,
    *,
    gamma: float = 2.0,
    foreground_alpha: float = 0.75,
) -> Tensor:
    """Masked binary focal loss for sparse changed-map foregrounds."""

    probabilities = torch.sigmoid(logits)
    probability_true = probabilities * target + (1.0 - probabilities) * (1.0 - target)
    alpha = foreground_alpha * target + (1.0 - foreground_alpha) * (1.0 - target)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    values = alpha * torch.pow(1.0 - probability_true, gamma) * bce
    return _masked_sample_mean(values, valid_mask)


def operation_probabilities(
    outputs: dict[str, Tensor],
    prior_mask: Tensor | None = None,
) -> Tensor:
    """Return KEEP/ADD/DELETE/RESHAPE probabilities with optional hierarchy."""

    auxiliary = (
        torch.softmax(outputs["edit_logits"], dim=-1)
        if "edit_logits" in outputs
        else None
    )
    if "presence_logits" not in outputs or "change_logits" not in outputs or prior_mask is None:
        if auxiliary is None:
            raise ValueError("operation probabilities require an edit head or hierarchical logits")
        return auxiliary
    presence = torch.sigmoid(outputs["presence_logits"])
    change = torch.sigmoid(outputs["change_logits"])
    with_prior = torch.stack(
        (
            presence * (1.0 - change),
            torch.zeros_like(presence),
            1.0 - presence,
            presence * change,
        ),
        dim=-1,
    )
    has_prior = prior_mask.flatten(1).amax(dim=1) > 0.5
    without_prior = torch.stack(
        (
            1.0 - presence,
            presence,
            torch.zeros_like(presence),
            torch.zeros_like(presence),
        ),
        dim=-1,
    )
    no_prior_probabilities = auxiliary if auxiliary is not None else without_prior
    return torch.where(has_prior[:, None], with_prior, no_prior_probabilities)


def hierarchical_edit_predictions(
    outputs: dict[str, Tensor],
    prior_mask: Tensor,
    *,
    presence_threshold: float | None = None,
    change_threshold: float = 0.5,
) -> Tensor:
    """Predict edit classes, optionally using calibrated hierarchy thresholds."""

    if presence_threshold is None or "presence_logits" not in outputs:
        return torch.argmax(operation_probabilities(outputs, prior_mask), dim=-1)
    probabilities = operation_probabilities(outputs, prior_mask)
    auxiliary = torch.argmax(probabilities, dim=-1)
    has_prior = prior_mask.flatten(1).amax(dim=1) > 0.5
    presence = torch.sigmoid(outputs["presence_logits"])
    change = torch.sigmoid(outputs["change_logits"])
    structured = torch.where(
        presence < presence_threshold,
        torch.full_like(auxiliary, 2),
        torch.where(
            change < change_threshold,
            torch.zeros_like(auxiliary),
            torch.full_like(auxiliary, 3),
        ),
    )
    if "edit_logits" in outputs:
        no_prior = auxiliary
    else:
        no_prior = torch.where(
            presence < presence_threshold,
            torch.zeros_like(auxiliary),
            torch.ones_like(auxiliary),
        )
    return torch.where(has_prior, structured, no_prior)


def updater_loss(
    outputs: dict[str, Tensor],
    *,
    target_mask: Tensor,
    valid_mask: Tensor,
    edit_target: Tensor,
    geometry_target: Tensor,
    segmentation_weight: float = 1.0,
    segmentation_bce_weight: float = 1.0,
    segmentation_dice_weight: float = 1.0,
    segmentation_focal_weight: float = 0.25,
    segmentation_cldice_weight: float = 0.0,
    cldice_iterations: int = 5,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.75,
    edit_weight: float = 1.0,
    edit_class_weights: Tensor | None = None,
    edit_label_smoothing: float = 0.05,
    geometry_weight: float = 0.5,
    geometry_beta: float = 0.1,
    false_edit_weight: float = 0.5,
    missed_edit_weight: float = 0.25,
    confidence_weight: float = 0.2,
    confidence_target_mode: str = "mean",
    presence_weight: float = 0.0,
    change_weight: float = 0.0,
    prior_mask: Tensor | None = None,
    full_scene_mask: Tensor | None = None,
    temporal_supervision_mask: Tensor | None = None,
    temporal_change_weight: float = 0.0,
    temporal_change_bce_weight: float = 1.0,
    temporal_change_dice_weight: float = 1.0,
    temporal_change_focal_weight: float = 0.5,
    temporal_change_focal_alpha: float = 0.9,
) -> tuple[Tensor, dict[str, Tensor]]:
    pixel_values = nn.functional.binary_cross_entropy_with_logits(
        outputs["segmentation_logits"], target_mask, reduction="none"
    )
    pixel_loss = _masked_sample_mean(pixel_values, valid_mask)
    dice = dice_loss(outputs["segmentation_logits"], target_mask, valid_mask)
    focal = focal_loss(
        outputs["segmentation_logits"],
        target_mask,
        valid_mask,
        gamma=focal_gamma,
        foreground_alpha=focal_alpha,
    )
    cldice = soft_cldice_loss(
        outputs["segmentation_logits"],
        target_mask,
        valid_mask,
        iterations=cldice_iterations,
    )
    segmentation = (
        segmentation_bce_weight * pixel_loss
        + segmentation_dice_weight * dice
        + segmentation_focal_weight * focal
        + segmentation_cldice_weight * cldice
    )
    zero = outputs["segmentation_logits"].sum() * 0.0
    if "edit_logits" in outputs:
        edit = nn.functional.cross_entropy(
            outputs["edit_logits"],
            edit_target,
            weight=edit_class_weights,
            label_smoothing=edit_label_smoothing,
        )
    else:
        edit = zero
    presence = zero
    change = zero
    temporal_change_bce = zero
    temporal_change_dice = zero
    temporal_change_focal = zero
    temporal_change = zero
    temporal_add = zero
    temporal_remove = zero
    object_mask = (
        ~full_scene_mask.bool()
        if full_scene_mask is not None
        else torch.ones_like(edit_target, dtype=torch.bool)
    )
    if "presence_logits" in outputs:
        presence_mask = object_mask & (edit_target != 1)
        if presence_mask.any():
            presence_target = (edit_target != 2).float()
            presence = nn.functional.binary_cross_entropy_with_logits(
                outputs["presence_logits"][presence_mask],
                presence_target[presence_mask],
            )
    if "change_logits" in outputs:
        change_mask = object_mask & ((edit_target == 0) | (edit_target == 3))
        if change_mask.any():
            change_target = (edit_target == 3).float()
            change = nn.functional.binary_cross_entropy_with_logits(
                outputs["change_logits"][change_mask],
                change_target[change_mask],
            )
    geometry_mask = (edit_target == 1) | (edit_target == 3)
    geometry_predictions = outputs["geometry_delta"]
    if "geometry_delta_by_edit" in outputs:
        geometry_predictions = outputs["geometry_delta_by_edit"][
            torch.arange(edit_target.shape[0], device=edit_target.device), edit_target
        ]
    if geometry_mask.any():
        geometry = nn.functional.smooth_l1_loss(
            geometry_predictions[geometry_mask],
            geometry_target[geometry_mask],
            beta=geometry_beta,
        )
    else:
        geometry = outputs["geometry_delta"].sum() * 0.0
    probabilities = operation_probabilities(outputs, prior_mask)
    if full_scene_mask is not None and full_scene_mask.any() and "edit_logits" in outputs:
        auxiliary_probabilities = torch.softmax(outputs["edit_logits"], dim=-1)
        probabilities = torch.where(
            full_scene_mask.bool()[:, None], auxiliary_probabilities, probabilities
        )
    temporal_mask = (
        temporal_supervision_mask.bool()
        if temporal_supervision_mask is not None
        else (full_scene_mask.bool() if full_scene_mask is not None else None)
    )
    if (
        temporal_change_weight > 0.0
        and prior_mask is not None
        and temporal_mask is not None
        and temporal_mask.any()
    ):
        selected = temporal_mask
        selected_prior = prior_mask[selected].float().clamp(0.0, 1.0)
        selected_target = target_mask[selected]
        selected_valid = valid_mask[selected]
        if "temporal_change_logits" in outputs:
            selected_logits = outputs["temporal_change_logits"][selected]
            add_target = selected_target * (1.0 - selected_prior)
            remove_target = (1.0 - selected_target) * selected_prior
            add_valid = selected_valid * (1.0 - selected_prior)
            remove_valid = selected_valid * selected_prior

            def channel_loss(
                logits: Tensor, target: Tensor, channel_valid: Tensor
            ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
                values = nn.functional.binary_cross_entropy_with_logits(
                    logits, target, reduction="none"
                )
                bce_value = _masked_sample_mean(values, channel_valid)
                dice_value = dice_loss(logits, target, channel_valid)
                focal_value = focal_loss(
                    logits,
                    target,
                    channel_valid,
                    gamma=focal_gamma,
                    foreground_alpha=temporal_change_focal_alpha,
                )
                combined = (
                    temporal_change_bce_weight * bce_value
                    + temporal_change_dice_weight * dice_value
                    + temporal_change_focal_weight * focal_value
                )
                return combined, bce_value, dice_value, focal_value

            temporal_add, add_bce, add_dice, add_focal = channel_loss(
                selected_logits[:, 0:1], add_target, add_valid
            )
            temporal_remove, remove_bce, remove_dice, remove_focal = channel_loss(
                selected_logits[:, 1:2], remove_target, remove_valid
            )
            temporal_change_bce = 0.5 * (add_bce + remove_bce)
            temporal_change_dice = 0.5 * (add_dice + remove_dice)
            temporal_change_focal = 0.5 * (add_focal + remove_focal)
            temporal_change = 0.5 * (temporal_add + temporal_remove)
        else:
            temporal_target = torch.abs(selected_target - selected_prior)
            temporal_logits = outputs["segmentation_logits"][selected] * (
                1.0 - 2.0 * selected_prior
            )
            temporal_values = nn.functional.binary_cross_entropy_with_logits(
                temporal_logits, temporal_target, reduction="none"
            )
            temporal_change_bce = _masked_sample_mean(temporal_values, selected_valid)
            temporal_change_dice = dice_loss(temporal_logits, temporal_target, selected_valid)
            temporal_change_focal = focal_loss(
                temporal_logits,
                temporal_target,
                selected_valid,
                gamma=focal_gamma,
                foreground_alpha=temporal_change_focal_alpha,
            )
            temporal_change = (
                temporal_change_bce_weight * temporal_change_bce
                + temporal_change_dice_weight * temporal_change_dice
                + temporal_change_focal_weight * temporal_change_focal
            )
    keep_mask = edit_target == 0
    if keep_mask.any():
        false_edit = probabilities[keep_mask, 1:].sum(dim=-1).mean()
    else:
        false_edit = probabilities.sum() * 0.0
    changed_mask = ~keep_mask
    if changed_mask.any():
        missed_edit = probabilities[changed_mask, 0].mean()
    else:
        missed_edit = probabilities.sum() * 0.0
    with torch.no_grad():
        mask_probabilities = torch.sigmoid(outputs["segmentation_logits"]) * valid_mask
        valid_target = target_mask * valid_mask
        dimensions = tuple(range(1, target_mask.ndim))
        intersection = torch.sum(mask_probabilities * valid_target, dim=dimensions)
        union = torch.sum(
            mask_probabilities + valid_target - mask_probabilities * valid_target,
            dim=dimensions,
        )
        soft_iou = (intersection + 1e-6) / (union + 1e-6)
        target_class_probability = torch.gather(probabilities, 1, edit_target[:, None]).squeeze(1)
        confidence_target = combine_confidence_targets(
            soft_iou,
            target_class_probability,
            mode=confidence_target_mode,
        )
    confidence = nn.functional.binary_cross_entropy_with_logits(
        outputs["confidence_logits"], confidence_target
    )
    total = (
        segmentation_weight * segmentation
        + edit_weight * edit
        + presence_weight * presence
        + change_weight * change
        + geometry_weight * geometry
        + false_edit_weight * false_edit
        + missed_edit_weight * missed_edit
        + confidence_weight * confidence
        + temporal_change_weight * temporal_change
    )
    return total, {
        "segmentation_bce": pixel_loss.detach(),
        "segmentation_dice": dice.detach(),
        "segmentation_focal": focal.detach(),
        "segmentation_cldice": cldice.detach(),
        "segmentation": segmentation.detach(),
        "edit": edit.detach(),
        "presence": presence.detach(),
        "change": change.detach(),
        "geometry": geometry.detach(),
        "false_edit": false_edit.detach(),
        "missed_edit": missed_edit.detach(),
        "confidence": confidence.detach(),
        "temporal_change_bce": temporal_change_bce.detach(),
        "temporal_change_dice": temporal_change_dice.detach(),
        "temporal_change_focal": temporal_change_focal.detach(),
        "temporal_change": temporal_change.detach(),
        "temporal_add": temporal_add.detach(),
        "temporal_remove": temporal_remove.detach(),
    }
