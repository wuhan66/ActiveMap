"""Prior-conditioned current/add/remove heads for editable road-map updates."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from activemap.integrations.sam_road import SAMRoadPredictor
from activemap.models import EditOperation


@dataclass(frozen=True)
class ChangeLossWeights:
    current: float = 0.5
    add: float = 1.0
    remove: float = 1.0
    reconstruction: float = 0.5
    unchanged_safety: float = 1.0
    exclusivity: float = 0.1
    operation: float = 1.0


DEFAULT_CHANGE_LOSS_WEIGHTS = ChangeLossWeights()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class PriorConditionedChangeHead:
    """Factory wrapper returning a torch module without a hard core dependency."""

    @staticmethod
    def build(
        source_decoder: Any,
        *,
        parameterization: str = "independent",
        operation_head: str = "global_stats",
    ) -> Any:
        import torch
        from torch import nn

        if parameterization not in {"independent", "current_difference"}:
            raise ValueError(f"unsupported change parameterization: {parameterization}")
        if operation_head not in {"global_stats", "spatial_pyramid"}:
            raise ValueError(f"unsupported operation head: {operation_head}")

        class _ChangeHead(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.road_decoder = copy.deepcopy(source_decoder)
                embedding_channels = next(
                    module.in_channels
                    for module in source_decoder.modules()
                    if isinstance(module, nn.Conv2d | nn.ConvTranspose2d)
                )
                self.embedding_projection = nn.Sequential(
                    nn.Conv2d(embedding_channels, 32, kernel_size=1),
                    nn.GroupNorm(8, 32),
                    nn.GELU(),
                )
                self.feature_extractor = nn.Sequential(
                    nn.Conv2d(36, 32, kernel_size=5, padding=2),
                    nn.GroupNorm(8, 32),
                    nn.GELU(),
                    nn.Conv2d(32, 32, kernel_size=3, padding=1),
                    nn.GroupNorm(8, 32),
                    nn.GELU(),
                )
                self.change_projection = nn.Conv2d(32, 3, kernel_size=1)
                operation_input = 78 if operation_head == "global_stats" else 663
                operation_hidden = 32 if operation_head == "global_stats" else 128
                self.operation_classifier = nn.Sequential(
                    nn.Linear(operation_input, operation_hidden),
                    nn.GELU(),
                    nn.Dropout(p=0.1 if operation_head == "global_stats" else 0.2),
                    nn.Linear(operation_hidden, 4),
                )
                final = self.change_projection
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)
                if parameterization == "independent":
                    with torch.no_grad():
                        final.bias[1:] = -4.0

            def forward(self, image_embeddings: Any, prior: Any) -> Any:
                road_logits = self.road_decoder(image_embeddings)[:, 1:2]
                road_probability = torch.sigmoid(road_logits)
                signed_change = road_probability - prior
                scalar_features = torch.cat(
                    [road_probability, prior, signed_change, signed_change.abs()], dim=1
                )
                image_features = torch.nn.functional.interpolate(
                    self.embedding_projection(image_embeddings),
                    size=prior.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                features = torch.cat([scalar_features, image_features], dim=1)
                hidden = self.feature_extractor(features)
                residual = self.change_projection(hidden)
                current = road_logits + residual[:, 0:1]
                if parameterization == "current_difference":
                    add = current - 8.0 * prior + residual[:, 1:2]
                    remove = -current - 8.0 * (1.0 - prior) + residual[:, 2:3]
                else:
                    add = residual[:, 1:2]
                    remove = residual[:, 2:3]
                change_logits = torch.cat([current, add, remove], dim=1)
                change_probability = torch.sigmoid(change_logits)
                if operation_head == "spatial_pyramid":
                    operation_features = torch.cat(
                        [
                            torch.nn.functional.adaptive_avg_pool2d(hidden, 4).flatten(1),
                            torch.nn.functional.adaptive_avg_pool2d(
                                scalar_features, 4
                            ).flatten(1),
                            torch.nn.functional.adaptive_avg_pool2d(
                                change_probability, 4
                            ).flatten(1),
                            hidden.amax(dim=(2, 3)),
                            scalar_features.amax(dim=(2, 3)),
                            change_probability.amax(dim=(2, 3)),
                        ],
                        dim=1,
                    )
                else:
                    operation_features = torch.cat(
                        [
                            hidden.mean(dim=(2, 3)),
                            hidden.amax(dim=(2, 3)),
                            scalar_features.mean(dim=(2, 3)),
                            scalar_features.amax(dim=(2, 3)),
                            change_probability.mean(dim=(2, 3)),
                            change_probability.amax(dim=(2, 3)),
                        ],
                        dim=1,
                    )
                return {
                    "change_logits": change_logits,
                    "operation_logits": self.operation_classifier(operation_features),
                }

        return _ChangeHead()


class PriorConditionedSAMRoadPredictor:
    """Deploy the trained map-relative SAM-Road change head."""

    def __init__(
        self,
        repo_root: Path,
        config_path: Path,
        source_checkpoint: Path,
        sam_checkpoint: Path,
        change_checkpoint: Path,
        *,
        device: str = "cuda",
        parameterization: str = "independent",
        operation_head: str = "global_stats",
        upstream_commit: str | None = None,
    ) -> None:
        if not change_checkpoint.is_file():
            raise FileNotFoundError(change_checkpoint)
        base = SAMRoadPredictor(
            repo_root,
            config_path,
            source_checkpoint,
            sam_checkpoint,
            device=device,
            upstream_commit=upstream_commit,
        )
        torch = base._torch
        payload = torch.load(change_checkpoint, map_location="cpu")
        if payload.get("schema_version") not in {
            "muno21-prior-conditioned-head-v1",
            "muno21-prior-conditioned-head-v2",
        }:
            raise ValueError("unsupported prior-conditioned change checkpoint")
        expected_source = payload.get("source_checkpoint_sha256")
        observed_source = _sha256(source_checkpoint)
        if expected_source != observed_source:
            raise ValueError(
                "change checkpoint source hash does not match the SAM-Road checkpoint"
            )
        state_dict = payload.get("head_state_dict")
        if not isinstance(state_dict, dict) or not state_dict:
            raise ValueError("change checkpoint has no head_state_dict")
        operation_point = payload.get("operation_point")
        if not isinstance(operation_point, dict):
            raise ValueError("change checkpoint has no calibrated operation point")
        commit_threshold = float(operation_point.get("commit_threshold", -1.0))
        if not 0.0 <= commit_threshold <= 1.0:
            raise ValueError("change checkpoint has an invalid commit threshold")
        head_config = payload.get("head_config")
        if isinstance(head_config, dict) and (
            head_config.get("parameterization") != parameterization
            or head_config.get("operation_head") != operation_head
        ):
            raise ValueError("requested head configuration differs from checkpoint metadata")

        head = PriorConditionedChangeHead.build(
            base.model.map_decoder,
            parameterization=parameterization,
            operation_head=operation_head,
        )
        head.load_state_dict(state_dict, strict=True)
        self._torch = torch
        self.base = base
        self.head = head.eval().to(base.device)
        self.device = base.device
        self.patch_size = base.patch_size
        self.commit_threshold = commit_threshold
        self.upstream_commit = base.upstream_commit
        self.source_checkpoint_name = source_checkpoint.name
        self.change_checkpoint_name = change_checkpoint.name

    @staticmethod
    def operation_decision(
        probabilities: np.ndarray, commit_threshold: float
    ) -> tuple[EditOperation, EditOperation]:
        values = np.asarray(probabilities, dtype=np.float32)
        if values.shape != (len(EditOperation),) or not np.all(np.isfinite(values)):
            raise ValueError("operation probabilities must contain four finite values")
        if not 0.0 <= commit_threshold <= 1.0:
            raise ValueError("commit threshold must be in [0, 1]")
        raw_index = int(np.argmax(values))
        nonkeep_index = int(np.argmax(values[1:])) + 1
        commit = (
            values[nonkeep_index] >= commit_threshold
            and values[nonkeep_index] > values[0]
        )
        gated_index = nonkeep_index if commit else 0
        edits = list(EditOperation)
        return edits[raw_index], edits[gated_index]

    def predict(self, image: np.ndarray, prior_mask: np.ndarray) -> dict[str, Any]:
        torch = self._torch
        rgb = self.base._rgb_255(image)
        prior = np.asarray(prior_mask, dtype=np.float32).squeeze()
        if prior.ndim != 2 or prior.shape != rgb.shape[1:]:
            raise ValueError("image and prior mask spatial shapes must match")
        if not np.all(np.isfinite(prior)):
            raise ValueError("prior mask contains non-finite pixels")
        original_shape = prior.shape
        image_tensor = torch.from_numpy(rgb).unsqueeze(0).to(self.device)
        prior_tensor = torch.from_numpy(np.clip(prior, 0.0, 1.0))[None, None].to(
            self.device
        )
        if original_shape != (self.patch_size, self.patch_size):
            image_tensor = torch.nn.functional.interpolate(
                image_tensor,
                size=(self.patch_size, self.patch_size),
                mode="bilinear",
                align_corners=False,
            )
            prior_tensor = torch.nn.functional.interpolate(
                prior_tensor,
                size=(self.patch_size, self.patch_size),
                mode="nearest",
            )
        with torch.inference_mode():
            normalized = (
                image_tensor - self.base.model.pixel_mean
            ) / self.base.model.pixel_std
            embeddings = self.base.model.image_encoder(normalized)
            output = self.head(embeddings, prior_tensor)
            change_probability = torch.sigmoid(output["change_logits"])
            if original_shape != (self.patch_size, self.patch_size):
                change_probability = torch.nn.functional.interpolate(
                    change_probability,
                    size=original_shape,
                    mode="bilinear",
                    align_corners=False,
                )
            operation_probability = torch.softmax(
                output["operation_logits"], dim=1
            )[0]
        change = change_probability[0].float().cpu().numpy()
        operations = operation_probability.float().cpu().numpy()
        raw_edit, gated_edit = self.operation_decision(
            operations, self.commit_threshold
        )
        selected_index = list(EditOperation).index(gated_edit)
        entropy = float(
            -np.sum(operations * np.log(np.clip(operations, 1e-8, None)))
            / np.log(len(operations))
        )
        return {
            "mask_probability": change[0],
            "add_probability": change[1],
            "remove_probability": change[2],
            "edit_probabilities": operations,
            "predicted_edit": raw_edit.value,
            "gated_edit": gated_edit.value,
            "update_probability": float(np.max(operations[1:])),
            "confidence": float(operations[selected_index]),
            "uncertainty": entropy,
            "commit_threshold": self.commit_threshold,
            "backend_version": self.upstream_commit,
            "model_checkpoint": self.change_checkpoint_name,
            "source_model_checkpoint": self.source_checkpoint_name,
        }


def change_targets(target: Any, prior: Any) -> tuple[Any, Any, Any]:
    target_binary = target >= 0.5
    prior_binary = prior >= 0.5
    add = (target_binary & ~prior_binary).to(target.dtype)
    remove = (prior_binary & ~target_binary).to(target.dtype)
    return target.float(), add, remove


def _masked_bce(logits: Any, target: Any, valid: Any, positive_weight: float) -> Any:
    import torch
    from torch.nn import functional as functional

    loss = functional.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=torch.as_tensor(positive_weight, device=logits.device),
        reduction="none",
    )
    return (loss * valid).sum() / valid.sum().clamp_min(1.0)


def _masked_dice(
    logits: Any,
    target: Any,
    valid: Any,
    *,
    positive_only: bool = False,
) -> Any:
    import torch

    probability = torch.sigmoid(logits) * valid
    target = target * valid
    dimensions = tuple(range(1, target.ndim))
    intersection = (probability * target).sum(dim=dimensions)
    denominator = probability.sum(dim=dimensions) + target.sum(dim=dimensions)
    score = (2.0 * intersection + 1.0) / (denominator + 1.0)
    if positive_only:
        positive = target.sum(dim=dimensions) > 0
        if not bool(positive.any()):
            return logits.sum() * 0.0
        score = score[positive]
    return 1.0 - score.mean()


def prior_conditioned_change_loss(
    logits: Any,
    operation_logits: Any,
    operation_target: Any,
    target: Any,
    prior: Any,
    valid: Any,
    *,
    positive_weights: tuple[float, float, float],
    operation_weights: Any,
    positive_only_change_dice: bool = False,
    weights: ChangeLossWeights = DEFAULT_CHANGE_LOSS_WEIGHTS,
) -> tuple[Any, dict[str, Any]]:
    import torch
    from torch.nn import functional as functional

    if logits.ndim != 4 or logits.shape[1] != 3:
        raise ValueError("change logits must have shape [B, 3, H, W]")
    if target.shape != prior.shape or target.shape != valid.shape:
        raise ValueError("target, prior and valid shapes must match")
    targets = change_targets(target, prior)
    channel_losses = []
    components = {}
    for index, name in enumerate(("current", "add", "remove")):
        bce = _masked_bce(
            logits[:, index], targets[index], valid, positive_weights[index]
        )
        dice = _masked_dice(
            logits[:, index],
            targets[index],
            valid,
            positive_only=positive_only_change_dice and index > 0,
        )
        channel = 0.5 * (bce + dice)
        components[f"{name}_bce"] = bce
        components[f"{name}_dice"] = dice
        components[name] = channel
        channel_losses.append(channel)

    probability = torch.sigmoid(logits)
    reconstructed = prior * (1.0 - probability[:, 2]) + (1.0 - prior) * probability[:, 1]
    reconstruction = functional.binary_cross_entropy(
        reconstructed.clamp(1e-6, 1.0 - 1e-6),
        target.float(),
        weight=valid,
        reduction="sum",
    ) / valid.sum().clamp_min(1.0)
    unchanged_pixels = (target >= 0.5).eq(prior >= 0.5).to(probability.dtype) * valid
    unchanged_safety = (
        ((probability[:, 1] + probability[:, 2]) * unchanged_pixels).sum()
        / unchanged_pixels.sum().clamp_min(1.0)
    )
    exclusivity = (
        probability[:, 1].mul(probability[:, 2]).mul(valid).sum()
        / valid.sum().clamp_min(1.0)
    )
    components.update(
        {
            "reconstruction": reconstruction,
            "unchanged_safety": unchanged_safety,
            "exclusivity": exclusivity,
            "operation": functional.cross_entropy(
                operation_logits,
                operation_target,
                weight=operation_weights,
            ),
        }
    )
    total = (
        weights.current * channel_losses[0]
        + weights.add * channel_losses[1]
        + weights.remove * channel_losses[2]
        + weights.reconstruction * reconstruction
        + weights.unchanged_safety * unchanged_safety
        + weights.exclusivity * exclusivity
        + weights.operation * components["operation"]
    )
    components["total"] = total
    return total, components
