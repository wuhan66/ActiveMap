"""Load ActiveMap checkpoints and expose NumPy inference functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from activemap.features import AblationSpec, apply_ablation
from activemap.models import EditOperation
from activemap.nn.operation_selector import (
    OperationSelector,
    OperationSelectorConfig,
    operation_selector_inputs,
)
from activemap.nn.selector import EvidenceSelector, SelectorConfig
from activemap.nn.updater import (
    PriorConditionedUNet,
    UpdaterConfig,
    operation_probabilities,
)
from activemap.selector_records import SelectorSample
from activemap.training.data import SelectorFeatureNormalizer


class SelectorPredictor:
    def __init__(
        self,
        checkpoint_path: Path,
        device: str = "cpu",
        stop_margin_override: float | None = None,
    ) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.model = EvidenceSelector(SelectorConfig(**checkpoint["model_config"]))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(device).eval()
        self.checkpoint_stop_margin = float(checkpoint.get("stop_margin", 0.0))
        self.stop_margin = (
            float(stop_margin_override)
            if stop_margin_override is not None
            else self.checkpoint_stop_margin
        )
        self.stop_margin_source = (
            "override" if stop_margin_override is not None else "checkpoint"
        )
        data_contract = checkpoint.get("data_contract")
        if data_contract is not None and not isinstance(data_contract, dict):
            raise ValueError("selector checkpoint data_contract must be a mapping")
        self.data_contract = dict(data_contract) if data_contract is not None else None
        normalizer_payload = checkpoint.get("feature_normalizer")
        self.feature_normalizer = (
            SelectorFeatureNormalizer.from_dict(normalizer_payload)
            if normalizer_payload is not None
            else None
        )
        ablation_payload = checkpoint.get("ablation", {})
        self.ablation = AblationSpec(
            name=str(ablation_payload.get("name", "full")),
            condition_on_hypothesis=bool(ablation_payload.get("condition_on_hypothesis", True)),
            drop_hypothesis_groups=tuple(ablation_payload.get("drop_hypothesis_groups", [])),
            drop_evidence_groups=tuple(ablation_payload.get("drop_evidence_groups", [])),
            drop_state_groups=tuple(ablation_payload.get("drop_state_groups", [])),
            false_edit_penalty=bool(ablation_payload.get("false_edit_penalty", True)),
            allow_stop=bool(ablation_payload.get("allow_stop", True)),
        )
        self.device = torch.device(device)

    @torch.no_grad()
    def action_scores(self, sample: SelectorSample) -> np.ndarray:
        hypothesis = np.asarray(sample.hypothesis_features, dtype=np.float32)
        evidence = np.asarray(sample.evidence_features, dtype=np.float32)
        state = np.asarray(sample.state_features, dtype=np.float32)
        if self.feature_normalizer is not None:
            hypothesis, evidence, state = self.feature_normalizer.transform(
                hypothesis, evidence, state
            )
        hypothesis, evidence, state = apply_ablation(
            hypothesis, evidence, state, self.ablation
        )
        logits = self.model(
            torch.from_numpy(evidence).unsqueeze(0).to(self.device),
            torch.from_numpy(hypothesis).unsqueeze(0).to(self.device),
            torch.from_numpy(state).unsqueeze(0).to(self.device),
        )[0]
        if self.model.config.allow_stop and self.stop_margin != 0.0:
            logits = logits.clone()
            logits[-1] += self.stop_margin
        return logits.cpu().numpy()

    def scores(self, sample: SelectorSample) -> np.ndarray:
        return self.action_scores(sample)[: len(sample.evidence_ids)]


class SelectorEnsemblePredictor:
    """Average calibrated action scores from independently trained selectors."""

    def __init__(self, checkpoint_paths: list[Path], device: str = "cpu") -> None:
        if not checkpoint_paths:
            raise ValueError("at least one selector checkpoint is required")
        self.members = [SelectorPredictor(path, device=device) for path in checkpoint_paths]

    def action_scores(self, sample: SelectorSample) -> np.ndarray:
        member_scores = [member.action_scores(sample) for member in self.members]
        return np.mean(np.stack(member_scores, axis=0), axis=0).astype(np.float32)

    def scores(self, sample: SelectorSample) -> np.ndarray:
        return self.action_scores(sample)[: len(sample.evidence_ids)]


def _image_channels_first(
    array: np.ndarray,
    *,
    expected_channels: int | None = None,
) -> np.ndarray:
    if array.ndim != 3:
        raise ValueError("image must have three dimensions")
    valid_channels = {1, 3, 4} if expected_channels is None else {expected_channels}
    if array.shape[0] in valid_channels:
        output = array
    elif array.shape[-1] in valid_channels:
        output = np.moveaxis(array, -1, 0)
    else:
        expected = sorted(valid_channels)
        raise ValueError(
            f"cannot infer image channel axis from {array.shape}; expected channels {expected}"
        )
    output = output.astype(np.float32)
    if output.max(initial=0.0) > 1.0:
        output /= 255.0
    return output


def _single_mask(array: np.ndarray) -> np.ndarray:
    output = np.asarray(array, dtype=np.float32)
    if output.ndim == 2:
        output = output[None, ...]
    if output.ndim != 3 or output.shape[0] != 1:
        raise ValueError("prior mask must have shape [H,W] or [1,H,W]")
    return np.clip(output, 0.0, 1.0)


class UpdaterPredictor:
    """Inference wrapper for typed edit, confidence, geometry, and raster mask."""

    def __init__(self, checkpoint_path: Path, device: str = "cpu") -> None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.model = PriorConditionedUNet(UpdaterConfig(**checkpoint["model_config"]))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(device).eval()
        self.device = torch.device(device)

    @torch.no_grad()
    def predict(self, image: np.ndarray, prior_mask: np.ndarray) -> dict[str, np.ndarray | float]:
        image_array = _image_channels_first(
            np.asarray(image), expected_channels=self.model.config.image_channels
        )
        prior_array = _single_mask(np.asarray(prior_mask))
        if image_array.shape[-2:] != prior_array.shape[-2:]:
            raise ValueError("image and prior mask spatial shapes must match")
        prior_tensor = torch.from_numpy(prior_array)[None].to(self.device)
        outputs = self.model(
            torch.from_numpy(image_array)[None].to(self.device),
            prior_tensor,
        )
        edit_probabilities = operation_probabilities(outputs, prior_tensor)[0].cpu().numpy()
        result: dict[str, np.ndarray | float] = {
            "mask_probability": torch.sigmoid(outputs["segmentation_logits"])[0, 0].cpu().numpy(),
            "edit_probabilities": edit_probabilities,
            "geometry_delta": outputs["geometry_delta"][0].cpu().numpy(),
            "confidence": float(torch.sigmoid(outputs["confidence_logits"])[0].cpu()),
        }
        if "temporal_change_logits" in outputs:
            temporal = torch.sigmoid(outputs["temporal_change_logits"])[0].cpu().numpy()
            result["add_probability"] = temporal[0]
            result["remove_probability"] = temporal[1]
        if "presence_logits" in outputs:
            result["presence_probability"] = float(
                torch.sigmoid(outputs["presence_logits"])[0].cpu()
            )
            result["change_probability"] = float(torch.sigmoid(outputs["change_logits"])[0].cpu())
        return result


class OperationSelectorPredictor:
    """Combined frozen updater and typed-operation selector inference."""

    def __init__(
        self,
        selector_checkpoint_path: Path,
        *,
        updater_checkpoint_path: Path | None = None,
        update_threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        if not 0.0 <= update_threshold <= 1.0:
            raise ValueError("update_threshold must be between zero and one")
        selector_checkpoint = torch.load(
            selector_checkpoint_path, map_location=device, weights_only=False
        )
        self.selector = OperationSelector(
            OperationSelectorConfig(**selector_checkpoint["model_config"])
        )
        self.selector.load_state_dict(selector_checkpoint["state_dict"])
        referenced_updater = updater_checkpoint_path or Path(
            selector_checkpoint["updater_checkpoint"]
        )
        updater_checkpoint = torch.load(referenced_updater, map_location=device, weights_only=False)
        self.updater = PriorConditionedUNet(UpdaterConfig(**updater_checkpoint["model_config"]))
        self.updater.load_state_dict(updater_checkpoint["state_dict"])
        self.device = torch.device(device)
        self.updater.to(self.device).eval()
        self.selector.to(self.device).eval()
        self.update_threshold = update_threshold
        self.spatial_size = self.selector.config.spatial_size
        self.selector_checkpoint_path = selector_checkpoint_path
        self.updater_checkpoint_path = referenced_updater

    @torch.no_grad()
    def predict(self, image: np.ndarray, prior_mask: np.ndarray) -> dict[str, Any]:
        image_array = _image_channels_first(
            np.asarray(image), expected_channels=self.updater.config.image_channels
        )
        prior_array = _single_mask(np.asarray(prior_mask))
        if image_array.shape[-2:] != prior_array.shape[-2:]:
            raise ValueError("image and prior mask spatial shapes must match")
        image_tensor = torch.from_numpy(image_array)[None].to(self.device)
        prior_tensor = torch.from_numpy(prior_array)[None].to(self.device)
        outputs = self.updater(image_tensor, prior_tensor)
        spatial, context = operation_selector_inputs(
            outputs, prior_tensor, spatial_size=self.spatial_size
        )
        selector_probabilities = torch.softmax(self.selector(spatial, context), dim=1)[0]
        updater_probabilities = operation_probabilities(outputs, prior_tensor)[0]
        update_probability = float(1.0 - selector_probabilities[0].cpu())
        raw_index = int(torch.argmax(selector_probabilities).cpu())
        gated_index = (
            int(torch.argmax(selector_probabilities[1:]).cpu()) + 1
            if update_probability >= self.update_threshold
            else 0
        )
        probabilities_array = selector_probabilities.cpu().numpy()
        entropy = float(
            -np.sum(probabilities_array * np.log(np.clip(probabilities_array, 1e-8, None)))
            / np.log(len(probabilities_array))
        )
        return {
            "mask_probability": torch.sigmoid(outputs["segmentation_logits"])[0, 0].cpu().numpy(),
            "edit_probabilities": probabilities_array,
            "updater_edit_probabilities": updater_probabilities.cpu().numpy(),
            "predicted_edit": list(EditOperation)[raw_index].value,
            "gated_edit": list(EditOperation)[gated_index].value,
            "update_probability": update_probability,
            "update_threshold": self.update_threshold,
            "confidence": float(selector_probabilities[gated_index].cpu()),
            "uncertainty": entropy,
            "geometry_delta": outputs["geometry_delta"][0].cpu().numpy(),
        }
