"""Outcome-conditioned evidence valuation for structured map-update actions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from activemap.features import EVIDENCE_DIM, HYPOTHESIS_DIM, STATE_DIM
from activemap.models import EditOperation
from activemap.selector_records import SelectorSample

EDIT_PROBABILITY_DIM = 4
GEOMETRY_DELTA_DIM = 8
CONTEXT_DIM = HYPOTHESIS_DIM + STATE_DIM
CANDIDATE_DIM = (
    EVIDENCE_DIM + 2 + EDIT_PROBABILITY_DIM + 1 + GEOMETRY_DELTA_DIM
)


@dataclass(frozen=True)
class EvidenceValueNormalizer:
    context_mean: np.ndarray
    context_std: np.ndarray
    candidate_mean: np.ndarray
    candidate_std: np.ndarray

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "context_mean": self.context_mean.tolist(),
            "context_std": self.context_std.tolist(),
            "candidate_mean": self.candidate_mean.tolist(),
            "candidate_std": self.candidate_std.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, list[float]]) -> EvidenceValueNormalizer:
        return cls(
            **{
                key: np.asarray(value, dtype=np.float32)
                for key, value in payload.items()
            }
        )

    def transform(
        self, context: np.ndarray, candidates: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            ((context - self.context_mean) / self.context_std).astype(np.float32),
            ((candidates - self.candidate_mean) / self.candidate_std).astype(
                np.float32
            ),
        )


def _mean_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = values.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def fit_evidence_value_normalizer(
    examples: list[dict[str, Any]],
) -> EvidenceValueNormalizer:
    if not examples:
        raise ValueError("normalizer requires at least one example")
    contexts = np.stack([row["context"] for row in examples])
    candidates = np.concatenate([row["candidates"] for row in examples], axis=0)
    context_mean, context_std = _mean_std(contexts)
    candidate_mean, candidate_std = _mean_std(candidates)
    return EvidenceValueNormalizer(
        context_mean=context_mean,
        context_std=context_std,
        candidate_mean=candidate_mean,
        candidate_std=candidate_std,
    )


def evidence_value_features(
    sample: SelectorSample,
) -> tuple[np.ndarray, np.ndarray]:
    """Build deployable context and candidate features without target outcomes."""

    predictions = sample.metadata.get("evidence_predictions")
    if not isinstance(predictions, dict):
        raise ValueError("sample lacks evidence predictions")

    context = np.asarray(
        sample.hypothesis_features + sample.state_features, dtype=np.float32
    )
    candidate_rows = []
    for index, evidence_id in enumerate(sample.evidence_ids):
        if evidence_id not in predictions:
            raise ValueError(f"missing evidence prediction for {evidence_id}")
        prediction = predictions[evidence_id]
        probabilities = list(prediction["edit_probabilities"])
        geometry_delta = list(prediction["geometry_delta"])
        if len(probabilities) != EDIT_PROBABILITY_DIM:
            raise ValueError("edit probability vector has the wrong dimension")
        if len(geometry_delta) != GEOMETRY_DELTA_DIM:
            raise ValueError("geometry delta has the wrong dimension")
        candidate_rows.append(
            list(sample.evidence_features[index])
            + [sample.evidence_costs[index], sample.false_edit_risks[index]]
            + probabilities
            + [float(prediction["confidence"])]
            + geometry_delta
        )

    candidates = np.asarray(candidate_rows, dtype=np.float32).reshape(
        len(candidate_rows),
        CANDIDATE_DIM,
    )
    if context.shape != (CONTEXT_DIM,) or candidates.shape != (
        len(sample.evidence_ids),
        CANDIDATE_DIM,
    ):
        raise ValueError("constructed Evidence Value features have the wrong dimension")
    if not np.isfinite(context).all() or not np.isfinite(candidates).all():
        raise ValueError("Evidence Value features contain non-finite values")
    return context, candidates


def executable_value_example(sample: SelectorSample) -> dict[str, Any]:
    """Convert one frozen counterfactual state into multi-head supervision."""

    if sample.split == "test":
        raise ValueError("test samples cannot be used to construct training examples")
    if sample.metadata.get("utility_mode") != "executable":
        raise ValueError("Evidence Value Head requires executable utility labels")
    outcomes = sample.metadata.get("executable_outcomes")
    if not isinstance(outcomes, dict):
        raise ValueError("sample lacks executable outcomes")
    context, candidates = evidence_value_features(sample)
    quality_gains = []
    unsafe = []
    missed = []
    for evidence_id in sample.evidence_ids:
        if evidence_id not in outcomes:
            raise ValueError(f"missing executable outcome for {evidence_id}")
        outcome = outcomes[evidence_id]
        quality_gains.append(float(outcome["quality_gain"]))
        unsafe.append(
            float(bool(outcome.get("false_edit")) or bool(outcome.get("wrong_edit")))
        )
        missed.append(float(bool(outcome.get("missed_edit"))))

    utility_gains = (
        np.asarray(sample.oracle_utilities, dtype=np.float32)
        - np.float32(sample.stop_utility)
    )
    target_edit = EditOperation(str(sample.metadata["gt_edit"]))
    arrays = [context, candidates, utility_gains, np.asarray(quality_gains)]
    if not all(np.isfinite(values).all() for values in arrays):
        raise ValueError("Evidence Value example contains non-finite values")
    return {
        "sample_id": sample.sample_id,
        "split": sample.split,
        "edit_type": sample.edit_type.value,
        "terminal_target": list(EditOperation).index(target_edit),
        "context": context,
        "candidates": candidates,
        "utility_gains": utility_gains,
        "quality_gains": np.asarray(quality_gains, dtype=np.float32),
        "beneficial": (utility_gains > 0.0).astype(np.float32),
        "unsafe": np.asarray(unsafe, dtype=np.float32),
        "missed": np.asarray(missed, dtype=np.float32),
        "stop_utility": float(sample.stop_utility),
        "utilities": np.asarray(sample.oracle_utilities, dtype=np.float32),
    }


@dataclass(frozen=True)
class EvidenceValueHeadConfig:
    context_dim: int = CONTEXT_DIM
    candidate_dim: int = CANDIDATE_DIM
    hidden_dim: int = 128
    dropout: float = 0.10
    terminal_classes: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvidenceValueHead(nn.Module):
    """Estimate map outcome, safety, and value for each evidence action."""

    def __init__(self, config: EvidenceValueHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.context_encoder = self._encoder(config.context_dim)
        self.candidate_encoder = self._encoder(config.candidate_dim)
        self.interaction = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.utility_head = nn.Linear(config.hidden_dim, 1)
        self.quality_head = nn.Linear(config.hidden_dim, 1)
        self.beneficial_head = nn.Linear(config.hidden_dim, 1)
        self.unsafe_head = nn.Linear(config.hidden_dim, 1)
        self.missed_head = nn.Linear(config.hidden_dim, 1)
        self.terminal_head = (
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.terminal_classes),
            )
            if config.terminal_classes > 0
            else None
        )

    def _encoder(self, input_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
        )

    def forward(self, context: Tensor, candidates: Tensor) -> dict[str, Tensor]:
        if context.shape[-1] != self.config.context_dim:
            raise ValueError("Evidence Value context dimension mismatch")
        if candidates.shape[-1] != self.config.candidate_dim:
            raise ValueError("Evidence Value candidate dimension mismatch")
        encoded_context = self.context_encoder(context)
        terminal_context = encoded_context
        encoded_candidates = self.candidate_encoder(candidates)
        while encoded_context.ndim < encoded_candidates.ndim:
            encoded_context = encoded_context.unsqueeze(-2)
        encoded_context = encoded_context.expand_as(encoded_candidates)
        latent = self.interaction(
            torch.cat(
                [
                    encoded_context,
                    encoded_candidates,
                    encoded_context * encoded_candidates,
                ],
                dim=-1,
            )
        )
        result = {
            "utility": self.utility_head(latent).squeeze(-1),
            "quality": self.quality_head(latent).squeeze(-1),
            "beneficial_logit": self.beneficial_head(latent).squeeze(-1),
            "unsafe_logit": self.unsafe_head(latent).squeeze(-1),
            "missed_logit": self.missed_head(latent).squeeze(-1),
        }
        if self.terminal_head is not None:
            result["terminal_edit_logits"] = self.terminal_head(terminal_context)
        return result

    def terminal_logits(self, context: Tensor) -> Tensor:
        if self.terminal_head is None:
            raise ValueError("terminal head is not configured")
        if context.shape[-1] != self.config.context_dim:
            raise ValueError("Evidence Value context dimension mismatch")
        return self.terminal_head(self.context_encoder(context))


def risk_adjusted_scores(
    outputs: dict[str, Tensor],
    *,
    unsafe_weight: float,
    missed_weight: float,
) -> Tensor:
    return (
        outputs["utility"]
        - unsafe_weight * torch.sigmoid(outputs["unsafe_logit"])
        - missed_weight * torch.sigmoid(outputs["missed_logit"])
    )


class EvidenceValuePredictor:
    """Callable rollout adapter with a calibrated explicit STOP score."""

    def __init__(self, checkpoint_path: str, device: str = "cpu") -> None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("protocol") != "evidence_value_head_v1":
            raise ValueError("unsupported Evidence Value checkpoint protocol")
        self.model = EvidenceValueHead(
            EvidenceValueHeadConfig(**checkpoint["model_config"])
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.normalizer = EvidenceValueNormalizer.from_dict(checkpoint["normalizer"])
        self.safety_margin = float(checkpoint["safety_margin"])
        self.unsafe_penalty = float(checkpoint.get("unsafe_penalty", 0.0))
        self.missed_penalty = float(checkpoint.get("missed_penalty", 0.0))

    @torch.no_grad()
    def score_sample(self, sample: SelectorSample) -> np.ndarray:
        context, candidates = evidence_value_features(sample)
        context, candidates = self.normalizer.transform(context, candidates)
        outputs = self.model(
            torch.from_numpy(context).to(self.device),
            torch.from_numpy(candidates).to(self.device),
        )
        return (
            risk_adjusted_scores(
                outputs,
                unsafe_weight=self.unsafe_penalty,
                missed_weight=self.missed_penalty,
            )
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    def __call__(self, sample: SelectorSample) -> np.ndarray:
        scores = self.score_sample(sample)
        return np.concatenate(
            [scores, np.asarray([self.safety_margin], dtype=np.float32)]
        )


class StructuredMapActionPredictor:
    """Joint evidence and terminal-edit policy for recurrent map maintenance."""

    def __init__(self, checkpoint_path: str, device: str = "cpu") -> None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("protocol") != "structured_map_action_policy_v1":
            raise ValueError("unsupported structured map action checkpoint")
        self.model = EvidenceValueHead(
            EvidenceValueHeadConfig(**checkpoint["model_config"])
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.normalizer = EvidenceValueNormalizer.from_dict(checkpoint["normalizer"])
        self.safety_margin = float(checkpoint["safety_margin"])
        self.unsafe_penalty = float(checkpoint.get("unsafe_penalty", 0.0))
        self.missed_penalty = float(checkpoint.get("missed_penalty", 0.0))
        self.last_terminal_edit: EditOperation | None = None

    @torch.no_grad()
    def predict(self, sample: SelectorSample) -> tuple[np.ndarray, EditOperation]:
        context, candidates = evidence_value_features(sample)
        context, candidates = self.normalizer.transform(context, candidates)
        outputs = self.model(
            torch.from_numpy(context).to(self.device),
            torch.from_numpy(candidates).to(self.device),
        )
        scores = (
            risk_adjusted_scores(
                outputs,
                unsafe_weight=self.unsafe_penalty,
                missed_weight=self.missed_penalty,
            )
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        terminal_index = int(outputs["terminal_edit_logits"].argmax().item())
        terminal_edit = list(EditOperation)[terminal_index]
        self.last_terminal_edit = terminal_edit
        return scores, terminal_edit

    def __call__(self, sample: SelectorSample) -> np.ndarray:
        scores, _ = self.predict(sample)
        return np.concatenate(
            [scores, np.asarray([self.safety_margin], dtype=np.float32)]
        )
