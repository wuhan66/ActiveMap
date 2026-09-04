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
MASK_FEATURE_DIM = 10
CONTEXT_DIM = HYPOTHESIS_DIM + STATE_DIM
CANDIDATE_DIM = (
    EVIDENCE_DIM + 2 + EDIT_PROBABILITY_DIM + 1 + GEOMETRY_DELTA_DIM
)
POLICY_RELATIVE_DIM = 18
POLICY_RELATIVE_MASK_CANDIDATE_DIM = (
    CANDIDATE_DIM + POLICY_RELATIVE_DIM + MASK_FEATURE_DIM
)
LEGACY_FEATURE_SET = "evidence-value"
MASK_FEATURE_SET = "policy-relative-mask"
MASK_FEATURES_SCHEMA = "target-free-mask-features-v2"
SUPPORTED_FEATURE_SETS = (LEGACY_FEATURE_SET, MASK_FEATURE_SET)


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


def normalized_edit_entropy(probabilities: np.ndarray) -> float:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-8, None)
    values /= values.sum()
    return float(-np.sum(values * np.log(values)) / np.log(len(values)))


def candidate_feature_dim(feature_set: str) -> int:
    if feature_set == LEGACY_FEATURE_SET:
        return CANDIDATE_DIM
    if feature_set == MASK_FEATURE_SET:
        return POLICY_RELATIVE_MASK_CANDIDATE_DIM
    raise ValueError(f"unsupported Evidence Value feature set: {feature_set}")


def evidence_value_features(
    sample: SelectorSample,
    *,
    feature_set: str = LEGACY_FEATURE_SET,
) -> tuple[np.ndarray, np.ndarray]:
    """Build deployable context and candidate features without target outcomes."""

    predictions = sample.metadata.get("evidence_predictions")
    if not isinstance(predictions, dict):
        raise ValueError("sample lacks evidence predictions")
    expected_candidate_dim = candidate_feature_dim(feature_set)
    if feature_set == MASK_FEATURE_SET:
        contract = sample.metadata.get("mask_feature_contract")
        if (
            not isinstance(contract, dict)
            or contract.get("schema_version") != MASK_FEATURES_SCHEMA
        ):
            raise ValueError("policy-relative-mask requires the v2 mask contract")
        if contract.get("target_free") is not True:
            raise ValueError("mask feature contract must declare target-free provenance")
        if contract.get("scope") != "candidate_local_grid":
            raise ValueError("mask features must use a candidate-local raster grid")

    context = np.asarray(
        sample.hypothesis_features + sample.state_features, dtype=np.float32
    )
    prior_probabilities = np.asarray(sample.hypothesis_features[:4], dtype=np.float32)
    prior_entropy = normalized_edit_entropy(prior_probabilities)
    prior_confidence = float(sample.hypothesis_features[13])
    initial_operation = sample.edit_type.value
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
        features = (
            list(sample.evidence_features[index])
            + [sample.evidence_costs[index], sample.false_edit_risks[index]]
            + probabilities
            + [float(prediction["confidence"])]
            + geometry_delta
        )
        if feature_set == MASK_FEATURE_SET:
            candidate_probabilities = np.asarray(probabilities, dtype=np.float32)
            probability_delta = candidate_probabilities - prior_probabilities
            gated_operation = str(
                prediction.get(
                    "gated_edit",
                    list(EditOperation)[int(np.argmax(candidate_probabilities))].value,
                )
            )
            valid_operations = tuple(operation.value for operation in EditOperation)
            if gated_operation not in valid_operations:
                raise ValueError(f"unsupported candidate operation: {gated_operation}")
            ordered = np.sort(candidate_probabilities)
            candidate_entropy = normalized_edit_entropy(candidate_probabilities)
            mask_features = list(prediction.get("mask_features", []))
            if len(mask_features) != MASK_FEATURE_DIM:
                raise ValueError("candidate prediction has invalid mask feature dimensions")
            features += (
                [float(operation == gated_operation) for operation in valid_operations]
                + probability_delta.tolist()
                + np.abs(probability_delta).tolist()
                + [
                    candidate_entropy,
                    candidate_entropy - prior_entropy,
                    float(ordered[-1] - ordered[-2]),
                    float(prediction["confidence"]) - prior_confidence,
                    float(gated_operation == initial_operation),
                    float(np.abs(probability_delta).sum()),
                ]
                + [float(value) for value in mask_features]
            )
        candidate_rows.append(features)

    candidates = np.asarray(candidate_rows, dtype=np.float32).reshape(
        len(candidate_rows),
        expected_candidate_dim,
    )
    if context.shape != (CONTEXT_DIM,) or candidates.shape != (
        len(sample.evidence_ids),
        expected_candidate_dim,
    ):
        raise ValueError("constructed Evidence Value features have the wrong dimension")
    if not np.isfinite(context).all() or not np.isfinite(candidates).all():
        raise ValueError("Evidence Value features contain non-finite values")
    return context, candidates


def executable_value_example(
    sample: SelectorSample,
    *,
    feature_set: str = LEGACY_FEATURE_SET,
) -> dict[str, Any]:
    """Convert one frozen counterfactual state into multi-head supervision."""

    if sample.split == "test":
        raise ValueError("test samples cannot be used to construct training examples")
    if sample.metadata.get("utility_mode") != "executable":
        raise ValueError("Evidence Value Head requires executable utility labels")
    outcomes = sample.metadata.get("executable_outcomes")
    if not isinstance(outcomes, dict):
        raise ValueError("sample lacks executable outcomes")
    context, candidates = evidence_value_features(sample, feature_set=feature_set)
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
        "source_episode": str(sample.metadata.get("source_episode", sample.sample_id)),
        "aoi_id": str(sample.metadata.get("aoi_id", "unknown")),
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
    set_aware: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvidenceValueHead(nn.Module):
    """Estimate map outcome, safety, and value for each evidence action."""

    def __init__(self, config: EvidenceValueHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.context_encoder = self._encoder(config.context_dim)
        self.candidate_encoder = self._encoder(config.candidate_dim)
        interaction_groups = 6 if config.set_aware else 3
        self.interaction = nn.Sequential(
            nn.Linear(config.hidden_dim * interaction_groups, config.hidden_dim),
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

    def forward(
        self,
        context: Tensor,
        candidates: Tensor,
        mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
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
        interaction_features = [
            encoded_context,
            encoded_candidates,
            encoded_context * encoded_candidates,
        ]
        if self.config.set_aware:
            if mask is None:
                mask = torch.ones(
                    candidates.shape[:-1], dtype=torch.bool, device=candidates.device
                )
            if mask.shape != candidates.shape[:-1]:
                raise ValueError("Evidence Value candidate mask shape mismatch")
            if not torch.all(mask.any(dim=-1)):
                raise ValueError("set-aware Evidence Value inputs require a candidate")
            expanded_mask = mask.unsqueeze(-1)
            set_mean = (encoded_candidates * expanded_mask).sum(dim=-2) / (
                expanded_mask.sum(dim=-2).clamp_min(1)
            )
            set_max = encoded_candidates.masked_fill(~expanded_mask, -torch.inf).amax(
                dim=-2
            )
            while set_mean.ndim < encoded_candidates.ndim:
                set_mean = set_mean.unsqueeze(-2)
                set_max = set_max.unsqueeze(-2)
            set_mean = set_mean.expand_as(encoded_candidates)
            set_max = set_max.expand_as(encoded_candidates)
            interaction_features.extend(
                [set_mean, set_max, encoded_candidates * set_mean]
            )
        latent = self.interaction(torch.cat(interaction_features, dim=-1))
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
    beneficial_weight: float = 0.0,
) -> Tensor:
    scores = outputs["utility"]
    if beneficial_weight != 0.0:
        scores = scores + beneficial_weight * torch.sigmoid(
            outputs["beneficial_logit"]
        )
    return scores - unsafe_weight * torch.sigmoid(
        outputs["unsafe_logit"]
    ) - missed_weight * torch.sigmoid(outputs["missed_logit"])


@dataclass(frozen=True)
class TopKSetEvidenceValueConfig:
    """Two-stage evidence ranking with a set-aware top-k second pass."""

    context_dim: int = CONTEXT_DIM
    candidate_dim: int = CANDIDATE_DIM
    hidden_dim: int = 128
    dropout: float = 0.10
    top_k: int = 3
    proposer_beneficial_weight: float = 0.10
    proposer_unsafe_penalty: float = 0.10
    proposer_missed_penalty: float = 0.05

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class TopKSetEvidenceValueHead(nn.Module):
    """Propose from all evidence and rerank a shortlist in candidate-set context."""

    def __init__(self, config: TopKSetEvidenceValueConfig) -> None:
        super().__init__()
        self.config = config
        self.proposer = EvidenceValueHead(
            EvidenceValueHeadConfig(
                context_dim=config.context_dim,
                candidate_dim=config.candidate_dim,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout,
            )
        )
        self.reranker = EvidenceValueHead(
            EvidenceValueHeadConfig(
                context_dim=config.context_dim,
                candidate_dim=config.candidate_dim + 2,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout,
                set_aware=True,
            )
        )

    def forward(
        self,
        context: Tensor,
        candidates: Tensor,
        mask: Tensor,
    ) -> dict[str, Any]:
        if candidates.ndim != 3 or context.ndim != 2 or mask.ndim != 2:
            raise ValueError("top-k Evidence Value training requires batched inputs")
        if mask.shape != candidates.shape[:-1]:
            raise ValueError("top-k Evidence Value candidate mask shape mismatch")
        if not torch.all(mask.any(dim=-1)):
            raise ValueError("top-k Evidence Value inputs require a candidate")

        proposer_outputs = self.proposer(context, candidates, mask)
        proposer_scores = risk_adjusted_scores(
            proposer_outputs,
            beneficial_weight=self.config.proposer_beneficial_weight,
            unsafe_weight=self.config.proposer_unsafe_penalty,
            missed_weight=self.config.proposer_missed_penalty,
        ).masked_fill(~mask, -1e4)
        shortlist_size = min(self.config.top_k, candidates.shape[1])
        shortlist_scores, shortlist_indices = torch.topk(
            proposer_scores, k=shortlist_size, dim=1
        )
        shortlist_mask = torch.gather(mask, 1, shortlist_indices)
        gather_indices = shortlist_indices.unsqueeze(-1).expand(
            -1, -1, candidates.shape[-1]
        )
        shortlist_candidates = torch.gather(candidates, 1, gather_indices)
        rank = torch.linspace(
            1.0,
            0.0,
            shortlist_size,
            dtype=candidates.dtype,
            device=candidates.device,
        )
        rank = rank.view(1, shortlist_size, 1).expand(candidates.shape[0], -1, -1)
        reranker_candidates = torch.cat(
            [shortlist_candidates, shortlist_scores.unsqueeze(-1), rank], dim=-1
        )
        reranker_outputs = self.reranker(
            context, reranker_candidates, shortlist_mask
        )
        return {
            "proposer": proposer_outputs,
            "reranker": reranker_outputs,
            "shortlist_indices": shortlist_indices,
            "shortlist_mask": shortlist_mask,
            "proposer_scores": proposer_scores,
        }


def topk_set_reranker_scores(
    outputs: dict[str, Any],
    *,
    beneficial_weight: float,
    beneficial_probability_threshold: float = 0.0,
    unsafe_weight: float,
    missed_weight: float,
) -> Tensor:
    """Map final shortlist scores back onto the original candidate axis."""

    shortlist_scores = risk_adjusted_scores(
        outputs["reranker"],
        beneficial_weight=beneficial_weight,
        unsafe_weight=unsafe_weight,
        missed_weight=missed_weight,
    )
    if beneficial_probability_threshold > 0.0:
        beneficial_probability = torch.sigmoid(
            outputs["reranker"]["beneficial_logit"]
        )
        shortlist_scores = shortlist_scores.masked_fill(
            beneficial_probability < beneficial_probability_threshold, -1e4
        )
    shortlist_scores = shortlist_scores.masked_fill(~outputs["shortlist_mask"], -1e4)
    full_scores = torch.full_like(outputs["proposer_scores"], -1e4)
    return full_scores.scatter(1, outputs["shortlist_indices"], shortlist_scores)


class EvidenceValuePredictor:
    """Callable rollout adapter with a calibrated explicit STOP score."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        stop_margin_override: float | None = None,
    ) -> None:
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
        self.feature_set = str(checkpoint.get("feature_set", LEGACY_FEATURE_SET))
        if self.model.config.candidate_dim != candidate_feature_dim(self.feature_set):
            raise ValueError("checkpoint feature set and candidate dimension disagree")
        self.checkpoint_stop_margin = float(
            checkpoint.get("stop_margin", checkpoint["safety_margin"])
        )
        self.stop_margin = (
            float(stop_margin_override)
            if stop_margin_override is not None
            else self.checkpoint_stop_margin
        )
        self.safety_margin = self.stop_margin
        data_contract = checkpoint.get("data_contract")
        if data_contract is not None and not isinstance(data_contract, dict):
            raise ValueError("Evidence Value checkpoint data_contract must be a mapping")
        self.data_contract = dict(data_contract) if data_contract is not None else None
        self.unsafe_penalty = float(checkpoint.get("unsafe_penalty", 0.0))
        self.missed_penalty = float(checkpoint.get("missed_penalty", 0.0))

    @torch.no_grad()
    def score_sample(self, sample: SelectorSample) -> np.ndarray:
        context, candidates = evidence_value_features(
            sample, feature_set=self.feature_set
        )
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

    def action_scores(self, sample: SelectorSample) -> np.ndarray:
        return self(sample)


class TopKSetEvidenceValuePredictor:
    """Rollout adapter for a frozen two-stage top-k Evidence Value policy."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        stop_margin_override: float | None = None,
    ) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("protocol") != "topk_set_evidence_reranker_v1":
            raise ValueError("unsupported top-k Evidence Value checkpoint protocol")
        self.model = TopKSetEvidenceValueHead(
            TopKSetEvidenceValueConfig(**checkpoint["model_config"])
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.normalizer = EvidenceValueNormalizer.from_dict(checkpoint["normalizer"])
        self.feature_set = str(checkpoint.get("feature_set", LEGACY_FEATURE_SET))
        if self.model.config.candidate_dim != candidate_feature_dim(self.feature_set):
            raise ValueError("checkpoint feature set and candidate dimension disagree")
        self.checkpoint_stop_margin = float(
            checkpoint.get("stop_margin", checkpoint["safety_margin"])
        )
        self.stop_margin = (
            float(stop_margin_override)
            if stop_margin_override is not None
            else self.checkpoint_stop_margin
        )
        self.safety_margin = self.stop_margin
        data_contract = checkpoint.get("data_contract")
        if data_contract is not None and not isinstance(data_contract, dict):
            raise ValueError("top-k checkpoint data_contract must be a mapping")
        self.data_contract = dict(data_contract) if data_contract is not None else None
        self.beneficial_weight = float(checkpoint.get("beneficial_score_weight", 0.0))
        self.beneficial_probability_threshold = float(
            checkpoint.get("beneficial_probability_threshold", 0.0)
        )
        self.unsafe_penalty = float(checkpoint.get("unsafe_penalty", 0.0))
        self.missed_penalty = float(checkpoint.get("missed_penalty", 0.0))

    @torch.no_grad()
    def score_sample(self, sample: SelectorSample) -> np.ndarray:
        context, candidates = evidence_value_features(
            sample, feature_set=self.feature_set
        )
        context, candidates = self.normalizer.transform(context, candidates)
        context_tensor = torch.from_numpy(context).to(self.device).unsqueeze(0)
        candidate_tensor = torch.from_numpy(candidates).to(self.device).unsqueeze(0)
        mask = torch.ones(
            (1, len(candidates)), dtype=torch.bool, device=self.device
        )
        outputs = self.model(context_tensor, candidate_tensor, mask)
        return (
            topk_set_reranker_scores(
                outputs,
                beneficial_weight=self.beneficial_weight,
                beneficial_probability_threshold=(
                    self.beneficial_probability_threshold
                ),
                unsafe_weight=self.unsafe_penalty,
                missed_weight=self.missed_penalty,
            )[0]
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    def __call__(self, sample: SelectorSample) -> np.ndarray:
        scores = self.score_sample(sample)
        return np.concatenate(
            [scores, np.asarray([self.safety_margin], dtype=np.float32)]
        )

    def action_scores(self, sample: SelectorSample) -> np.ndarray:
        return self(sample)


def load_evidence_value_predictor(
    checkpoint_path: str,
    device: str = "cpu",
    stop_margin_override: float | None = None,
) -> EvidenceValuePredictor | TopKSetEvidenceValuePredictor:
    """Load either Evidence Value protocol without weakening protocol checks."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    protocol = checkpoint.get("protocol")
    if protocol == "evidence_value_head_v1":
        return EvidenceValuePredictor(
            checkpoint_path,
            device=device,
            stop_margin_override=stop_margin_override,
        )
    if protocol == "topk_set_evidence_reranker_v1":
        return TopKSetEvidenceValuePredictor(
            checkpoint_path,
            device=device,
            stop_margin_override=stop_margin_override,
        )
    raise ValueError(f"unsupported Evidence Value checkpoint protocol: {protocol!r}")


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
        self.feature_set = str(checkpoint.get("feature_set", LEGACY_FEATURE_SET))
        if self.model.config.candidate_dim != candidate_feature_dim(self.feature_set):
            raise ValueError("checkpoint feature set and candidate dimension disagree")
        self.safety_margin = float(checkpoint["safety_margin"])
        self.unsafe_penalty = float(checkpoint.get("unsafe_penalty", 0.0))
        self.missed_penalty = float(checkpoint.get("missed_penalty", 0.0))
        self.last_terminal_edit: EditOperation | None = None

    @torch.no_grad()
    def predict(self, sample: SelectorSample) -> tuple[np.ndarray, EditOperation]:
        context, candidates = evidence_value_features(
            sample, feature_set=self.feature_set
        )
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
