"""Torch data pipeline for variable-size evidence candidate sets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from activemap.features import AblationSpec, apply_ablation
from activemap.selector_records import SelectorSample


@dataclass(frozen=True)
class SelectorFeatureNormalizer:
    hypothesis_mean: np.ndarray
    hypothesis_std: np.ndarray
    evidence_mean: np.ndarray
    evidence_std: np.ndarray
    state_mean: np.ndarray
    state_std: np.ndarray

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "hypothesis_mean": self.hypothesis_mean.tolist(),
            "hypothesis_std": self.hypothesis_std.tolist(),
            "evidence_mean": self.evidence_mean.tolist(),
            "evidence_std": self.evidence_std.tolist(),
            "state_mean": self.state_mean.tolist(),
            "state_std": self.state_std.tolist(),
        }

    @classmethod
    def from_dict(
        cls, payload: dict[str, list[float]]
    ) -> SelectorFeatureNormalizer:
        return cls(
            **{
                key: np.asarray(value, dtype=np.float32)
                for key, value in payload.items()
            }
        )

    def transform(
        self,
        hypothesis: np.ndarray,
        evidence: np.ndarray,
        state: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            (hypothesis - self.hypothesis_mean) / self.hypothesis_std,
            (evidence - self.evidence_mean) / self.evidence_std,
            (state - self.state_mean) / self.state_std,
        )


def _mean_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = values.std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(std, np.float32(1e-6))


def fit_selector_feature_normalizer(
    samples: list[SelectorSample],
) -> SelectorFeatureNormalizer:
    if not samples:
        raise ValueError("normalizer requires at least one fit sample")
    hypothesis = np.asarray(
        [sample.hypothesis_features for sample in samples], dtype=np.float32
    )
    state = np.asarray([sample.state_features for sample in samples], dtype=np.float32)
    evidence = np.concatenate(
        [np.asarray(sample.evidence_features, dtype=np.float32) for sample in samples],
        axis=0,
    )
    hypothesis_mean, hypothesis_std = _mean_std(hypothesis)
    evidence_mean, evidence_std = _mean_std(evidence)
    state_mean, state_std = _mean_std(state)
    return SelectorFeatureNormalizer(
        hypothesis_mean=hypothesis_mean,
        hypothesis_std=hypothesis_std,
        evidence_mean=evidence_mean,
        evidence_std=evidence_std,
        state_mean=state_mean,
        state_std=state_std,
    )


def load_selector_samples(path: Path, *, split: str | None = None) -> list[SelectorSample]:
    records: list[SelectorSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = SelectorSample.model_validate_json(line)
            except Exception as exc:
                raise ValueError(f"invalid selector sample at line {line_number}") from exc
            if split is None or record.split == split:
                records.append(record)
    if not records:
        raise ValueError(f"no selector samples found for split={split!r} in {path}")
    return records


class SelectorDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        samples: list[SelectorSample],
        ablation: AblationSpec,
        normalizer: SelectorFeatureNormalizer | None = None,
    ) -> None:
        self.samples = samples
        self.ablation = ablation
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        hypothesis = np.asarray(sample.hypothesis_features, dtype=np.float32)
        evidence = np.asarray(sample.evidence_features, dtype=np.float32)
        state = np.asarray(sample.state_features, dtype=np.float32)
        if self.normalizer is not None:
            hypothesis, evidence, state = self.normalizer.transform(
                hypothesis, evidence, state
            )
        hypothesis, evidence, state = apply_ablation(
            hypothesis, evidence, state, self.ablation
        )
        utilities = np.asarray(sample.oracle_utilities, dtype=np.float32)
        if not self.ablation.false_edit_penalty:
            utilities = utilities + sample.false_edit_penalty_weight * np.asarray(
                sample.false_edit_risks, dtype=np.float32
            )
        return {
            "sample_id": sample.sample_id,
            "hypothesis": hypothesis,
            "evidence": evidence,
            "state": state,
            "utilities": utilities,
            "stop_utility": (
                float(sample.stop_utility) if self.ablation.allow_stop else float("-inf")
            ),
        }


def collate_selector_batch(items: list[dict[str, Any]]) -> dict[str, Tensor | list[str]]:
    batch_size = len(items)
    max_candidates = max(item["evidence"].shape[0] for item in items)
    evidence_dim = items[0]["evidence"].shape[1]
    evidence = torch.zeros((batch_size, max_candidates, evidence_dim), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_candidates), dtype=torch.bool)
    utilities = torch.full((batch_size, max_candidates + 1), -torch.inf, dtype=torch.float32)
    targets = torch.empty(batch_size, dtype=torch.long)
    hypotheses = torch.from_numpy(np.stack([item["hypothesis"] for item in items]))
    states = torch.from_numpy(np.stack([item["state"] for item in items]))

    for batch_index, item in enumerate(items):
        candidate_count = item["evidence"].shape[0]
        evidence[batch_index, :candidate_count] = torch.from_numpy(item["evidence"])
        mask[batch_index, :candidate_count] = True
        utilities[batch_index, :candidate_count] = torch.from_numpy(item["utilities"])
        utilities[batch_index, max_candidates] = item["stop_utility"]
        targets[batch_index] = int(torch.argmax(utilities[batch_index]))

    return {
        "sample_ids": [str(item["sample_id"]) for item in items],
        "hypothesis": hypotheses,
        "evidence": evidence,
        "state": states,
        "mask": mask,
        "utilities": utilities,
        "targets": targets,
    }
