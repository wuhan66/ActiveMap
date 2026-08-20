"""Small deterministic benchmark used to test the entire selector pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from activemap.features import EVIDENCE_DIM, HYPOTHESIS_DIM, STATE_DIM
from activemap.models import EditOperation
from activemap.selector_records import SelectorSample

EDIT_ORDER = list(EditOperation)


def _split_for_index(index: int, sample_count: int) -> str:
    ratio = index / sample_count
    if ratio < 0.70:
        return "train"
    if ratio < 0.85:
        return "val"
    return "test"


def generate_selector_smoke_samples(
    *,
    sample_count: int = 512,
    candidate_count: int = 8,
    seed: int = 20260710,
) -> list[SelectorSample]:
    rng = np.random.default_rng(seed)
    samples: list[SelectorSample] = []
    for sample_index in range(sample_count):
        operation = EDIT_ORDER[sample_index % len(EDIT_ORDER)]
        hypothesis = rng.normal(0.0, 0.25, size=HYPOTHESIS_DIM).astype(np.float32)
        hypothesis[:4] = 0.0
        hypothesis[EDIT_ORDER.index(operation)] = 1.0
        hypothesis[12] = rng.uniform(0.2, 1.0)
        hypothesis[13] = rng.uniform(0.4, 0.95)

        state = rng.uniform(0.0, 1.0, size=STATE_DIM).astype(np.float32)
        state[0] = rng.choice([0.125, 0.25, 0.5, 1.0])
        evidence = rng.uniform(0.0, 1.0, size=(candidate_count, EVIDENCE_DIM)).astype(np.float32)
        scale_index = rng.integers(0, 3, size=candidate_count)
        evidence[:, 4:7] = 0.0
        evidence[np.arange(candidate_count), 4 + scale_index] = 1.0
        costs = 1.0 + 0.25 * scale_index + 0.2 * (1.0 - evidence[:, 0])
        evidence[:, 12] = costs / max(float(np.max(costs)), 1e-6)
        risks = np.clip(0.65 * (1.0 - evidence[:, 0]) + 0.35 * evidence[:, 11], 0, 1)

        quality = evidence[:, 0]
        temporal_late = evidence[:, 2]
        temporal_consistency = evidence[:, 3]
        fine_scale = evidence[:, 4]
        broad_context = evidence[:, 6]
        coverage = evidence[:, 9]
        uncertainty = evidence[:, 11]
        if operation == EditOperation.ADD:
            gain = 0.9 * temporal_late + 0.8 * temporal_consistency + 0.5 * broad_context
        elif operation == EditOperation.DELETE:
            gain = 0.7 * temporal_late + 1.0 * temporal_consistency + 0.5 * coverage
        elif operation == EditOperation.RESHAPE:
            gain = 1.0 * fine_scale + 0.7 * quality + 0.4 * uncertainty
        else:
            gain = 0.7 * quality - 0.8 * uncertainty - 0.4 * broad_context
        utility = gain - 0.18 * costs - 0.35 * risks
        stop_utility = 0.15 if operation == EditOperation.KEEP and state[2] > 0.65 else -0.25

        samples.append(
            SelectorSample(
                sample_id=f"smoke-{sample_index:06d}",
                split=_split_for_index(sample_index, sample_count),
                edit_type=operation,
                hypothesis_features=hypothesis.tolist(),
                state_features=state.tolist(),
                evidence_ids=[
                    f"ev-{sample_index:06d}-{index:02d}" for index in range(candidate_count)
                ],
                evidence_features=evidence.tolist(),
                evidence_costs=costs.tolist(),
                false_edit_risks=risks.tolist(),
                oracle_utilities=utility.tolist(),
                stop_utility=float(stop_utility),
                metadata={"generator": "selector_smoke_v1", "seed": seed},
            )
        )
    return samples


def write_selector_samples(samples: list[SelectorSample], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for sample in samples:
            payload = sample.model_dump(mode="json")
            handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
