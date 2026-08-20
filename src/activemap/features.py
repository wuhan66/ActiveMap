"""Canonical feature layouts shared by generation, models, and ablations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HYPOTHESIS_DIM = 16
EVIDENCE_DIM = 13
STATE_DIM = 8

HYPOTHESIS_GROUPS: dict[str, slice] = {
    "edit_type": slice(0, 4),
    "prior_geometry": slice(4, 12),
    "conflict": slice(12, 13),
    "confidence": slice(13, 14),
    "history": slice(14, 16),
}

EVIDENCE_GROUPS: dict[str, slice] = {
    "quality": slice(0, 2),
    "time": slice(2, 4),
    "scale": slice(4, 7),
    "region": slice(7, 11),
    "uncertainty": slice(11, 12),
    "cost": slice(12, 13),
}

STATE_GROUPS: dict[str, slice] = {
    "budget": slice(0, 2),
    "belief": slice(2, 6),
    "history": slice(6, 8),
}

# Online controllers may only consume state available before the next map edit.
# This contract makes state[7] a belief summary rather than realised utility.
ONLINE_OBSERVABLE_STATE_CONTRACT: dict[str, str] = {
    "version": "online-observable-state-v1",
    "state7": "fused_belief_confidence",
}


@dataclass(frozen=True)
class AblationSpec:
    name: str = "full"
    condition_on_hypothesis: bool = True
    drop_hypothesis_groups: tuple[str, ...] = ()
    drop_evidence_groups: tuple[str, ...] = ()
    drop_state_groups: tuple[str, ...] = ()
    false_edit_penalty: bool = True
    allow_stop: bool = True


def _mask_groups(
    values: np.ndarray, groups: dict[str, slice], dropped: tuple[str, ...]
) -> np.ndarray:
    output = np.asarray(values, dtype=np.float32).copy()
    for name in dropped:
        if name not in groups:
            raise ValueError(f"unknown feature group: {name}")
        output[..., groups[name]] = 0.0
    return output


def apply_ablation(
    hypothesis: np.ndarray,
    evidence: np.ndarray,
    state: np.ndarray,
    spec: AblationSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if hypothesis.shape[-1] != HYPOTHESIS_DIM:
        raise ValueError(f"hypothesis feature dimension must be {HYPOTHESIS_DIM}")
    if evidence.shape[-1] != EVIDENCE_DIM:
        raise ValueError(f"evidence feature dimension must be {EVIDENCE_DIM}")
    if state.shape[-1] != STATE_DIM:
        raise ValueError(f"state feature dimension must be {STATE_DIM}")
    hypothesis_output = _mask_groups(hypothesis, HYPOTHESIS_GROUPS, spec.drop_hypothesis_groups)
    if not spec.condition_on_hypothesis:
        hypothesis_output[...] = 0.0
    return (
        hypothesis_output,
        _mask_groups(evidence, EVIDENCE_GROUPS, spec.drop_evidence_groups),
        _mask_groups(state, STATE_GROUPS, spec.drop_state_groups),
    )
