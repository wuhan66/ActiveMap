"""Deterministic controls for validation-only feature interventions."""

from __future__ import annotations

import hashlib

import numpy as np


def grouped_derangement(groups: np.ndarray, *, seed: int) -> np.ndarray:
    """Return a permutation that maps every row to a row from another group."""
    values = np.asarray(groups)
    if values.ndim != 1:
        raise ValueError("groups must be one-dimensional")
    count = len(values)
    if count < 2:
        raise ValueError("at least two rows are required for a grouped derangement")
    _, frequencies = np.unique(values, return_counts=True)
    if int(frequencies.max()) * 2 > count:
        raise ValueError("grouped derangement is impossible for the largest group")

    rng = np.random.default_rng(seed)
    base = np.arange(count)
    for offset in rng.permutation(np.arange(1, count)):
        candidate = np.roll(base, int(offset))
        if np.all(values != values[candidate]):
            return candidate
    for _ in range(10_000):
        candidate = rng.permutation(base)
        if np.all(values != values[candidate]):
            return candidate
    raise RuntimeError("failed to construct grouped derangement")


def permutation_sha256(indices: np.ndarray) -> str:
    values = np.asarray(indices, dtype="<i8")
    if values.ndim != 1:
        raise ValueError("indices must be one-dimensional")
    return hashlib.sha256(values.tobytes()).hexdigest()
