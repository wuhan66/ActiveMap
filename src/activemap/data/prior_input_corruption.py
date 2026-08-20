"""Deterministic prior-input corruption for robustness evaluation."""

from __future__ import annotations

import random

import numpy as np


def translate_no_wrap(
    value: np.ndarray,
    shift_y: int,
    shift_x: int,
) -> np.ndarray:
    """Translate a 2D array while filling exposed pixels with zeros."""

    if value.ndim != 2:
        raise ValueError("prior input must be a 2D array")
    output = np.zeros_like(value)
    height, width = value.shape
    source_y0 = max(-shift_y, 0)
    source_y1 = min(height - shift_y, height)
    source_x0 = max(-shift_x, 0)
    source_x1 = min(width - shift_x, width)
    if source_y0 >= source_y1 or source_x0 >= source_x1:
        return output
    target_y0 = source_y0 + shift_y
    target_y1 = source_y1 + shift_y
    target_x0 = source_x0 + shift_x
    target_x1 = source_x1 + shift_x
    output[target_y0:target_y1, target_x0:target_x1] = value[
        source_y0:source_y1,
        source_x0:source_x1,
    ]
    return output


def deterministic_prior_translation(
    value: np.ndarray,
    *,
    identity: str,
    max_pixels: int,
    seed: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Apply a stable per-episode translation to the model's prior input."""

    if max_pixels < 0:
        raise ValueError("max_pixels must be non-negative")
    if max_pixels == 0:
        return value.copy(), (0, 0)
    rng = random.Random(f"{seed}:{identity}")
    shift = (
        rng.randint(-max_pixels, max_pixels),
        rng.randint(-max_pixels, max_pixels),
    )
    return translate_no_wrap(value, *shift), shift


def morph_prior_no_wrap(
    value: np.ndarray,
    *,
    operation: str,
    pixels: int,
) -> np.ndarray:
    """Dilate or erode a 2D prior without treating opposite edges as adjacent."""

    if value.ndim != 2:
        raise ValueError("prior input must be a 2D array")
    if operation not in {"none", "dilate", "erode"}:
        raise ValueError("prior morphology must be none, dilate, or erode")
    if pixels < 0:
        raise ValueError("prior morphology pixels must be non-negative")
    if operation == "none":
        if pixels != 0:
            raise ValueError("none morphology requires zero pixels")
        return value.copy()
    if pixels == 0:
        raise ValueError(f"{operation} morphology requires positive pixels")

    height, width = value.shape
    reducer = np.maximum if operation == "dilate" else np.minimum
    padded_x = np.pad(value, ((0, 0), (pixels, pixels)), constant_values=0)
    horizontal = padded_x[:, :width].copy()
    for offset in range(1, 2 * pixels + 1):
        reducer(horizontal, padded_x[:, offset : offset + width], out=horizontal)

    padded_y = np.pad(horizontal, ((pixels, pixels), (0, 0)), constant_values=0)
    output = padded_y[:height].copy()
    for offset in range(1, 2 * pixels + 1):
        reducer(output, padded_y[offset : offset + height], out=output)
    return output
