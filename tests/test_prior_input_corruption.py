import numpy as np
import pytest

from activemap.data.prior_input_corruption import (
    deterministic_prior_translation,
    morph_prior_no_wrap,
    translate_no_wrap,
)


def test_translate_no_wrap_does_not_wrap_pixels() -> None:
    value = np.zeros((4, 4), dtype=np.float32)
    value[0, 0] = 1.0

    shifted = translate_no_wrap(value, 1, 2)
    removed = translate_no_wrap(value, -1, -1)

    assert shifted[1, 2] == 1.0
    assert shifted.sum() == 1.0
    assert removed.sum() == 0.0


def test_deterministic_prior_translation_is_stable_and_non_mutating() -> None:
    value = np.arange(36, dtype=np.float32).reshape(6, 6)

    first, first_shift = deterministic_prior_translation(
        value,
        identity="episode",
        max_pixels=2,
        seed=17,
    )
    second, second_shift = deterministic_prior_translation(
        value,
        identity="episode",
        max_pixels=2,
        seed=17,
    )

    assert first_shift == second_shift
    assert np.array_equal(first, second)
    assert np.array_equal(value, np.arange(36, dtype=np.float32).reshape(6, 6))


def test_zero_translation_returns_an_independent_copy() -> None:
    value = np.ones((3, 3), dtype=np.float32)
    output, shift = deterministic_prior_translation(
        value,
        identity="episode",
        max_pixels=0,
        seed=17,
    )

    assert shift == (0, 0)
    assert np.array_equal(output, value)
    assert output is not value


def test_invalid_prior_translation_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="2D"):
        translate_no_wrap(np.zeros((1, 2, 2)), 1, 1)
    with pytest.raises(ValueError, match="non-negative"):
        deterministic_prior_translation(
            np.zeros((2, 2)),
            identity="episode",
            max_pixels=-1,
            seed=0,
        )


def test_prior_morphology_dilates_and_erodes_without_mutation() -> None:
    point = np.zeros((5, 5), dtype=np.float32)
    point[2, 2] = 1.0
    block = np.zeros((5, 5), dtype=np.float32)
    block[1:4, 1:4] = 1.0

    dilated = morph_prior_no_wrap(point, operation="dilate", pixels=1)
    eroded = morph_prior_no_wrap(block, operation="erode", pixels=1)

    assert dilated.sum() == 9.0
    assert eroded.sum() == 1.0
    assert point.sum() == 1.0
    assert block.sum() == 9.0


@pytest.mark.parametrize("operation", ["dilate", "erode"])
def test_separable_radius_four_matches_four_unit_iterations(operation: str) -> None:
    rng = np.random.default_rng(20260730)
    value = (rng.random((31, 29)) > 0.6).astype(np.float32)

    direct = morph_prior_no_wrap(value, operation=operation, pixels=4)
    iterative = value
    for _ in range(4):
        iterative = morph_prior_no_wrap(iterative, operation=operation, pixels=1)

    assert np.array_equal(direct, iterative)


@pytest.mark.parametrize(
    ("operation", "pixels"),
    [("invalid", 1), ("dilate", 0), ("erode", -1), ("none", 1)],
)
def test_invalid_prior_morphology_is_rejected(operation: str, pixels: int) -> None:
    with pytest.raises(ValueError):
        morph_prior_no_wrap(
            np.zeros((3, 3), dtype=np.float32),
            operation=operation,
            pixels=pixels,
        )
