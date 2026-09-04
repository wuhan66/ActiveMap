import numpy as np
import pytest

from activemap.evaluation_controls import grouped_derangement, permutation_sha256


def test_grouped_derangement_is_seeded_and_cross_task():
    groups = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"])
    first = grouped_derangement(groups, seed=17)
    second = grouped_derangement(groups, seed=17)
    assert np.array_equal(first, second)
    assert sorted(first.tolist()) == list(range(len(groups)))
    assert np.all(groups != groups[first])
    assert permutation_sha256(first) == permutation_sha256(second)


def test_grouped_derangement_rejects_impossible_partition():
    groups = np.asarray(["a", "a", "a", "b"])
    with pytest.raises(ValueError, match="impossible"):
        grouped_derangement(groups, seed=1)
