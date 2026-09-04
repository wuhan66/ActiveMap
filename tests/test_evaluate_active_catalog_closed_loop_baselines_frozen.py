import pytest

from scripts.evaluate_active_catalog_closed_loop_baselines import authorize_split


def test_validation_rejects_frozen_flag():
    with pytest.raises(ValueError, match="only for the test split"):
        authorize_split("val", True)


def test_test_requires_frozen_flag():
    with pytest.raises(PermissionError, match="requires --frozen-test"):
        authorize_split("test", False)


def test_validation_is_not_test_access():
    assert authorize_split("val", False) is False
