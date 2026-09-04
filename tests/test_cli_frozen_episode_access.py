import pytest

from activemap.frozen_test import authorize_manifest_test_access


def test_test_manifest_requires_frozen_flag():
    with pytest.raises(PermissionError, match="requires --frozen-test"):
        authorize_manifest_test_access(["test"], False)


def test_validation_manifest_rejects_frozen_flag():
    with pytest.raises(ValueError, match="containing test rows"):
        authorize_manifest_test_access(["val"], True)
