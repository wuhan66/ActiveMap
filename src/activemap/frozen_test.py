"""Ephemeral authorization for one-shot frozen-test access."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any, Iterable


def authorize_manifest_test_access(
    splits: Iterable[str], frozen_test: bool
) -> bool:
    contains_test = any(str(value) == "test" for value in splits)
    if contains_test:
        if not frozen_test:
            raise PermissionError("test episode construction requires --frozen-test")
        assert_frozen_test_access()
    elif frozen_test:
        raise ValueError("--frozen-test requires a manifest containing test rows")
    return contains_test


def assert_frozen_test_access() -> dict[str, Any]:
    """Validate the token and active ledger created by the frozen-test runner."""
    if os.environ.get("ACTIVEMAP_DISABLE_FROZEN_TEST") == "1":
        raise PermissionError("frozen test access is disabled on this cluster role")
    if os.environ.get("ACTIVEMAP_FROZEN_TEST") != "1":
        raise PermissionError("frozen test access requires run_frozen_paper_test.py")
    ledger_value = os.environ.get("ACTIVEMAP_FROZEN_TEST_LEDGER")
    token = os.environ.get("ACTIVEMAP_FROZEN_TEST_TOKEN")
    if not ledger_value or not token:
        raise PermissionError("frozen test authorization environment is incomplete")
    ledger_path = Path(ledger_value)
    if not ledger_path.is_file():
        raise PermissionError("frozen test ledger does not exist")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if ledger.get("schema_version") != "activemap-frozen-test-access-v1":
        raise PermissionError("invalid frozen test ledger schema")
    if ledger.get("status") != "started":
        raise PermissionError("frozen test ledger is not in its one-shot execution window")
    actual = hashlib.sha256(token.encode()).hexdigest()
    expected = str(ledger.get("authorization_sha256", ""))
    if not hmac.compare_digest(actual, expected):
        raise PermissionError("frozen test authorization token mismatch")
    return ledger
