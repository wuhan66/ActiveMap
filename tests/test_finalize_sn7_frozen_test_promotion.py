import json
from pathlib import Path

import pytest

from scripts.finalize_sn7_frozen_test_promotion import finalize


def _promotion(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "active-catalog-tool-writeback-promotion-v1",
                "promote": True,
                "minimum_seed_count": 3,
                "test_assets_read": True,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_finalizer_binds_promotion_to_completed_ledger(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps(
            {
                "schema_version": "activemap-frozen-test-access-v1",
                "status": "complete",
                "returncode": 0,
            }
        ),
        encoding="utf-8",
    )
    result = finalize(ledger, _promotion(tmp_path / "promotion.json"))
    assert result["formalized_from_completed_ledger"] is True
    assert len(result["frozen_test_ledger_sha256"]) == 64


def test_finalizer_rejects_started_ledger(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    ledger.write_text('{"status":"started"}', encoding="utf-8")
    with pytest.raises(ValueError, match="not complete"):
        finalize(ledger, _promotion(tmp_path / "promotion.json"))
