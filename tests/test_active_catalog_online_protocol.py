from pathlib import Path

import pytest

from scripts.audit_active_catalog_policy_pair import audit
from scripts.train_active_catalog_vlm_dpo import audit_onpolicy_preferences


def _preference(snapshot: Path):
    return {
        "on_policy_executed_recurrent": True,
        "rollout_policy_snapshot": str(snapshot),
        "test_assets_read": False,
    }


def test_onpolicy_preferences_must_match_updated_adapter(tmp_path):
    adapter = tmp_path / "adapter"
    rows = [_preference(adapter)]
    assert audit_onpolicy_preferences(rows, rows, adapter)["on_policy"] is True
    with pytest.raises(ValueError, match="snapshot mismatch"):
        audit_onpolicy_preferences(rows, rows, tmp_path / "different")


def test_policy_pair_requires_identical_tool_and_budget_protocol():
    summary = {
        "split": "val", "sample_count": 10, "test_assets_read": False,
        "protocol": {
            "tool_mode": "selective", "max_candidates": 16,
            "max_acquisitions": 2, "stochastic_policy_sampling": False,
        },
    }
    assert audit(summary, summary)["passed"] is True
    different = {**summary, "protocol": {**summary["protocol"], "tool_mode": "none"}}
    assert audit(summary, different)["passed"] is False
