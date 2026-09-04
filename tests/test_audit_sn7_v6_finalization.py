from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.audit_sn7_v6_finalization import audit, markdown

POLICIES = [
    "direct_commit",
    "direct_safe_commit",
    "selected_commit",
    "selected_safe_commit",
    "forced_safe_commit",
]
SEEDS = [20260817, 20260818, 20260819]
PROMOTION_CHECKS = {
    "selection_final_map_quality_lower_positive": False,
    "selection_false_edit_upper_nonpositive": True,
    "selection_missed_edit_upper_nonpositive": True,
    "safe_commit_false_edit_upper_nonpositive": True,
    "safe_commit_final_map_quality_lower_nonnegative": True,
    "selected_cost_upper_strictly_below_forced": True,
}


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_completed_v6_root(tmp_path):
    v6_root = tmp_path / "v6"
    finalization_dir = tmp_path / "finalization"
    v6_root.mkdir()
    finalization_dir.mkdir()

    protocol_path = v6_root / "protocol.json"
    _write_json(protocol_path, {"test_assets_read": False})
    protocol_sha256 = hashlib.sha256(protocol_path.read_bytes()).hexdigest()

    _write_json(
        v6_root / "queue_status.json",
        {"status": "complete", "test_assets_read": False},
    )
    _write_json(
        finalization_dir / "finalization_receipt.json",
        {
            "schema_version": "sn7-v6-evidence-value-finalization-v1",
            "test_assets_read": False,
            "forced_safe_commit_only_uses_train_fitted_gates": True,
            "v6_protocol_sha256": protocol_sha256,
        },
    )
    _write_json(
        finalization_dir / "three_seed_factorial_with_forced_summary.json",
        {
            "schema_version": "sn7-v6-evidence-value-matched-factorial-v1",
            "test_assets_read": False,
            "model_seeds": SEEDS,
            "policies": POLICIES,
            "v6_protocol_sha256": protocol_sha256,
            "promotion": {
                "eligible_for_extension_claim": False,
                "checks": PROMOTION_CHECKS,
                "reason": "Quality gate did not pass.",
                "forced_acquisition_cost_control": {"paired_delta": -0.1},
            },
        },
    )
    return v6_root, finalization_dir


def test_audit_accepts_completed_registered_v6_and_emits_summary(tmp_path):
    v6_root, finalization_dir = _make_completed_v6_root(tmp_path)

    result = audit(v6_root, finalization_dir)

    assert result["integrity_passed"] is True
    assert result["promotion_eligible"] is False
    assert result["model_seeds"] == SEEDS
    assert "Quality gate did not pass." in markdown(result)


def test_audit_rejects_incomplete_or_test_accessing_queue(tmp_path):
    v6_root, finalization_dir = _make_completed_v6_root(tmp_path)
    _write_json(
        v6_root / "queue_status.json",
        {"status": "running", "test_assets_read": False},
    )

    with pytest.raises(ValueError, match="queue is not complete"):
        audit(v6_root, finalization_dir)

    _write_json(
        v6_root / "queue_status.json",
        {"status": "complete", "test_assets_read": True},
    )

    with pytest.raises(ValueError, match="accessed test assets"):
        audit(v6_root, finalization_dir)


def test_audit_rejects_tampered_aggregate_protocol_or_policy_set(tmp_path):
    v6_root, finalization_dir = _make_completed_v6_root(tmp_path)
    aggregate_path = finalization_dir / "three_seed_factorial_with_forced_summary.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate["v6_protocol_sha256"] = "0" * 64
    _write_json(aggregate_path, aggregate)

    with pytest.raises(ValueError, match="protocol hash"):
        audit(v6_root, finalization_dir)

    aggregate["v6_protocol_sha256"] = hashlib.sha256(
        (v6_root / "protocol.json").read_bytes()
    ).hexdigest()
    aggregate["policies"] = POLICIES[:-1]
    _write_json(aggregate_path, aggregate)

    with pytest.raises(ValueError, match="registered five policy cells"):
        audit(v6_root, finalization_dir)


def test_audit_rejects_incomplete_or_inconsistent_promotion_checks(tmp_path):
    v6_root, finalization_dir = _make_completed_v6_root(tmp_path)
    aggregate_path = finalization_dir / "three_seed_factorial_with_forced_summary.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate["promotion"]["checks"].pop("selected_cost_upper_strictly_below_forced")
    _write_json(aggregate_path, aggregate)

    with pytest.raises(ValueError, match="six-gate"):
        audit(v6_root, finalization_dir)

    aggregate["promotion"]["checks"] = PROMOTION_CHECKS
    aggregate["promotion"]["eligible_for_extension_claim"] = True
    _write_json(aggregate_path, aggregate)

    with pytest.raises(ValueError, match="inconsistent"):
        audit(v6_root, finalization_dir)
