import json
from pathlib import Path

import yaml

from scripts.audit_cross_dataset_protocol import audit
from scripts.build_cross_dataset_evidence_matrix import build_matrix


PROTOCOL = Path("configs/experiments/cross_dataset_protocol.yaml")


def test_repository_cross_dataset_protocol_passes() -> None:
    report = audit(PROTOCOL, Path(".").resolve())
    assert report["passed"] is True
    assert report["checks"]["sample_pooling_forbidden"] is True
    assert report["checks"]["transfer_configs_valid"] is True


def _sn7(path: Path, *, test_assets_read: bool = False) -> Path:
    payload = {
        "schema_version": "active-catalog-tool-writeback-promotion-v1",
        "promote": True,
        "checks": {"required_seed_count": True},
        "minimum_seed_count": 3,
        "test_assets_read": test_assets_read,
        "formalized_from_completed_ledger": test_assets_read,
        "frozen_test_ledger_sha256": "a" * 64 if test_assets_read else None,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _muno(path: Path) -> Path:
    payload = {
        "schema_version": "activemap-paired-rollout-report-v1",
        "comparison_id": "agent_vs_edit_conditioned",
        "split": "test",
        "test_assets_read": True,
        "all_required_gates_passed": True,
        "baseline": {"seeds": [1, 2, 3]},
        "candidate": {"seeds": [1, 2, 3]},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_validation_sn7_cannot_support_cross_geometry_claim(tmp_path: Path) -> None:
    result = build_matrix(
        PROTOCOL,
        sn7_promotion_path=_sn7(tmp_path / "sn7.json"),
        muno21_report_path=_muno(tmp_path / "muno.json"),
    )
    assert result["gates"]["sn7_independent_gain"] is True
    assert result["gates"]["sn7_frozen_test"] is False
    assert result["gates"]["cross_geometry_generalization_supported"] is False
    assert result["pooled_score"] is None


def test_independent_frozen_evidence_supports_cross_geometry_claim(tmp_path: Path) -> None:
    result = build_matrix(
        PROTOCOL,
        sn7_promotion_path=_sn7(tmp_path / "sn7.json", test_assets_read=True),
        muno21_report_path=_muno(tmp_path / "muno.json"),
    )
    assert result["claim_status"] == "cross_geometry_frozen_test_supported"


def test_inria_agent_role_is_rejected(tmp_path: Path) -> None:
    payload = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    payload["datasets"]["inria"]["agent_evaluation"] = True
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    assert audit(path, Path(".").resolve())["passed"] is False
