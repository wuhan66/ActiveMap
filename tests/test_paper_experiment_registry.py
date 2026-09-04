import json
from pathlib import Path

import yaml

from scripts.audit_paper_experiment_registry import audit_registry


def test_repository_paper_registry_is_protocol_valid(tmp_path: Path) -> None:
    registry = Path("configs/experiments/paper_registry.yaml")
    registry_payload = yaml.safe_load(registry.read_text(encoding="utf-8"))
    report = audit_registry(registry, tmp_path / "storage")
    assert report["protocol_valid"] is True
    assert report["ready_for_frozen_test"] is False
    assert report["artifact_summary"] == {"ready": 0, "total": 47}
    assert report["experiment_summary"]["total"] == 37
    assert len(registry_payload["required_paired_comparisons"]) == 24
    assert registry_payload["protocol"]["paired_comparison_margins"] == {
        "min_primary_utility_delta": 0.0,
        "min_quality_cost_auc_delta": 0.0,
        "max_false_edit_rate_delta": 0.01,
        "max_mean_cost_delta_for_matched_cost": 0.0,
    }

    selector_matrix = yaml.safe_load(
        Path("configs/experiments/muno21_selector_paper_ablations.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert selector_matrix["seeds"] == [20260821, 20260822, 20260823]
    assert {row["name"] for row in selector_matrix["experiments"]} == {
        "generic",
        "full",
        "no_stop",
    }


def test_registry_audit_reads_json_gates(tmp_path: Path) -> None:
    storage = tmp_path / "storage"
    storage.mkdir()
    gate = storage / "gate.json"
    gate.write_text(json.dumps({"outer": {"approved": True}}), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("seed: 1\n", encoding="utf-8")
    registry = {
        "protocol": {"test_policy": "frozen_once"},
        "required_artifacts": [
            {
                "id": "manual_qc_approval",
                "stage": "updater",
                "path": "${STORAGE_ROOT}/gate.json",
                "kind": "json_gate",
                "field": "outer.approved",
                "equals": True,
            }
        ],
        "experiments": [
            {
                "id": family,
                "family": family,
                "role": "ablation",
                "trainable": False,
                "requires": ["manual_qc_approval"],
                "test_policy": "validation_only",
            }
            for family in ("updater", "selector", "agent", "rl", "writeback")
        ],
    }
    registry_path = tmp_path / "a" / "b" / "registry.yaml"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(yaml.safe_dump(registry), encoding="utf-8")
    report = audit_registry(registry_path, storage)
    assert report["protocol_valid"] is True
    assert report["artifacts"][0]["ready"] is True
    assert report["experiment_summary"]["ready"] == 5
