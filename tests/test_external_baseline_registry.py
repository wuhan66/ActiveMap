import json
from pathlib import Path

import pytest
import yaml

from scripts.audit_external_baselines import audit_registry
from scripts.prepare_external_baseline_run import build_manifest

REGISTRY = Path("configs/experiments/external_baselines.yaml")


def test_repository_external_baseline_registry_is_valid() -> None:
    report = audit_registry(REGISTRY)
    assert report["protocol_valid"] is True
    assert report["ready_for_frozen_test"] is False
    assert report["frozen_test_complete"] is False
    assert report["summary"]["suites"] == 4
    assert report["summary"]["baselines"] == 36
    assert report["summary"]["critical"] == 11


def test_validation_complete_is_ready_but_not_frozen_complete(tmp_path: Path) -> None:
    payload = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    for suite in payload["suites"]:
        for baseline in suite["baselines"]:
            if baseline.get("priority", suite["priority"]) == "critical":
                baseline["status"] = "validation_complete"
                baseline["source_commit"] = baseline.get("source_commit") or "test-commit"
    registry = tmp_path / "registry.yaml"
    registry.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    report = audit_registry(registry)

    assert report["ready_for_frozen_test"] is True
    assert report["frozen_test_complete"] is False
    assert report["summary"]["critical_validation_ready"] == 11
    assert report["summary"]["critical_frozen_test_complete"] == 0


def test_declared_validation_evidence_must_exist(tmp_path: Path) -> None:
    payload = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    baseline = payload["suites"][-1]["baselines"][0]
    baseline["validation_evidence"] = "artifacts/does-not-exist/result.json"
    registry = tmp_path / "registry.yaml"
    registry.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    report = audit_registry(registry)

    assert report["protocol_valid"] is False
    assert any("validation_evidence does not exist" in error for error in report["errors"])


def test_all_critical_frozen_results_close_both_gates(tmp_path: Path) -> None:
    payload = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    for suite in payload["suites"]:
        for baseline in suite["baselines"]:
            if baseline.get("priority", suite["priority"]) == "critical":
                baseline["status"] = "frozen_test_complete"
                baseline["source_commit"] = baseline.get("source_commit") or "test-commit"
    registry = tmp_path / "registry.yaml"
    registry.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    report = audit_registry(registry)

    assert report["ready_for_frozen_test"] is True
    assert report["frozen_test_complete"] is True


def test_run_manifest_requires_source_lock(tmp_path: Path) -> None:
    dataset_manifest = tmp_path / "dataset.json"
    dataset_manifest.write_text(json.dumps({"split": "validation"}), encoding="utf-8")
    with pytest.raises(ValueError, match="source_commit"):
        build_manifest(
            REGISTRY,
            "sn7_building_map_update",
            "changeformer",
            "validate",
            1,
            "validation",
            dataset_manifest,
            tmp_path / "run",
        )


def test_run_manifest_hashes_inputs(tmp_path: Path) -> None:
    payload = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    registry = tmp_path / "registry.yaml"
    registry.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    dataset_manifest = tmp_path / "dataset.json"
    dataset_manifest.write_text(json.dumps({"split": "validation"}), encoding="utf-8")
    output_dir = tmp_path / "run"
    manifest = build_manifest(
        registry,
        "sn7_building_map_update",
        "concat_unet",
        "validate",
        20260716,
        "validation",
        dataset_manifest,
        output_dir,
    )
    assert manifest["schema_version"] == "activemap-external-baseline-run-v1"
    assert len(manifest["inputs"]["registry_sha256"]) == 64
    assert len(manifest["inputs"]["dataset_manifest_sha256"]) == 64
    assert manifest["source"]["commit"] == "in_tree"
    assert manifest["source"]["publication_year"] == 2026


def test_unlicensed_public_source_cannot_start(tmp_path: Path) -> None:
    dataset_manifest = tmp_path / "dataset.json"
    dataset_manifest.write_text(json.dumps({"split": "validation"}), encoding="utf-8")
    payload = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    baseline = next(
        item
        for suite in payload["suites"]
        for item in suite["baselines"]
        if item["id"] == "sam_road_plus_plus"
    )
    baseline["source_commit"] = "deadbeef"
    registry = tmp_path / "registry.yaml"
    registry.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="no_license_detected"):
        build_manifest(
            registry,
            "muno21_official_map_update",
            "sam_road_plus_plus",
            "validate",
            1,
            "validation",
            dataset_manifest,
            tmp_path / "run",
        )
