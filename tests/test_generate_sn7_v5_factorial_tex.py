from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.generate_sn7_v5_factorial_tex import generate

POLICIES = (
    "direct_commit",
    "direct_safe_commit",
    "selected_commit",
    "selected_safe_commit",
)


def _interval(value: float) -> dict[str, float]:
    return {"delta": value, "ci95_low": value - 0.01, "ci95_high": value + 0.01}


def _summary() -> dict:
    metrics = {
        "final_map_quality": 0.70,
        "false_edit_rate": 0.10,
        "missed_edit_rate": 0.20,
        "commit_rate": 0.50,
        "additional_evidence_rate": 0.40,
        "additional_cost": 0.30,
    }
    contrast = {
        "final_map_quality": _interval(0.02),
        "false_edit_rate": _interval(-0.01),
        "missed_edit_rate": _interval(0.00),
    }
    return {
        "schema_version": "sn7-v5-matched-nonkeep-factorial-v1",
        "split": "val",
        "test_assets_read": False,
        "model_seeds": [20260817, 20260818, 20260819],
        "policies": list(POLICIES),
        "policy_metrics": {policy: {"three_seed_aoi_macro": metrics} for policy in POLICIES},
        "selection_factor": {"paired_delta": contrast},
        "safe_commit_factor": {"paired_delta": contrast},
        "operation_slices": {
            operation: {
                "selection_factor": {"paired_delta": contrast},
                "safe_commit_factor": {"paired_delta": contrast},
            }
            for operation in ("ADD", "DELETE", "RESHAPE")
        },
    }


def test_generates_validation_only_factorial_tables(tmp_path: Path) -> None:
    source = tmp_path / "summary.json"
    source.write_text(json.dumps(_summary()), encoding="utf-8")

    outputs = generate(source, tmp_path / "generated")

    policy = outputs["policy"].read_text(encoding="utf-8")
    contrasts = outputs["contrasts"].read_text(encoding="utf-8")
    assert "Selected + Safe Commit" in policy
    assert "0.7000" in policy
    assert "Selection & Reshape" in contrasts
    assert "0.0200 [0.0100, 0.0300]" in contrasts
    manifest_path = source.parent / "generated" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_summary"] == str(source.resolve())
    assert manifest["source_summary_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert manifest["promotion"] is None


def test_rejects_test_summary(tmp_path: Path) -> None:
    summary = _summary()
    summary["test_assets_read"] = True
    source = tmp_path / "summary.json"
    source.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="validation-only"):
        generate(source, tmp_path / "generated")


def test_generates_forced_cost_control_when_registered(tmp_path: Path) -> None:
    summary = _summary()
    summary["schema_version"] = "sn7-v5-matched-nonkeep-factorial-v2"
    summary["policies"].append("forced_safe_commit")
    summary["promotion"] = {
        "forced_acquisition_cost_control": {
            "paired_delta": {"additional_cost": _interval(-0.20)}
        }
    }
    source = tmp_path / "summary.json"
    source.write_text(json.dumps(summary), encoding="utf-8")

    outputs = generate(source, tmp_path / "generated")

    assert "forced_cost" in outputs
    assert "Negative values favor" in outputs["forced_cost"].read_text(encoding="utf-8")
    manifest_path = source.parent / "generated" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["promotion"] == summary["promotion"]
