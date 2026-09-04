from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="requires Bash")


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
    policies = (
        "direct_commit",
        "direct_safe_commit",
        "selected_commit",
        "selected_safe_commit",
        "forced_safe_commit",
    )
    return {
        "schema_version": "sn7-v5-matched-nonkeep-factorial-v2",
        "split": "val",
        "test_assets_read": False,
        "model_seeds": [20260817, 20260818, 20260819],
        "policies": list(policies),
        "policy_metrics": {policy: {"three_seed_aoi_macro": metrics} for policy in policies},
        "selection_factor": {"paired_delta": contrast},
        "safe_commit_factor": {"paired_delta": contrast},
        "operation_slices": {
            operation: {
                "selection_factor": {"paired_delta": contrast},
                "safe_commit_factor": {"paired_delta": contrast},
            }
            for operation in ("ADD", "DELETE", "RESHAPE")
        },
        "promotion": {
            "eligible_for_extension_claim": False,
            "checks": {"selection_final_map_quality_lower_positive": True},
            "forced_acquisition_cost_control": {
                "paired_delta": {"additional_cost": _interval(-0.20)}
            },
        },
    }


def test_post_intake_tex_sidecar_exports_hash_bound_validation_tables(tmp_path: Path) -> None:
    project_root = Path.cwd().resolve()
    run_root = tmp_path / "v5_run"
    storage_root = tmp_path / "storage"
    run_root.mkdir()
    summary_path = run_root / "three_seed_nonkeep_factorial_with_forced_summary.json"
    summary_path.write_text(json.dumps(_summary()), encoding="utf-8")
    summary_hash = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    audit_path = run_root / "v5_matched_intake_audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "passed": True,
                "split": "val",
                "test_assets_read": False,
                "aggregate": {"path": str(summary_path.resolve()), "sha256": summary_hash},
            }
        ),
        encoding="utf-8",
    )
    environment = os.environ | {
        "PROJECT_ROOT": str(project_root),
        "STORAGE_ROOT": str(storage_root),
        "ACTIVEMAP_PYTHON": sys.executable,
        "V5_MATCHED_RUN_ROOT": str(run_root),
    }
    script = project_root / "scripts" / "queue_sn7_v5_factorial_tex_export_hdpi.sh"

    subprocess.run(
        ["bash", str(script)], check=True, env=environment, capture_output=True, text=True
    )

    output_root = run_root / "generated_tex"
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    receipt = json.loads(
        (output_root / "v5_factorial_tex_export_receipt.json").read_text(encoding="utf-8")
    )
    status = json.loads((run_root / "v5_tex_export_status.json").read_text(encoding="utf-8"))
    assert status == {"status": "complete", "split": "val", "test_assets_read": False}
    assert manifest["source_summary"] == str(summary_path.resolve())
    assert manifest["source_summary_sha256"] == summary_hash
    assert receipt["aggregate"]["sha256"] == summary_hash
    assert receipt["promotion"] == _summary()["promotion"]
    assert (output_root / "sn7_v5_factorial_policy_table.tex").is_file()
    assert (output_root / "sn7_v5_factorial_contrast_table.tex").is_file()
    assert (output_root / "sn7_v5_forced_cost_control_table.tex").is_file()
