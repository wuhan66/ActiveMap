import hashlib
import json
from pathlib import Path

import pytest

from scripts.export_sn7_step0_frozen_test_tables_v3_cap20 import export
from scripts.export_sn7_step0_paper_tables import (
    CONTROLLER_METRICS,
    WRITEBACK_METRICS,
)


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _summary() -> dict:
    rows = []
    for seed in (1, 2, 3):
        for variant_index, variant in enumerate(("notool", "forced", "benefit")):
            row = {"seed": seed, "variant": variant}
            for metric_index, (metric, _) in enumerate(CONTROLLER_METRICS):
                row[metric] = 0.1 * (metric_index + 1) + 0.01 * variant_index
            rows.append(row)
    return {
        "seeds": [1, 2, 3],
        "per_seed": rows,
        "comparisons": {"benefit_vs_notool": {}, "benefit_vs_forced": {}},
        "protocol": {"split": "test", "test_assets_read": True},
    }


def _comparison(offset: float) -> dict:
    candidate = {
        metric: 0.2 + 0.01 * index
        for index, (metric, _) in enumerate(WRITEBACK_METRICS)
    }
    baseline = {metric: value + offset for metric, value in candidate.items()}
    return {
        "model_seeds": [1, 2, 3],
        "baseline": baseline,
        "candidate": candidate,
        "paired_delta": {
            metric: {
                "delta": candidate[metric] - baseline[metric],
                "ci95_low": -0.02,
                "ci95_high": 0.02,
            }
            for metric, _ in WRITEBACK_METRICS
        },
        "split": "test",
        "test_assets_read": True,
    }


def _fixture(tmp_path: Path) -> tuple[Path, ...]:
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "\n".join(
            (
                "schema_version: sn7-step0-frozen-registry-v2",
                "status: validation_promoted_test_ready_protocol_recovery",
                "protocol:",
                "  test_episode_construction:",
                "    max_per_operation: 20",
                '    recovery_basis: "pre-existing validation cap"',
            )
        )
        + "\n",
        encoding="utf-8",
    )
    ledger = _write(
        tmp_path / "ledger.json",
        {
            "schema_version": "activemap-frozen-test-access-v1",
            "status": "complete",
            "returncode": 0,
            "registry_sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
            "command": ["bash", "scripts/run_sn7_step0_frozen_test_v3_cap20.sh"],
        },
    )
    run_root = tmp_path / "run"
    _write(
        run_root / "COMPLETE.json",
        {
            "schema_version": "sn7-step0-frozen-test-complete-v2",
            "test_assets_read": True,
            "promotion_passed": False,
        },
    )
    _write(run_root / "three_policy_summary.json", _summary())
    _write(run_root / "benefit_vs_notool.json", _comparison(-0.01))
    _write(run_root / "benefit_vs_forced.json", _comparison(0.01))
    _write(
        run_root / "writeback_promotion.json",
        {
            "promote": False,
            "claim_boundary": "Three-seed frozen-test outcome.",
            "test_assets_read": True,
        },
    )
    return registry, ledger, run_root, tmp_path / "tables"


def test_recovery_adapter_preserves_frozen_metric_reporter(tmp_path: Path) -> None:
    registry, ledger, run_root, output = _fixture(tmp_path)
    manifest = export(registry, ledger, run_root, output)

    assert manifest["protocol_recovery"] == "v3_cap20_launcher_only"
    assert len(manifest["metric_reporter_sha256"]) == 64
    assert len(manifest["compatibility_adapter_sha256"]) == 64
    assert manifest["reporting_policy"] == "report_regardless_of_promotion_outcome"


def test_recovery_adapter_rejects_wrong_launcher(tmp_path: Path) -> None:
    registry, ledger, run_root, output = _fixture(tmp_path)
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    payload["command"] = ["bash", "scripts/run_sn7_step0_frozen_test_v2.sh"]
    _write(ledger, payload)

    with pytest.raises(ValueError, match="cap20 recovery launcher"):
        export(registry, ledger, run_root, output)


def test_recovery_adapter_accepts_split_schema_resume_launcher(tmp_path: Path) -> None:
    registry, ledger, run_root, output = _fixture(tmp_path)
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    payload["command"] = [
        "bash",
        "scripts/resume_sn7_step0_frozen_test_v3_cap20_after_split_fix.sh",
    ]
    _write(ledger, payload)

    manifest = export(registry, ledger, run_root, output)

    assert manifest["protocol_recovery"] == "v3_cap20_launcher_only"


def test_recovery_adapter_rejects_registry_without_cap(tmp_path: Path) -> None:
    registry, ledger, run_root, output = _fixture(tmp_path)
    registry.write_text(
        registry.read_text(encoding="utf-8").replace("max_per_operation: 20", ""),
        encoding="utf-8",
    )
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    payload["registry_sha256"] = hashlib.sha256(registry.read_bytes()).hexdigest()
    _write(ledger, payload)

    with pytest.raises(ValueError, match="audited cap20 recovery contract"):
        export(registry, ledger, run_root, output)
